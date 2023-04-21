import jax.numpy as jnp
from jax import grad, hessian
from jax import vmap, jit
import jax.ops as jop
import jax.scipy as jsp
from SolvingRoughPDEs.utilities.domain import *
from functools import partial
import time
import numpy as np

from jax.config import config
import jax.scipy as jsp
config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

class NonlinearElliptic1D(object):
    """
    solve the equation os -Delta u + u^3 = f,
    where u = \\sum_{j=1}^m j^\\alpha, \\xi_j \\phi_j(x), where \\sqrt{2}\\sin(j\\pi x)
    """
    def __init__(self, kernel, domain, alpha, m, N, s, gamma):
        """
        Input:

        kernel: object
            the kernel which generates the RKHS that we find the solution u
        domain: object
            the domain of the problem
        alpha:  real number
            the power in the definition of the function u
        m:      positive integer
            the number of series truncated for the function u
        N:      positive integer
            2 * N is the number of eigenfunctions
        s:      real number
            the order of the function space H^s, which represents the function space of the rough data
        gamma:   positive real number
            the regularization parameter in front of the RKHS norm of u
        """

        self.kernel = kernel
        self.domain = domain
        self.alpha = alpha
        self.m = m
        self.sample_f(m)
        self.N = N
        self.s = s
        self.gauss_samples, self.gauss_weights = np.polynomial.legendre.leggauss(100)
        self.gauss_samples = (self.gauss_samples + 1) / 2
        self.gauss_weights = self.gauss_weights / 2
        self.Q = len(self.gauss_samples) # the number of gauss quadrature points
        self.gamma = gamma

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.prepare_data()


    def tau(self, u):
        return u**3

    def dtau(self, u):
        return 3 * u ** 2

    def sample_f(self, m):
        self.fxi = jnp.array(np.random.normal(0, 1, m))

    def prepare_data(self):
        self.gbdr = vmap(self.u)(jnp.array([0, 1]))

        eigen_func1 = lambda _i, _q, _wq: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, 0, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, None, None))

        eigen_func2 = lambda _i, _q, _wq: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, 0, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, None, None))

        scalers = jnp.array(jnp.arange(0, self.N) + 1)
        eigen_func1_vals = eigen_func1(scalers, self.gauss_samples, self.gauss_weights)
        eigen_func2_vals = eigen_func2(scalers, self.gauss_samples, self.gauss_weights)

        self.eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=0)

        self.fs = vmap(self.f)(self.gauss_samples)

        lbdas = (jnp.array(jnp.arange(0, self.N) + 1) * jnp.pi) ** 2
        self.eigen_vals = jnp.concatenate((lbdas, lbdas))

    def u(self, x):
        scalers = jnp.array(jnp.arange(0, self.m) + 1)
        # components = vmap(lambda j, xi_j: j ** self.alpha * xi_j * jnp.sqrt(2) * (jnp.cos(4 * j * jnp.pi * x) + jnp.sin(j * jnp.pi * x)) / (j ** 2 * jnp.pi ** 2))(scalers, self.fxi)
        components = vmap(lambda j, xi_j: j ** self.alpha * xi_j * jnp.sqrt(2) * jnp.sin(j * jnp.pi * x) / (j ** 2 * jnp.pi ** 2))(scalers, self.fxi)
        return jnp.sum(components)
        # return jnp.sin(jnp.pi * x) #+ jnp.sin(2 * jnp.pi * x)

    def f(self, x):
        return -grad(grad(self.u, 0), 0)(x) + self.tau(self.u(x))

    def global_loss(self, uweights, hs, L):
        u_gauss = self.eval_u(self.gauss_samples, uweights, hs)
        Delta_u_gauss = self.eval_Delta_u(self.gauss_samples, uweights, hs)

        udata = - Delta_u_gauss + vmap(self.tau)(u_gauss) - self.fs
        udata = jnp.dot(self.eigen_func_vals, udata)
        p1 = jnp.sum(udata ** 2 * self.eigen_vals ** (-self.s))
        Uobs = jnp.dot(L, jnp.dot(L.T, uweights))
        p2 = self.gamma * jnp.dot(Uobs, uweights)
        return p1 + p2
    def loss(self, z, data, L):
        z1 = z[:self.M_Omega]
        z2 = z[self.M_Omega:self.M_Omega + 2 * self.N]
        z3 = z[self.M_Omega + 2 * self.N:]

        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.append(zz, z3)

        zz = jsp.linalg.solve_triangular(L, zz, lower=True)
        return self.gamma * jnp.dot(zz, zz) + jnp.sum((-z2 + z3 + data)**2 * self.eigen_vals ** (-self.s))

    def grad_loss(self, z, data, L):
        return grad(self.loss)(z, data, L)

    def GN_loss(self, z, zold, data, L):
        z1 = z[:self.M_Omega]
        z2 = z[self.M_Omega:self.M_Omega + 2 * self.N]
        z3 = z[self.M_Omega + 2 * self.N:]

        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.append(zz, z3)

        zz = jsp.linalg.solve_triangular(L, zz, lower=True)
        return self.gamma * jnp.dot(zz, zz) + jnp.sum((-z2 + z3 + data) ** 2 * self.eigen_vals ** (-self.s))

    def Hessian_GN(self, z, zold, data, L):
        return hessian(self.GN_loss)(z, zold, data, L)

    def train(self, cfg):
        # set the initial value of z
        uk_weights = jnp.zeros(self.M + 4 * self.N)
        uk_hs = jnp.zeros(self.Q)

        uk_theta = self.build_theta(self.samples, self.M, self.M_Omega, self.N, self.gauss_samples, self.Q,
                                 self.gauss_weights, uk_hs)
        uk_nuggets = self.build_nuggets(uk_theta, self.M, self.N)
        uk_gram = uk_theta + cfg.nugget * uk_nuggets
        uk_L = jnp.linalg.cholesky(uk_gram)
        uk_z = jnp.zeros(self.M + 4 * self.N)

        loss_hist = []  # history of loss function values
        current_loss = self.global_loss(uk_weights, uk_hs, uk_L)
        loss_hist.append(current_loss)

        error = 1
        iter = 0
        while iter < cfg.epoch and error > cfg.tol:
            uk_gauss = self.eval_u(self.gauss_samples, uk_weights, uk_hs)

            uk1_hs = vmap(self.dtau)(uk_gauss)
            uk1_data = vmap(self.tau)(uk_gauss) - vmap(self.dtau)(uk_gauss) * uk_gauss - self.fs
            uk1_data = jnp.dot(self.eigen_func_vals, uk1_data)

            uk1_theta = self.build_theta(self.samples, self.M, self.M_Omega, self.N, self.gauss_samples, self.Q,
                                        self.gauss_weights, uk1_hs)
            uk1_nuggets = self.build_nuggets(uk1_theta, self.M, self.N)
            uk1_gram = uk1_theta + cfg.nugget * uk1_nuggets
            uk1_L = jnp.linalg.cholesky(uk1_gram)

            zl = np.zeros(self.M_Omega + 4 * self.N)
            coeffs = self.Hessian_GN(zl, zl, uk1_data, uk1_L)
            zl = jnp.linalg.solve(coeffs, -self.grad_loss(zl, uk1_data, uk1_L))

            z1 = zl[:self.M_Omega]
            z2 = zl[self.M_Omega:self.M_Omega + 2 * self.N]
            z3 = zl[self.M_Omega + 2 * self.N:]

            zz = jnp.append(z1, self.gbdr)
            zz = jnp.append(zz, z2)
            zz = jnp.append(zz, z3)

            uk1_weights = jsp.linalg.solve_triangular(uk1_L.T, jsp.linalg.solve_triangular(uk1_L, zz, lower=True), lower=False)
            uk1_z = zz

            error = jnp.linalg.norm(uk1_z - uk_z)

            current_loss = self.global_loss(uk1_weights, uk1_hs, uk1_L)

            loss_hist.append(current_loss)

            uk_weights = uk1_weights
            uk_hs = uk1_hs
            uk_z = uk1_z
            uk_L = uk1_L
            iter = iter + 1

            print("Outer Epoch {}, Outer Error {}, Outer Loss {}".format(iter, error, current_loss))


        self.num_iter = iter
        self.loss_hist = loss_hist
        self.weights = uk_weights
        self.hs = uk_hs

    def Knm(self, xl, Ml, M_Omegal, Nl, hsl, xr, Mr, M_Omegar, Nr, hsr, q, Q, qweights):
        theta = jnp.zeros((Ml + 4 * Nl, Mr + 4 * Nr))

        xl0 = np.reshape(xl, (Ml, 1))
        xr0 = np.reshape(xr, (Mr, 1))
        q0 = np.reshape(q, (Q, 1))

        xl0xr0v = jnp.tile(xl0, Mr).flatten()
        xl0xr0h = jnp.tile(np.transpose(xr0), (Ml, 1)).flatten()

        xl0q0v = jnp.tile(xl0, Q).flatten()
        xl0q0h = jnp.tile(jnp.transpose(q0), (Ml, 1)).flatten()

        q0xr0v = jnp.tile(q0, Mr).flatten()
        q0xr0h = jnp.tile(jnp.transpose(xr0), (Q, 1)).flatten()

        q0q0v = jnp.tile(q0, Q).flatten()
        q0q0h = jnp.tile(jnp.transpose(q0), (Q, 1)).flatten()

        # Acting Dirac_x, Dirac_y on K
        # K(x, y)
        val = vmap(lambda _x, _y: self.kernel.kappa(_x, _y))(xl0xr0v, xl0xr0h)
        theta = theta.at[:Ml, :Mr].set(jnp.reshape(val, (Ml, Mr)))

        # Acting Dirac_x, L1_y on K
        # \Delta_y K(x, q)
        val = vmap(lambda _x, _q: self.kernel.Delta_y_kappa(_x, _q))(xl0q0v, xl0q0h)
        mtx = np.reshape(val, (Ml, Q))

        eigen_func1r = lambda _q, _wq, _i: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1r = vmap(eigen_func1r, in_axes=(None, None, 0))
        eigen_func1r = vmap(eigen_func1r, in_axes=(0, 0, None))

        eigen_func2r = lambda _q, _wq, _i: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2r = vmap(eigen_func2r, in_axes=(None, None, 0))
        eigen_func2r = vmap(eigen_func2r, in_axes=(0, 0, None))

        scalers = jnp.array(jnp.arange(0, Nr) + 1)
        eigen_func1_valsr = eigen_func1r(q, qweights, scalers)
        eigen_func2_valsr = eigen_func2r(q, qweights, scalers)

        eigen_func_valsr = jnp.concatenate((eigen_func1_valsr, eigen_func2_valsr), axis=1)

        theta = theta.at[:Ml, Mr:Mr + 2 * Nr].set(mtx @ eigen_func_valsr)

        # Acting Dirac_x, L2_y on K
        # K(x, q)
        val = vmap(lambda _x, _q: self.kernel.kappa(_x, _q))(xl0q0v, xl0q0h)
        mtx = np.reshape(val, (Ml, Q))

        lop1r = lambda _h, _q, _wq, _i: _h * jnp.sin(_i * jnp.pi * _q) * _wq
        lop1r = vmap(lop1r, in_axes=(None, None, None, 0))
        lop1r = vmap(lop1r, in_axes=(0, 0, 0, None))

        lop2r = lambda _h, _q, _wq, _i: _h * jnp.cos(_i * jnp.pi * _q) * _wq
        lop2r = vmap(lop2r, in_axes=(None, None, None, 0))
        lop2r = vmap(lop2r, in_axes=(0, 0, 0, None))

        scalers = jnp.array(jnp.arange(0, Nr) + 1)
        lop1_valsr = lop1r(hsr, q, qweights, scalers)
        lop2_valsr = lop2r(hsr, q, qweights, scalers)

        lop_valsr = jnp.concatenate((lop1_valsr, lop2_valsr), axis=1)

        theta = theta.at[:Ml, Mr + 2 * Nr:Mr + 4 * Nr].set(mtx @ lop_valsr)

        #######################################################################################
        # Acting L1_x, Dirac_y on K
        # \Delta_x K(q, x)
        val = vmap(lambda _q, _x: self.kernel.Delta_x_kappa(_q, _x))(q0xr0v, q0xr0h)
        mtx = np.reshape(val, (Q, Mr))

        eigen_func1l = lambda _i, _q, _wq: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1l = vmap(eigen_func1l, in_axes=(None, 0, 0))
        eigen_func1l = vmap(eigen_func1l, in_axes=(0, None, None))

        eigen_func2l = lambda _i, _q, _wq: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2l = vmap(eigen_func2l, in_axes=(None, 0, 0))
        eigen_func2l = vmap(eigen_func2l, in_axes=(0, None, None))

        scalers = jnp.array(jnp.arange(0, Nl) + 1)
        eigen_func1_valsl = eigen_func1l(scalers, q, qweights)
        eigen_func2_valsl = eigen_func2l(scalers, q, qweights)

        eigen_func_valsl = jnp.concatenate((eigen_func1_valsl, eigen_func2_valsl), axis=0)

        theta = theta.at[Ml:Ml + 2 * Nl, 0:Mr].set(eigen_func_valsl @ mtx)

        # Acting L1_x, L1_y on K
        # \Delta_x \Delta_y K(q, q)
        val = vmap(lambda _qx, _qy: self.kernel.Delta_x_Delta_y_kappa(_qx, _qy))(q0q0v, q0q0h)
        mtx = jnp.reshape(val, (Q, Q))

        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(mtx, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(eigen_func_valsl, eigen_func_valsr.T)

        theta = theta.at[Ml:Ml + 2 * Nl, Mr:Mr + 2 * Nr].set(tmp)

        # Acting L1_x, L2_y on K
        val = vmap(lambda _p, _q: self.kernel.Delta_x_kappa(_p, _q))(q0q0v, q0q0h)
        mtx = np.reshape(val, (Q, Q))

        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(mtx, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(eigen_func_valsl, lop_valsr.T)

        theta = theta.at[Ml:Ml + 2 * Nl, Mr + 2 * Nr:Mr + 4 * Nr].set(tmp)
        #######################################################################################
        # Acting L2_x, Dirac_y on K
        # K(x, q)
        val = vmap(lambda _q, _x: self.kernel.kappa(_q, _x))(q0xr0v, q0xr0h)
        mtx = np.reshape(val, (Q, Mr))

        lop1l = lambda _h, _q, _wq, _i: _h * jnp.sin(_i * jnp.pi * _q) * _wq
        lop1l = vmap(lop1l, in_axes=(None, None, None, 0))
        lop1l = vmap(lop1l, in_axes=(0, 0, 0, None))

        lop2l = lambda _h, _q, _wq, _i: _h * jnp.cos(_i * jnp.pi * _q) * _wq
        lop2l = vmap(lop2l, in_axes=(None, None, None, 0))
        lop2l = vmap(lop2l, in_axes=(0, 0, 0, None))

        scalers = jnp.array(jnp.arange(0, Nl) + 1)
        lop1_valsl = lop1l(hsl, q, qweights, scalers)
        lop2_valsl = lop2l(hsl, q, qweights, scalers)

        lop_valsl = jnp.concatenate((lop1_valsl, lop2_valsl), axis=1)

        theta = theta.at[Ml + 2 * Nl:Ml + 4 * Nl, :Mr].set(lop_valsl.T @ mtx)

        # Acting L2_x, L1_y on K
        val = vmap(lambda _p, _q: self.kernel.Delta_y_kappa(_p, _q))(q0q0v, q0q0h)
        mtx = np.reshape(val, (Q, Q))

        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(mtx, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(lop_valsl.T, eigen_func_valsr.T)

        theta = theta.at[Ml + 2 * Nl:Ml + 4 * Nl, Mr:Mr + 2 * Nr].set(tmp)

        # Acting L2_x, L2_y on K
        val = vmap(lambda _p, _q: self.kernel.kappa(_p, _q))(q0q0v, q0q0h)
        mtx = np.reshape(val, (Q, Q))

        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(mtx, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(lop_valsl.T, lop_valsr.T)

        theta = theta.at[Ml + 2 * Nl:Ml + 4 * Nl, Mr + 2 * Nr:Mr + 4 * Nr].set(tmp)

        return theta


    def build_theta(self, x, M, M_Omega, N, q, Q, qweights, hs):
        '''
        Input:

        x:          list
            the sample points
        M:          integer
            the number of points, including those on the boundary
        M_Omega:    integer
            the number of points int the interior
        N:          integer
            2 * N is the number of eigenfunctions used, each eivenfunction is either sin(i\pi x) or cos(i\pi x)
        q:          list
            the Gauss-quadrature points
        Q:          integer
            the number of Gauss-quadrature points
        qweights:   list
            the Gauss-quadrature weights
        hs:         list
            the values of tau'(u^n) at quadrature points
        '''
        return self.Knm(x, M, M_Omega, N, hs, x, M, M_Omega, N, hs, q, Q, qweights)

    # build Nuggets
    def build_nuggets(self, theta, M, N):
        trace11 = np.trace(theta[0:M, 0:M])
        trace22 = np.trace(theta[M:M + 2 * N, M:M + 2 * N])
        trace33 = np.trace(theta[M + 2 * N:, M + 2 * N:])
        ratio = [trace22 / trace11, np.maximum(trace33 / trace11, 1e-10)]
        r_diag = np.concatenate((np.ones((1, M)), ratio[0] * np.ones((1, 2 * N)), ratio[1] * np.ones((1, 2 * N))), axis=1)
        r = np.diag(r_diag[0])
        return r

    def fit(self, nx):
        return self.eval_u(nx, self.weights, self.hs)

    def eval_u(self, nx, uweights, hs):
        """
        Evaluate a function u at points nx, where u = <K(x, \Phi), uweights> and uweights = K(\Phi, \Phi)^{-1}U,
        U is the observation of u.

        Input:

        nx:         list
            the list of points where we want to evaluate the values of functions
        uweights:    list
            the list of weights of a function
        hs:         list
            the values of tau'(u^n) at quadrature points

        Output:

        list
            the values of u at points nx
        """
        x = self.samples
        x0 = jnp.reshape(x, (self.M, 1))

        nxl = len(nx)
        nx0 = jnp.reshape(nx, (nxl, 1))
        nxx0v = jnp.tile(nx0, self.M).flatten()
        nxx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()

        q0 = jnp.reshape(self.gauss_samples, (self.Q, 1))

        nx0q0v = jnp.tile(nx0, self.Q).flatten()
        nx0q0h = jnp.tile(np.transpose(q0), (nxl, 1)).flatten()

        mtx = jnp.zeros((nxl, self.M + 4 * self.N))

        val0 = vmap(lambda _x, _y: self.kernel.kappa(_x, _y))(nxx0v, nxx0h)
        mtx = mtx.at[0:nxl, :self.M].set(jnp.reshape(val0, (nxl, self.M)))

        val1 = vmap(lambda _x, _y: self.kernel.Delta_y_kappa(_x, _y))(nx0q0v, nx0q0h)
        tmp = jnp.reshape(val1, (nxl, self.Q))

        eigen_func1 = lambda _q, _wq, _i: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, None, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, 0, None))

        eigen_func2 = lambda _q, _wq, _i: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, None, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, 0, None))

        scalers = jnp.array(jnp.arange(0, self.N) + 1)
        eigen_func1_vals = eigen_func1(self.gauss_samples, self.gauss_weights, scalers)
        eigen_func2_vals = eigen_func2(self.gauss_samples, self.gauss_weights, scalers)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=1)

        mtx = mtx.at[:nxl, self.M:self.M + 2 * self.N].set(tmp @ eigen_func_vals)

        # Acting Dirac_x, L2_y on K
        # K(x, q)
        val = vmap(lambda _x, _q: self.kernel.kappa(_x, _q))(nx0q0v, nx0q0h)
        tmp = jnp.reshape(val, (nxl, self.Q))

        lop1 = lambda _h, _q, _wq, _i: _h * jnp.sin(_i * jnp.pi * _q) * _wq
        lop1 = vmap(lop1, in_axes=(None, None, None, 0))
        lop1 = vmap(lop1, in_axes=(0, 0, 0, None))

        lop2 = lambda _h, _q, _wq, _i: _h * jnp.cos(_i * jnp.pi * _q) * _wq
        lop2 = vmap(lop2, in_axes=(None, None, None, 0))
        lop2 = vmap(lop2, in_axes=(0, 0, 0, None))

        scalers = jnp.array(jnp.arange(0, self.N) + 1)
        lop1_vals = lop1(hs, self.gauss_samples, self.gauss_weights, scalers)
        lop2_vals = lop2(hs, self.gauss_samples, self.gauss_weights, scalers)

        lop_vals = jnp.concatenate((lop1_vals, lop2_vals), axis=1)

        mtx = mtx.at[:nxl, self.M + 2 * self.N:].set(tmp @ lop_vals)

        u = mtx.dot(uweights)

        return u


    def eval_Delta_u(self, nx, uweights, hs):
        """
        Evaluate Delta u at points nx, where u = <K(x, \Phi), uweights> and uweights = K(\Phi, \Phi)^{-1}U,
        U is the observation of u.

        Input:

        nx:         list
            the list of points where we want to evaluate the values of functions
        uweights:   list
            the list of weights of a function
        hs:         list
            the values of tau'(u^n) at quadrature points
        Output:

        list
            the values of Delta u at points nx
        """
        x = self.samples
        x0 = np.reshape(x, (self.M, 1))

        nxl = len(nx)
        nx0 = np.reshape(nx, (nxl, 1))
        nxx0v = jnp.tile(nx0, self.M).flatten()
        nxx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()

        q0 = np.reshape(self.gauss_samples, (self.Q, 1))

        nx0q0v = jnp.tile(nx0, self.Q).flatten()
        nx0q0h = jnp.tile(np.transpose(q0), (nxl, 1)).flatten()

        mtx = np.zeros((nxl, self.M + 4 * self.N))

        val0 = vmap(lambda _x, _y: self.kernel.Delta_x_kappa(_x, _y))(nxx0v, nxx0h)
        mtx[0:nxl, :self.M] = np.reshape(val0, (nxl, self.M))

        val1 = vmap(lambda _x, _y: self.kernel.Delta_x_Delta_y_kappa(_x, _y))(nx0q0v, nx0q0h)
        tmp = np.reshape(val1, (nxl, self.Q))

        eigen_func1 = lambda _q, _wq, _i: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, None, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, 0, None))

        eigen_func2 = lambda _q, _wq, _i: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, None, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, 0, None))

        scalers = jnp.array(jnp.arange(0, self.N) + 1)
        eigen_func1_vals = eigen_func1(self.gauss_samples, self.gauss_weights, scalers)
        eigen_func2_vals = eigen_func2(self.gauss_samples, self.gauss_weights, scalers)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=1)

        mtx[:nxl, self.M:self.M + 2 * self.N] = tmp @ eigen_func_vals

        # Acting Dirac_x, L2_y on K
        # K(x, q)
        val = vmap(lambda _x, _q: self.kernel.Delta_x_kappa(_x, _q))(nx0q0v, nx0q0h)
        tmp = np.reshape(val, (nxl, self.Q))

        lop1 = lambda _h, _q, _wq, _i: _h * jnp.sin(_i * jnp.pi * _q) * _wq
        lop1 = vmap(lop1, in_axes=(None, None, None, 0))
        lop1 = vmap(lop1, in_axes=(0, 0, 0, None))

        lop2 = lambda _h, _q, _wq, _i: _h * jnp.cos(_i * jnp.pi * _q) * _wq
        lop2 = vmap(lop2, in_axes=(None, None, None, 0))
        lop2 = vmap(lop2, in_axes=(0, 0, 0, None))

        scalers = jnp.array(jnp.arange(0, self.N) + 1)
        lop1_vals = lop1(hs, self.gauss_samples, self.gauss_weights, scalers)
        lop2_vals = lop2(hs, self.gauss_samples, self.gauss_weights, scalers)

        lop_vals = jnp.concatenate((lop1_vals, lop2_vals), axis=1)

        mtx[:nxl, self.M + 2 * self.N:] = tmp @ lop_vals

        u = mtx.dot(uweights)

        return u
