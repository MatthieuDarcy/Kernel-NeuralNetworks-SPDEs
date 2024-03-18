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

"""
This file consider both sin and cos eigen functions, while
NonlinearElliptic2D_Faster.py only consider sin eigen functions. 
"""

class NonlinearElliptic2D(object):
    """
    solve the equation os -Delta u + u^3 = f,
    where u = \\sum{i=1}^m \\sum_{j=1}^m i^\\alpha * j^\\beta * \\xi_i * \\xi_j * \\phi_i(x) * \\phi_j(x),
    where \\sqrt{2}\\sin(j\\pi x)
    """
    def __init__(self, kernel, domain, alpha, beta, m, N, s, gamma, deg):
        """
        Input:

        kernel: object
            the kernel which generates the RKHS that we find the solution u
        domain: object
            the domain of the problem
        alpha:  real number
            the power in the definition of the function u
        m:      positive integer
            m^2 is the number of series truncated for the function u
        N:      positive integer
            N^2 is the number of eigenfunctions
        s:      real number
            the order of the function space H^s, which represents the function space of the rough data
        gamma:   positive real number
            the regularization parameter in front of the RKHS norm of u
        deg:    real number
            the number of gauss quadrature points used for integration
        """

        self.kernel = kernel
        self.domain = domain
        self.alpha = alpha
        self.beta = beta
        self.mm = m ** 2
        self.NN = 2 * N ** 2
        self.s = s
        gauss_samples, gauss_weights = np.polynomial.legendre.leggauss(deg)
        gauss_samples = (gauss_samples + 1) / 2
        gauss_weights = gauss_weights / 2
        gXX, gYY = jnp.meshgrid(gauss_samples, gauss_samples, indexing='ij')
        self.gauss_samples = jnp.concatenate((jnp.reshape(gXX.flatten(), (-1, 1)), jnp.reshape(gYY.flatten(), (-1, 1))), axis=1)
        wXX, wYY = jnp.meshgrid(gauss_weights, gauss_weights, indexing='ij')
        self.gauss_weights = wXX.flatten() * wYY.flatten()

        self.Q = len(self.gauss_samples) # the number of gauss quadrature points
        self.gamma = gamma

        scalers = jnp.array(jnp.arange(0, N) + 1)
        sXX, sYY = jnp.meshgrid(scalers, scalers, indexing='ij')
        self.eigen_func_indices = jnp.concatenate((jnp.reshape(sXX.flatten(), (-1, 1)), jnp.reshape(sYY.flatten(), (-1, 1))), axis=1)

        scalers = jnp.array(jnp.arange(0, m) + 1)
        sXX, sYY = jnp.meshgrid(scalers, scalers, indexing='ij')
        self.u_func_indices = jnp.concatenate((jnp.reshape(sXX.flatten(), (-1, 1)), jnp.reshape(sYY.flatten(), (-1, 1))), axis=1)

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.sample_xi(self.mm)
        self.prepare_data()

    def tau(self, u):
        return u**3

    def dtau(self, u):
        return 3 * u ** 2

    def sample_xi(self, m):
        self.xi = np.concatenate((np.random.uniform(0, 1, (m, 1)), np.random.uniform(0, 1, (m, 1))), axis=1)

    def prepare_data(self):
        self.gbdr = vmap(self.u)(self.samples[self.M_Omega:self.M, 0], self.samples[self.M_Omega:self.M, 1])

        eigen_func1 = lambda _i, _j, _qi, _qj, _wq: jnp.sin(_i * jnp.pi * _qi + _j * jnp.pi * _qj) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, None, 0, 0, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, 0, None, None, None))

        eigen_func1_vals = eigen_func1(self.eigen_func_indices[:, 0], self.eigen_func_indices[:, 1],
                                       self.gauss_samples[:, 0], self.gauss_samples[:, 1], self.gauss_weights)

        eigen_func2 = lambda _i, _j, _qi, _qj, _wq: jnp.cos(
            _i * jnp.pi * _qi + _j * jnp.pi * _qj) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, None, 0, 0, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, 0, None, None, None))

        eigen_func2_vals = eigen_func2(self.eigen_func_indices[:, 0], self.eigen_func_indices[:, 1],
                                       self.gauss_samples[:, 0], self.gauss_samples[:, 1], self.gauss_weights)

        self.eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=0)

        self.fs = vmap(self.f)(self.gauss_samples[:, 0], self.gauss_samples[:, 1])

        evfunc = lambda i, j: (i * jnp.pi) ** 2 + (j * jnp.pi) ** 2

        eigen_vals = evfunc(self.eigen_func_indices[:, 0], self.eigen_func_indices[:, 1])
        self.eigen_vals = jnp.concatenate((eigen_vals, eigen_vals))

    def u(self, x, y):
        func = lambda xi_i, xi_j, i, j: i ** self.alpha * xi_i \
                                         * j ** self.beta * xi_j * jnp.sqrt(2) * \
                                         jnp.sin(i * jnp.pi * x) * jnp.sqrt(2) \
                                         * jnp.sin(j * jnp.pi * y)
        vals = func(self.xi[:, 0], self.xi[:, 1], self.u_func_indices[:, 0], self.u_func_indices[:, 1])
        return jnp.sum(vals)
        # return jnp.sin(jnp.pi * x) + jnp.sin(2 * jnp.pi * y)

    def f(self, x, y):
        return -grad(grad(self.u, 0), 0)(x, y) - grad(grad(self.u, 1), 1)(x, y) + self.tau(self.u(x, y))

    def global_loss(self, uweights, hs, L):
        u_gauss = self.__eval_u_at_gauss_quadrature_points(uweights, hs)
        Delta_u_gauss = self.__eval_Delta_u_at_gauss_quadrature_points(uweights, hs)

        udata = - Delta_u_gauss + vmap(self.tau)(u_gauss) - self.fs
        udata = jnp.dot(self.eigen_func_vals, udata)
        p1 = jnp.sum(udata ** 2 * self.eigen_vals ** (-self.s))
        Uobs = jnp.dot(L, jnp.dot(L.T, uweights))
        p2 = self.gamma * jnp.dot(Uobs, uweights)
        return p1 + p2
    def loss(self, z, data, L):
        z1 = z[:self.M_Omega]
        z2 = z[self.M_Omega:self.M_Omega + self.NN]
        z3 = z[self.M_Omega + self.NN:]

        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.append(zz, z3)

        zz = jsp.linalg.solve_triangular(L, zz, lower=True)
        return self.gamma * jnp.dot(zz, zz) + jnp.sum((-z2 + z3 + data)**2 * self.eigen_vals ** (-self.s))

    def grad_loss(self, z, data, L):
        return grad(self.loss)(z, data, L)

    def GN_loss(self, z, zold, data, L):
        z1 = z[:self.M_Omega]
        z2 = z[self.M_Omega:self.M_Omega + self.NN]
        z3 = z[self.M_Omega + self.NN:]

        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.append(zz, z3)

        zz = jsp.linalg.solve_triangular(L, zz, lower=True)
        return self.gamma * jnp.dot(zz, zz) + jnp.sum((-z2 + z3 + data) ** 2 * self.eigen_vals ** (-self.s))

    def Hessian_GN(self, z, zold, data, L):
        return hessian(self.GN_loss)(z, zold, data, L)

    def train(self, cfg):
        self.__build_theta_invariants(self.samples, self.M, self.M_Omega, self.NN, self.gauss_samples, self.Q,
                                    self.gauss_weights, self.eigen_func_indices)
        self.__build_u_invariants()

        # set the initial value of z
        uk_weights = jnp.zeros(self.M + 2 * self.NN)
        uk_hs = jnp.zeros(self.Q)

        uk_theta = self.build_theta(self.samples, self.M, self.M_Omega, self.NN, uk_hs, self.gauss_samples, self.Q,
                                 self.gauss_weights, self.eigen_func_indices)
        uk_nuggets = self.build_nuggets(uk_theta, self.M, self.NN)
        uk_gram = uk_theta + cfg.nugget * uk_nuggets
        uk_L = jnp.linalg.cholesky(uk_gram)
        uk_z = jnp.zeros(self.M + 2 * self.NN)

        loss_hist = []  # history of loss function values
        current_loss = self.global_loss(uk_weights, uk_hs, uk_L)
        loss_hist.append(current_loss)

        error = 1
        iter = 0

        while iter < cfg.epoch and error > cfg.tol:
            uk_gauss = self.__eval_u_at_gauss_quadrature_points(uk_weights, uk_hs)

            uk1_hs = vmap(self.dtau)(uk_gauss)
            uk1_data = vmap(self.tau)(uk_gauss) - vmap(self.dtau)(uk_gauss) * uk_gauss - self.fs
            uk1_data = jnp.dot(self.eigen_func_vals, uk1_data)

            uk1_theta = self.build_theta(self.samples, self.M, self.M_Omega, self.NN, uk1_hs, self.gauss_samples, self.Q,
                                        self.gauss_weights, self.eigen_func_indices)

            uk1_nuggets = self.build_nuggets(uk1_theta, self.M, self.NN)
            uk1_gram = uk1_theta + cfg.nugget * uk1_nuggets

            uk1_L = jnp.linalg.cholesky(uk1_gram)

            zl = np.zeros(self.M_Omega + 2 * self.NN)
            coeffs = self.Hessian_GN(zl, zl, uk1_data, uk1_L)

            zl = jnp.linalg.solve(coeffs, -self.grad_loss(zl, uk1_data, uk1_L))

            z1 = zl[:self.M_Omega]
            z2 = zl[self.M_Omega:self.M_Omega + self.NN]
            z3 = zl[self.M_Omega + self.NN:]

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

    def __build_theta_invariants(self, x, M, M_Omega, N, q, Q, qweights, eigen_func_indices):
        """
                Input:

                q:          matrix
                        The coordinates of gauss quadrature points, it is a matrix with Q rows and 2 columns,
                        each column contains coordinates in each axis.

                Q:          integer
                        The number of gauss quadrature points.

                qweights:   list
                        A vector with Q elements. The gauss quadrature weights associated with each point.

                eigen_func_indices: list
                        The indices of eigen functions, it is a matrix with 2 columns,
                        each column contains indices associated with each axis. For instance, if the k-th row
                        contains an element [i, j], then, the basis function is sin(i * pi * x + j * pi * y).
                """
        theta = jnp.zeros((M + N, M + N))

        x0 = jnp.reshape(x[:, 0], (M, 1))
        q0 = jnp.reshape(q[:, 0], (Q, 1))

        x1 = jnp.reshape(x[:, 1], (M, 1))
        q1 = jnp.reshape(q[:, 1], (Q, 1))

        x0x0v = jnp.tile(x0, M).flatten()
        x0x0h = jnp.tile(jnp.transpose(x0), (M, 1)).flatten()

        x1x1v = jnp.tile(x1, M).flatten()
        x1x1h = jnp.tile(jnp.transpose(x1), (M, 1)).flatten()

        x0q0v = jnp.tile(x0, Q).flatten()
        x0q0h = jnp.tile(jnp.transpose(q0), (M, 1)).flatten()

        x1q1v = jnp.tile(x1, Q).flatten()
        x1q1h = jnp.tile(jnp.transpose(q1), (M, 1)).flatten()

        q0q0v = jnp.tile(q0, Q).flatten()
        q0q0h = jnp.tile(jnp.transpose(q0), (Q, 1)).flatten()

        q1q1v = jnp.tile(q1, Q).flatten()
        q1q1h = jnp.tile(jnp.transpose(q1), (Q, 1)).flatten()

        # Acting Dirac_x, Dirac_y on K
        # K(x, y)
        val = vmap(lambda _x0, _x1, _y0, _y1: self.kernel.kappa(_x0, _x1, _y0, _y1))(x0x0v, x1x1v, x0x0h, x1x1h)
        theta = theta.at[:M, :M].set(jnp.reshape(val, (M, M)))

        # Acting Dirac_x, L1_y on K
        # \Delta_y K(x, q)
        val = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.Delta_y_kappa(_x0, _x1, _q0, _q1))(x0q0v, x1q1v, x0q0h, x1q1h)
        mtx = np.reshape(val, (M, Q))

        tmp = jnp.dot(mtx, self.eigen_func_vals.T)
        theta = theta.at[:M, M:M + N].set(tmp)
        theta = theta.at[M:M + N, :M].set(tmp.T)

        #######################################################################################
        # Acting L1_x, L1_y on K
        # \Delta_x \Delta_y K(q, q)
        val = vmap(lambda _qx0, _qx1, _qy0, _qy1: self.kernel.Delta_x_Delta_y_kappa(_qx0, _qx1, _qy0, _qy1))(q0q0v,
                                                                                                             q1q1v,
                                                                                                             q0q0h,
                                                                                                             q1q1h)
        mtx = jnp.reshape(val, (Q, Q))

        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(mtx, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(self.eigen_func_vals, self.eigen_func_vals)

        theta = theta.at[M:M + N, M:M + N].set(tmp)

        self.__theta_invariant = theta

        val = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.kappa(_x0, _x1, _q0, _q1))(x0q0v, x1q1v, x0q0h, x1q1h)
        mtx = np.reshape(val, (M, Q))
        self.__xqkappa = mtx

        val = vmap(lambda _p0, _p1, _q0, _q1: self.kernel.Delta_x_kappa(_p0, _p1, _q0, _q1))(q0q0v, q1q1v, q0q0h, q1q1h)
        mtx = np.reshape(val, (Q, Q))
        self.__qqDeltakappa = mtx

        val = vmap(lambda _p0, _p1, _q0, _q1: self.kernel.kappa(_p0, _p1, _q0, _q1))(q0q0v, q1q1v, q0q0h, q1q1h)
        mtx = np.reshape(val, (Q, Q))
        self.__qqkappa = mtx

    def build_theta(self, x, M, M_Omega, N, hs, q, Q, qweights, eigen_func_indices):
        """
        Input:

        q:          matrix
                The coordinates of gauss quadrature points, it is a matrix with Q rows and 2 columns,
                each column contains coordinates in each axis.

        Q:          integer
                The number of gauss quadrature points.

        qweights:   list
                A vector with Q elements. The gauss quadrature weights associated with each point.

        eigen_func_indices: list
                The indices of eigen functions, it is a matrix with 2 columns,
                each column contains indices associated with each axis. For instance, if the k-th row
                contains an element [i, j], then, the basis function is sin(i * pi * x + j * pi * y).
        """
        theta = jnp.zeros((M + 2 * N, M + 2 * N))

        # Setting the invariant parts
        theta = theta.at[:M + N, :M + N].set(self.__theta_invariant)
        # Acting Dirac_x, L2_y on K
        # K(x, q)
        lop_vals = vmap(lambda _t: hs * _t)(self.eigen_func_vals).T

        tmp = jnp.dot(self.__xqkappa, lop_vals)
        theta = theta.at[:M, M + N:M + 2 * N].set(tmp)
        theta = theta.at[M + N:M + 2 * N, :M].set(tmp.T)

        #######################################################################################
        # Acting L1_x, L2_y on K
        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(self.__qqDeltakappa, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(self.eigen_func_vals, lop_vals.T)

        theta = theta.at[M:M + N, M + N:M + 2 * N].set(tmp)
        theta = theta.at[M + N:M + 2 * N, M:M + N].set(tmp.T)

        # Acting L2_x, L2_y on K
        func = (lambda ar, ac: jnp.dot(ar, jnp.dot(self.__qqkappa, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = (vmap(func, in_axes=(0, None)))
        tmp = func(lop_vals.T, lop_vals.T)

        theta = theta.at[M + N:M + 2 * N, M + N:M + 2 * N].set(tmp)

        return theta

    # build Nuggets
    def build_nuggets(self, theta, M, N):
        trace11 = np.trace(theta[0:M, 0:M])
        trace22 = np.trace(theta[M:M + N, M:M + N])
        trace33 = np.trace(theta[M + N:, M + N:])
        ratio = [trace22 / trace11, np.maximum(trace33 / trace11, 1e-10)]
        r_diag = np.concatenate((np.ones((1, M)), ratio[0] * np.ones((1, N)), ratio[1] * np.ones((1, N))), axis=1)
        r = np.diag(r_diag[0])
        return r

    def fit(self, nx):
        return self.eval_u(nx, self.weights, self.hs)

    def __build_u_invariants(self):
        nx = self.gauss_samples

        x = self.samples
        x0 = jnp.reshape(x[:, 0], (self.M, 1))
        x1 = jnp.reshape(x[:, 1], (self.M, 1))

        nxl = len(nx)
        nx0 = jnp.reshape(nx[:, 0], (nxl, 1))
        nx1 = jnp.reshape(nx[:, 1], (nxl, 1))
        nxx0v = jnp.tile(nx0, self.M).flatten()
        nxx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()
        nxx1v = jnp.tile(nx1, self.M).flatten()
        nxx1h = jnp.tile(np.transpose(x1), (nxl, 1)).flatten()

        q0 = jnp.reshape(self.gauss_samples[:, 0], (self.Q, 1))
        q1 = jnp.reshape(self.gauss_samples[:, 1], (self.Q, 1))

        nx0q0v = jnp.tile(nx0, self.Q).flatten()
        nx0q0h = jnp.tile(np.transpose(q0), (nxl, 1)).flatten()
        nx1q1v = jnp.tile(nx1, self.Q).flatten()
        nx1q1h = jnp.tile(np.transpose(q1), (nxl, 1)).flatten()

        ##################################################################################
        mtx = jnp.zeros((nxl, self.M + self.NN))

        val0 = vmap(lambda _x0, _x1, _y0, _y1: self.kernel.kappa(_x0, _x1, _y0, _y1))(nxx0v, nxx1v, nxx0h, nxx1h)
        mtx = mtx.at[0:nxl, :self.M].set(jnp.reshape(val0, (nxl, self.M)))

        val1 = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.Delta_y_kappa(_x0, _x1, _q0, _q1))(nx0q0v, nx1q1v, nx0q0h,
                                                                                              nx1q1h)
        tmp = jnp.reshape(val1, (nxl, self.Q))

        mtx = mtx.at[:nxl, self.M:self.M + self.NN].set(jnp.dot(tmp, self.eigen_func_vals.T))
        self.u_invariants = mtx

        ##################################################################################
        mtx = jnp.zeros((nxl, self.M + self.NN))

        val0 = vmap(lambda _x0, _x1, _y0, _y1: self.kernel.Delta_x_kappa(_x0, _x1, _y0, _y1))(nxx0v, nxx1v, nxx0h,
                                                                                              nxx1h)
        mtx = mtx.at[0:nxl, :self.M].set(jnp.reshape(val0, (nxl, self.M)))

        val1 = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.Delta_x_Delta_y_kappa(_x0, _x1, _q0, _q1))(nx0q0v, nx1q1v,
                                                                                                      nx0q0h, nx1q1h)
        tmp = np.reshape(val1, (nxl, self.Q))

        mtx = mtx.at[:nxl, self.M:self.M + self.NN].set(jnp.dot(tmp, self.eigen_func_vals.T))
        self.Delta_u_invariants = mtx

    def __eval_u_at_gauss_quadrature_points(self, uweights, hs):
        nx = self.gauss_samples
        nxl = len(nx)
        mtx = jnp.zeros((nxl, self.M + 2 * self.NN))

        mtx = mtx.at[0:nxl, :self.M + self.NN].set(self.u_invariants)
        # Acting Dirac_x, L2_y on K
        # K(x, q)
        lop_vals = vmap(lambda _t: hs * _t)(self.eigen_func_vals).T
        mtx = mtx.at[:nxl, self.M + self.NN:].set(jnp.dot(self.__qqkappa, lop_vals))

        u = mtx.dot(uweights)

        return u

    def __eval_Delta_u_at_gauss_quadrature_points(self, uweights, hs):
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
        nx = self.gauss_samples
        nxl = len(nx)
        mtx = jnp.zeros((nxl, self.M + 2 * self.NN))

        mtx = mtx.at[:nxl, :self.M + self.NN].set(self.Delta_u_invariants)
        # Acting Dirac_x, L2_y on K
        # K(x, q)
        lop_vals = vmap(lambda _t: hs * _t)(self.eigen_func_vals).T

        mtx = mtx.at[:nxl, self.M + self.NN:].set(jnp.dot(self.__qqDeltakappa, lop_vals))

        u = mtx.dot(uweights)

        return u

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
        x0 = jnp.reshape(x[:, 0], (self.M, 1))
        x1 = jnp.reshape(x[:, 1], (self.M, 1))

        nxl = len(nx)
        nx0 = jnp.reshape(nx[:, 0], (nxl, 1))
        nx1 = jnp.reshape(nx[:, 1], (nxl, 1))
        nxx0v = jnp.tile(nx0, self.M).flatten()
        nxx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()
        nxx1v = jnp.tile(nx1, self.M).flatten()
        nxx1h = jnp.tile(np.transpose(x1), (nxl, 1)).flatten()

        q0 = jnp.reshape(self.gauss_samples[:, 0], (self.Q, 1))
        q1 = jnp.reshape(self.gauss_samples[:, 1], (self.Q, 1))

        nx0q0v = jnp.tile(nx0, self.Q).flatten()
        nx0q0h = jnp.tile(np.transpose(q0), (nxl, 1)).flatten()
        nx1q1v = jnp.tile(nx1, self.Q).flatten()
        nx1q1h = jnp.tile(np.transpose(q1), (nxl, 1)).flatten()

        mtx = jnp.zeros((nxl, self.M + 2 * self.NN))

        val0 = vmap(lambda _x0, _x1, _y0, _y1: self.kernel.kappa(_x0, _x1, _y0, _y1))(nxx0v, nxx1v, nxx0h, nxx1h)
        mtx = mtx.at[0:nxl, :self.M].set(jnp.reshape(val0, (nxl, self.M)))

        val1 = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.Delta_y_kappa(_x0, _x1, _q0, _q1))(nx0q0v, nx1q1v, nx0q0h, nx1q1h)
        tmp = jnp.reshape(val1, (nxl, self.Q))

        mtx = mtx.at[:nxl, self.M:self.M + self.NN].set(jnp.dot(tmp, self.eigen_func_vals.T))

        # Acting Dirac_x, L2_y on K
        # K(x, q)
        val = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.kappa(_x0, _x1, _q0, _q1))(nx0q0v, nx1q1v, nx0q0h, nx1q1h)
        tmp = jnp.reshape(val, (nxl, self.Q))

        lop_vals = vmap(lambda _t: hs * _t)(self.eigen_func_vals).T

        mtx = mtx.at[:nxl, self.M + self.NN:].set(jnp.dot(tmp, lop_vals))

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
        x0 = jnp.reshape(x[:, 0], (self.M, 1))
        x1 = jnp.reshape(x[:, 1], (self.M, 1))

        nxl = len(nx)
        nx0 = jnp.reshape(nx[:, 0], (nxl, 1))
        nx1 = jnp.reshape(nx[:, 1], (nxl, 1))
        nxx0v = jnp.tile(nx0, self.M).flatten()
        nxx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()
        nxx1v = jnp.tile(nx1, self.M).flatten()
        nxx1h = jnp.tile(np.transpose(x1), (nxl, 1)).flatten()

        q0 = jnp.reshape(self.gauss_samples[:, 0], (self.Q, 1))
        q1 = jnp.reshape(self.gauss_samples[:, 1], (self.Q, 1))

        nx0q0v = jnp.tile(nx0, self.Q).flatten()
        nx0q0h = jnp.tile(np.transpose(q0), (nxl, 1)).flatten()
        nx1q1v = jnp.tile(nx1, self.Q).flatten()
        nx1q1h = jnp.tile(np.transpose(q1), (nxl, 1)).flatten()

        mtx = jnp.zeros((nxl, self.M + 2 * self.NN))

        val0 = vmap(lambda _x0, _x1, _y0, _y1: self.kernel.Delta_x_kappa(_x0, _x1, _y0, _y1))(nxx0v, nxx1v, nxx0h, nxx1h)
        mtx = mtx.at[0:nxl, :self.M].set(jnp.reshape(val0, (nxl, self.M)))

        val1 = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.Delta_x_Delta_y_kappa(_x0, _x1, _q0, _q1))(nx0q0v, nx1q1v, nx0q0h, nx1q1h)
        tmp = np.reshape(val1, (nxl, self.Q))

        mtx = mtx.at[:nxl, self.M:self.M + self.NN].set(jnp.dot(tmp, self.eigen_func_vals.T))

        # Acting Dirac_x, L2_y on K
        # K(x, q)
        val = vmap(lambda _x0, _x1, _q0, _q1: self.kernel.Delta_x_kappa(_x0, _x1, _q0, _q1))(nx0q0v, nx1q1v, nx0q0h, nx1q1h)
        tmp = np.reshape(val, (nxl, self.Q))

        lop_vals = vmap(lambda _t: hs * _t)(self.eigen_func_vals).T

        mtx = mtx.at[:nxl, self.M + self.NN:].set(jnp.dot(tmp, lop_vals))

        u = mtx.dot(uweights)

        return u
