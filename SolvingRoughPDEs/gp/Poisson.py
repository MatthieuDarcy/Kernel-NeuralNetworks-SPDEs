import jax.numpy as jnp
from jax import grad, hessian
from jax import vmap, jit
import jax.ops as jop
from SolvingRoughPDEs.utilities.domain import *
from functools import partial
import time
import numpy as np

from jax.config import config
import jax.scipy as jsp
config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

class Poisson(object):
    """
    solve the equation os -Delta u  = f,
    where f = \sum_{j=1}^m j^\alpha, \xi_j \phi_j(x), where \sqrt{2}\sin(j\pi x)
    """

    def __init__(self, kernel, domain, alpha, m, N, s, gamma):
        """
        Input:

        kernel: object
            the kernel which generates the RKHS that we find the solution u
        domain: object
            the domain of the problem
        alpha:  real number
            the power in the definition of the function f
        m:      positive integer
            the number of series truncated for the function f
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
        self.gauss_samples, self.gauss_weights = np.polynomial.legendre.leggauss(80)
        self.Q = len(self.gauss_samples) # the number of gauss quadrature points
        self.gamma = gamma

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.prepare_data()

    def sample_f(self, m):
        self.fxi = jnp.array(np.random.normal(0, 1, m))

    def prepare_data(self):
        self.gbdr = np.zeros(self.M - self.M_Omega)

        eigen_func1 = lambda _i, _q, _wq: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, 0, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, None, None))

        eigen_func2 = lambda _i, _q, _wq: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, 0, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, None, None))

        scalers = jnp.array(jnp.arange(0, N) + 1)
        eigen_func1_vals = eigen_func1(scalers, self.gauss_samples, self.gauss_weights)
        eigen_func2_vals = eigen_func2(scalers, self.gauss_samples, self.gauss_weights)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=0)

        fs = vmap(self.f)(self.gauss_samples)

        self.fs = jnp.dot(eigen_func_vals, fs)

        lbdas =  (jnp.array(jnp.arange(0, self.N) + 1) * jnp.pi) ** 2
        self.eigen_vals = jnp.concatenate((lbdas, lbdas))

    def u(self, x):
        pass

    def f(self, x):
        scalers = jnp.array(jnp.arange(0, self.m) + 1)
        components = vmap(lambda j, xi_j: j ** self.alpha * xi_j * jnp.sqrt(2) * jnp.sin(j * jnp.pi * x))(scalers, self.fxi)
        return jnp.sum(components)

    def loss(self, z):
        z1 = z[:self.M_Omega]
        z2 = z[self.M_Omega:]

        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.linalg.solve(self.L, zz)
        return self.gamma * jnp.dot(zz, zz) + jnp.sum((z + self.fs)**2 * self.eigen_vals ** (-self.s))

    def grad_loss(self, z):
        return grad(self.loss)(z)

    def GN_loss(self, z, zold):
        z1 = z[:self.M_Omega]
        z2 = z[self.M_Omega:]
        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.linalg.solve(self.L, zz)
        return self.gamma * jnp.dot(zz, zz) + jnp.sum((z + self.fs) ** 2 * self.eigen_vals ** (-self.s))

    def Hessian_GN(self, z, zold):
        return hessian(self.GN_loss)(z, zold)

    def build_gram(self, nugget = 1e-8):
        theta = self.build_theta(self.samples, self.M, self.M_Omega, self.N, self.gauss_samples, self.Q, self.gauss_weights)
        nuggets = self.build_nuggets(theta, self.M, self.Q)
        self.gram = theta + nugget * nuggets

    def gram_Cholesky(self):
        self.L = jnp.linalg.cholesky(self.gram)

    def train(self, cfg):
        error = 1
        iter = 0
        # set the initial value of z
        zl = np.random.rand(self.M_Omega + self.N)

        self.gram_Cholesky()

        loss_hist = []  # history of loss function values
        current_loss = self.loss(zl)
        loss_hist.append(current_loss)

        while error > cfg.tol and iter < cfg.epoch:
            coeffs = self.Hessian_GN(zl, zl)

            dz = jnp.linalg.solve(coeffs, self.grad_loss(zl))

            zl = zl - dz

            current_loss = self.loss(zl)
            loss_hist.append(current_loss)

            error = np.linalg.norm(dz)
            print("Epoch {}, Error {}, Loss {}".format(iter, error, current_loss))

            iter = iter + 1

        z1 = zl[:self.M_Omega]
        z2 = zl[self.M_Omega:]

        zz = jnp.append(z1, self.gbdr)
        zz = jnp.append(zz, z2)
        zz = jnp.linalg.solve(self.L, zz)

        w = jnp.linalg.solve(jnp.transpose(self.L), jnp.linalg.solve(self.L, zz))

        self.num_iter = iter
        self.loss_hist = loss_hist
        self.weights = w
        return w


    def build_theta(self, x, M, M_Omega, N, q, Q, weights):
        """
        Input:

        x: the sample points
        M: the number of points, including those on the boundary
        M_Omega: the number of points int the interior
        N: 2 * N is the number of eigenfunctions used, each eivenfunction is either sin(i\pi x) or cos(i\pi x)
        q: the Gauss-quadrature points
        Q: the number of Gauss-quadrature points
        """
        theta = jnp.zeros((M + 2 * N, M + 2 * N))

        x0 = np.reshape(x, (M, 1))
        q0 = np.reshape(q, (Q, 1))

        x0x0v = jnp.tile(x0, M).flatten()
        x0x0h = jnp.tile(np.transpose(x0), (M, 1)).flatten()

        x0q0v = jnp.tile(x0, Q).flatten()
        x0q0h = jnp.tile(jnp.transpose(q0), (M, 1)).flatten()

        q0x0v = jnp.tile(q0, M).flatten()
        q0x0h = jnp.tile(jnp.transpose(x0), (Q, 1)).flatten()

        q0q0v = jnp.tile(q0, Q).flatten()
        q0q0h = jnp.tile(jnp.transpose(q0), (Q, 1)).flatten()

        # K(x, y)
        val = vmap(lambda _x, _y: self.kernel.kappa(_x, _y))(x0x0v, x0x0h)
        theta = theta.at[:M, :M].set(jnp.reshape(val, (M, M)))

        # \Delta_y K(x, q)
        val = vmap(lambda _x, _q: self.kernel.Delta_y_kappa(_x, _q))(x0q0v, x0q0h)
        mtx = np.reshape(val, (M, Q))

        eigen_func1 = lambda _q, _wq, _i: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, None, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, 0, None))

        eigen_func2 = lambda _q, _wq, _i: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, None, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, 0, None))

        scalers = jnp.array(jnp.arange(0, N) + 1)
        eigen_func1_vals = eigen_func1(q, weights, scalers)
        eigen_func2_vals = eigen_func2(q, weights, scalers)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=1)

        theta = theta.at[:M, M:].set(mtx @ eigen_func_vals)

        # \Delta_x K(q, x)
        val = vmap(lambda _x, _q: self.kernel.Delta_x_kappa(_x, _q))(q0x0v, q0x0h)
        mtx = np.reshape(val, (Q, M))

        eigen_func1 = lambda _i, _q, _wq: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, 0, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, None, None))

        eigen_func2 = lambda _i, _q, _wq: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, 0, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, None, None))

        scalers = jnp.array(jnp.arange(0, N) + 1)
        eigen_func1_vals = eigen_func1(scalers, q, weights)
        eigen_func2_vals = eigen_func2(scalers, q, weights)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=0)

        theta = theta.at[M:, 0:M].set(eigen_func_vals @ mtx)


        # \Delta_x \Delta_y K(q, q)
        val = vmap(lambda _qx, _qy: self.kernel.Delta_y_kappa(_qx, _qy))(q0q0v, q0q0h)
        mtx = np.reshape(val, (Q, Q))

        func = (lambda ar, ac: np.dot(ar, np.dot(mtx, ac)))
        func = (vmap(func, in_axes=(None, 0)))
        func = jit(vmap(func, in_axes=(0, None)))
        tmp = func(eigen_func_vals, eigen_func_vals)

        theta = theta.at[M:, M:].set(tmp)
        return theta

    # build Nuggets
    def build_nuggets(self, theta, M, Q):
        trace11 = np.trace(theta[0:M, 0:M])
        trace22 = np.trace(theta[M:, M:])
        ratio = trace22 / trace11
        r_diag = np.concatenate(np.ones((1, M)), ratio * np.ones((1, Q)), axis=1)
        r = np.diag(r_diag[0])
        return r

    def fit(self, nx):
        x = self.samples
        xint0 = np.reshape(x[:self.M_Omega], (self.M_Omega, 1))
        x0 = np.reshape(x, (self.M, 1))

        nxl = len(nx)
        nx0 = np.reshape(nx, (nxl, 1))
        nxx0v = jnp.tile(nx0, self.M).flatten()
        nxx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()

        q0 = np.reshape(self.gauss_samples, (self.Q, 1))

        nx0q0v = jnp.tile(nx0, self.Q).flatten()
        nx0q0h = jnp.tile(np.transpose(q0), (nxl, 1)).flatten()

        mtx = np.zeros((nxl, self.M + self.N))

        val0 = vmap(lambda _x, _y: self.kernel.kappa(_x, _y))(nxx0v, nxx0h)
        mtx[0:nxl, :self.M] = np.reshape(val0, (nxl, self.M))

        val1 = vmap(lambda _x, _y: self.kernel.Delta_y_kappa(_x, _y))(nx0q0v, nx0q0h)
        tmp = np.reshape(val1, (nxl, self.Q))

        eigen_func1 = lambda _q, _wq, _i: jnp.sin(_i * jnp.pi * _q) * _wq
        eigen_func1 = vmap(eigen_func1, in_axes=(None, None, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, 0, None))

        eigen_func2 = lambda _q, _wq, _i: jnp.cos(_i * jnp.pi * _q) * _wq
        eigen_func2 = vmap(eigen_func2, in_axes=(None, None, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, 0, None))

        scalers = jnp.array(jnp.arange(0, N) + 1)
        eigen_func1_vals = eigen_func1(self.gauss_samples, self.gauss_weights, scalers)
        eigen_func2_vals = eigen_func2(self.gauss_samples, self.gauss_weights, scalers)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=1)

        mtx[:nxl, self.M:].set(tmp @ eigen_func_vals)

        u = mtx.dot(self.weights)
        return u