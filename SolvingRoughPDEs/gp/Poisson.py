import jax.numpy as jnp
from jax import grad, hessian
from jax import vmap, jit
import jax.ops as jop
from utilities.domains import *
from functools import partial
import time
import numpy as np

from jax.config import config
import jax.scipy as jsp
config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

class NonlinearElliptic(object):
    # the equation os -Delta u  = f,
    # where f = \sum_{j=1}^m j^\alpha, \xi_j \phi_j(x), where \sqrt{2}\sin(j\pi x)
    def __init__(self, kernel, domain, alpha, m, N):
        self.kernel = kernel
        self.domain = domain
        self.alpha = alpha
        self.m = m
        self.sample_f(m)
        self.N = N
        self.gauss_samples, self.gauss_weights = np.polynomial.legendre.leggauss(80)
        self.gp_size = len(self.gauss_samples) # the number of gauss quadrature points

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.prepare_data()

    def sample_f(self, m):
        self.fxi = jnp.array(np.random.normal(0, 1, m))

    def prepare_data(self):
        # compute the values of f in the interior
        self.fxomega = vmap(self.f)(self.samples[:self.M_Omega])
        self.gbdr = np.zeros(self.M - self.M_Omega)

    def u(self, x):
        pass

    def f(self, x):
        scalers = jnp.array(jnp.arange(0, self.m) + 1)
        components = vmap(lambda j, xi_j: j ** self.alpha * xi_j * jnp.sqrt(2) * jnp.sin(j * jnp.pi * x))(scalers, self.fxi)
        return jnp.sum(components)

    def loss(self, z):
        zz = jnp.append(self.alpha * (z ** self.m) - self.fxomega, z)
        zz = jnp.append(zz, self.gbdr)
        zz = jnp.linalg.solve(self.L, zz)
        return jnp.dot(zz, zz)

    def grad_loss(self, z):
        return grad(self.loss)(z)

    def GN_loss(self, z, zold):
        zz = jnp.append(self.alpha * self.m * (zold**(self.m - 1)) * z - self.fxomega, z)
        zz = jnp.append(zz, self.gbdr)
        zz = jnp.linalg.solve(self.L, zz)
        return jnp.dot(zz, zz)

    def Hessian_GN(self, z, zold):
        return hessian(self.GN_loss)(z, zold)

    def build_gram(self, nugget = 1e-8):
        theta = self.build_theta(self.samples, self.M, self.M_Omega)
        nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
        self.gram = theta + nugget * nuggets

    def train(self, cfg):
        error = 1
        iter = 0
        # set the initial value of z
        #zl = np.zeros(self.M_Omega)
        zl = np.random.rand(self.M_Omega)

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

        z2 = self.alpha * (zl ** self.m) - self.fxomega
        z = np.concatenate((z2, zl, self.gbdr))

        w = jnp.linalg.solve(jnp.transpose(self.L), jnp.linalg.solve(self.L, z))

        self.num_iter = iter
        self.loss_hist = loss_hist
        self.weights = w
        return w


    # build theta
    # x: the sample points
    # M: the number of points, including those on the boundary
    # M_Omega: the number of points int the interior
    # N: 2 * N is the number of eigenfunctions used, each eivenfunction is either sin(i\pi x) or cos(i\pi x)
    # q: the Gauss-quadrature points
    # Q: the number of Gauss-quadrature points
    def build_theta(self, x, M, M_Omega, N, q, Q):
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

        eigen_func1 = lambda _q, _i: jnp.sin(_i * jnp.pi * _q)
        eigen_func1 = vmap(eigen_func1, in_axes=(None, 0))
        eigen_func1 = vmap(eigen_func1, in_axes=(0, None))

        eigen_func2 = lambda _q, _i: jnp.sin(_i * jnp.pi * _q)
        eigen_func2 = vmap(eigen_func2, in_axes=(None, 0))
        eigen_func2 = vmap(eigen_func2, in_axes=(0, None))

        scalers = jnp.array(jnp.arange(0, N) + 1)
        eigen_func1_vals = eigen_func1(q, scalers)
        eigen_func2_vals = eigen_func2(q, scalers)

        eigen_func_vals = jnp.concatenate((eigen_func1_vals, eigen_func2_vals), axis=1)

        # \Delta_x K(q, x)
        val = vmap(lambda _x, _q: self.kernel.Delta_x_kappa(_x, _q))(q0x0v, q0x0h)
        mtx = np.reshape(val, (Q, M))

        # \Delta_x \Delta_y K(q, q)
        val = vmap(lambda _qx, _qy: self.kernel.Delta_y_kappa(_qx, _qy))(q0q0v, q0q0h)
        mtx = np.reshape(val, (Q, Q))

        return theta

    # build Nuggets
    def build_nuggets(self, theta, M, Q):
        trace11 = np.trace(theta[0:M, 0:M])
        trace22 = np.trace(theta[M:, M:])
        ratio = trace22 / trace11
        r_diag = np.concatenate(np.ones((1, M)), ratio * np.ones((1, Q)), axis=1)
        r = np.diag(r_diag[0])
        return r

    def resampling(self, nx):
        x = self.samples
        xint0 = np.reshape(x[:self.M_Omega, 0], (self.M_Omega, 1))
        xint1 = np.reshape(x[:self.M_Omega, 1], (self.M_Omega, 1))
        x0 = np.reshape(x[:, 0], (self.M, 1))
        x1 = np.reshape(x[:, 1], (self.M, 1))

        nxl = len(nx)
        nx0 = np.reshape(nx[:, 0], (nxl, 1))
        nx1 = np.reshape(nx[:, 1], (nxl, 1))
        # for K(x,x)
        xx0v = jnp.tile(nx0, self.M).flatten()
        xx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()
        xx1v = jnp.tile(nx1, self.M).flatten()
        xx1h = jnp.tile(np.transpose(x1), (nxl, 1)).flatten()

        # for Delta K(x, x)
        xxint0v = jnp.tile(nx0, self.M_Omega).flatten()
        xxint0h = jnp.tile(jnp.transpose(xint0), (nxl, 1)).flatten()
        xxint1v = jnp.tile(nx1, self.M_Omega).flatten()
        xxint1h = jnp.tile(jnp.transpose(xint1), (nxl, 1)).flatten()

        mtx = np.zeros((nxl, self.M + self.M_Omega))
        # K(x,x)
        val0 = vmap(lambda x1, x2, y1, y2: self.kernel.kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        mtx[0:nxl, self.M_Omega:self.M + self.M_Omega] = np.reshape(val0, (nxl, self.M))

        # Delta K(x,x)
        val1 = vmap(lambda x1, x2, y1, y2: self.kernel.Delta_x_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        mtx[0:nxl, 0:self.M_Omega] = np.reshape(val1, (nxl, self.M_Omega))

        u = mtx.dot(self.weights)
        return u