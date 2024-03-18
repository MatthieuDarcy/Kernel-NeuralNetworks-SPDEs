import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, hessian
from jax import vmap, jit
import jax.ops as jop
import jax.scipy as jsp
from SolvingRoughPDEs.utilities.domain import *
from functools import partial
import time
import numpy as np
import jax
from jax.config import config
import jax.scipy as jsp
config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

class PathSampling(object):
    def __init__(self, a, b, T, Deltau, Deltat, sigma, xa, xb):
        self.a = a
        self.b = b
        self.T = T
        self.Deltau = Deltau
        self.Deltat = Deltat
        self.nN = int((b - a) / Deltau)     # The number of intervals in space
        self.nT = int(T / Deltat)           # The number of intervals in time
        self.sigma = sigma
        self.xa = xa                    # Fixed value at the left boundary
        self.xb = xb                    # Fixed value at the right boundary
        self.key = jax.random.PRNGKey(0)
    def CoeffsMtx(self):
        Mtx = np.zeros((self.nN - 1, self.nN - 1))
        Mtx[jnp.arange(self.nN - 1), jnp.arange(self.nN - 1)] = -2
        Mtx[jnp.arange(self.nN - 2), jnp.arange(1, self.nN - 1)] = 1
        Mtx[jnp.arange(1, self.nN - 1), jnp.arange(self.nN - 2)] = 1
        #return Mtx
        return -1/(2 * self.sigma ** 2 * self.Deltau** 2) * Mtx + jnp.eye(self.nN - 1)

    def f(self, x):
        return x * (8/(1 + x**2)**2 - 2)

    def df(self, x):
        return grad(self.f)(x)

    def ddf(self, x):
        return grad(self.df)(x)

    def G(self, xs):
        self.key, subkey = jax.random.split(self.key)
        bms = jax.random.normal(subkey, (self.nN - 1,))
        aug_xs = jnp.concatenate((jnp.array([self.xa]), xs, jnp.array([self.xb])))
        xs_right = jnp.roll(aug_xs, -1)
        xs_left = jnp.roll(aug_xs, 1)
        Delta = (xs_right - 2 * aug_xs + xs_left) / (2 * self.Deltau ** 2)
        Delta = Delta[1:-1]

        fx      = vmap(self.f)(xs)
        dfx     = vmap(self.df)(xs)
        ddfx    = vmap(self.ddf)(xs)

        update = (Delta - fx * dfx - self.sigma**2/2 * ddfx) * self.Deltat / self.sigma ** 2\
                 + jnp.sqrt(2 * self.Deltat / self.Deltau) * bms

        update = update.at[0].set(update[0] + 1/(2 * (self.Deltau * self.sigma) ** 2) * self.xa)
        update = update.at[-1].set(update[-1] + 1/(2 * (self.Deltau * self.sigma) ** 2) * self.xb)

        return xs + update

    def solve(self):
        itv = 100
        sols = jnp.zeros((itv, self.nN + 1))
        current_xs = jnp.linspace(self.xa, self.xb, self.nN + 1)
        sols = sols.at[0, :].set(current_xs)
        Mtx = self.CoeffsMtx()
        cho_factor = jsp.linalg.cho_factor(Mtx)
        for i in range(self.nT):
            b = self.G(current_xs[1:-1])
            xs = jsp.linalg.cho_solve(cho_factor, b)
            current_xs = jnp.concatenate((jnp.array([self.xa]), xs, jnp.array([self.xb])))
            sols = sols.at[(i+1) % itv, :].set(current_xs)
            if i % 1000 == 0:
                plt.figure()
                xx = jnp.linspace(self.a, self.b, self.nN + 1)
                plt.plot(xx, current_xs)
                plt.savefig(f"./results/SP/{i}.png")

        return sols, itv, (self.nT + 1) % itv

























