import jax.numpy as jnp
import numpy as np
from jax.config import config
from jax import vmap, hessian, grad
import munch
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

gauss_samples, gauss_weights = np.polynomial.legendre.leggauss(2000)
gauss_samples = (gauss_samples + 1) / 2
gauss_weights = gauss_weights / 2
Q = len(gauss_samples) # the number of gauss quadrature points

def kernel(x):
    return jnp.exp(-(x-0.1)**2/0.04)

def func(x):
    return hessian(kernel)(x) * jnp.sin(1000 * jnp.pi * x) * jnp.cos(1000 * jnp.pi * x)
    #return kernel(x) * jnp.sin(200 * jnp.pi * x)

integral = jnp.sum(vmap(func)(gauss_samples) * gauss_weights)
print(integral)