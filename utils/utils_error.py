from scipy import integrate
import jax.numpy as jnp

def compute_1d_error(pred, u, x, _):
    norm_u = jnp.sqrt(integrate.trapezoid(u**2, x))
    norm_diff = jnp.sqrt(integrate.trapezoid((pred - u)**2, x))
    return norm_diff, norm_diff/norm_u