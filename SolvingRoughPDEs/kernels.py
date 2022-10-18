import jax.numpy as jnp
from functools import partial
from jax import jit

@jit
def pairwise_norm_squared(x,y):
    # Given dim(X) = (N x d) and dim(Y) = (M x d), compute matrix
    # of size (N x M) containining pairwise squared distances
    if len(x.shape) <= 1:
        x = x.reshape(x.size,1)
    Xsq = (x ** 2).sum(1)
    Ysq = (y ** 2).sum(1)
    XY  = x @ y.T
    return Xsq[:, None] + Ysq[None,:] - 2 * XY

@partial(jit, static_argnums=(2,))
def gaussian_kernel(X1, X2, l=0.05):
    dsq = pairwise_norm_squared(X1, X2)    
    return jnp.exp(-dsq/(2*l**2))

@partial(jit, static_argnums=(2,))
def matern_kernel_5(matrix_1, matrix_2, l=1.0):
    d = _sqrt(pairwise_norm_squared(matrix_1,matrix_2))
    return (1 + jnp.sqrt(5)*d/l + 5/3*d**2/l**2)*jnp.exp(-jnp.sqrt(5)*d/l)

@partial(jit, static_argnums=(2,))
def matern_kernel_3(matrix_1, matrix_2, l=1.0):
    d = _sqrt(pairwise_norm_squared(matrix_1,matrix_2))
    z = jnp.sqrt(3.)*d/l
    return (1. + z)*jnp.exp(-1.*z)

def _sqrt(x, eps=1e-10):
    return jnp.sqrt(x + eps)
