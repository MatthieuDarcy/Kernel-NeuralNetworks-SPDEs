

import jax.numpy as jnp
from jax import vmap, jit, hessian, grad

########################################################################################################################

# Utils for elliptic operators with coefficients

def matern_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)

vmap_kernel_row = vmap(matern_kernel, in_axes=(None, 0, None))
# Now we apply vmap to the result to vectorize over the rows of the first argument
vmap_kernel = jit(vmap(vmap_kernel_row, in_axes=(0, None, None)))


def neg_laplacian_x(x,y, l ):
    hess = -hessian(matern_kernel, argnums = 0)(x,y,l)
    nu = 5/2
    hess = jnp.where(jnp.allclose(x,y), nu/(l**2*(nu-1)), hess)
    return jnp.sum(hess)

def neg_laplacian_y(x,y, l ):
    hess = -hessian(matern_kernel, argnums = 1)(x,y,l)
    nu = 5/2
    hess = jnp.where(jnp.allclose(x,y), nu/(l**2*(nu-1)), hess)
    return jnp.sum(hess)

def double_neg_laplacian(x,y,l):
    hess = -hessian(neg_laplacian_x, argnums = 1)(x,y,l)

    nu = 5/2
    hess = jnp.where(jnp.allclose(x,y), nu**2/(8*(2-3*nu+nu**2))*math.factorial(4)/l**4, hess)
    return jnp.sum(hess)

def L_operator_x(x, y, l, epsilon, b_x):
    return epsilon*neg_laplacian_x(x, y, l) + b_x*matern_kernel(x, y, l)

vmap_L_operator_x = jit(vmap(vmap(L_operator_x, in_axes=(0, None, None, None, 0)), in_axes=(None, 0, None, None, None)))

def L_operator_y(x, y, l, epsilon, b_y):
    return epsilon*neg_laplacian_y(x, y, l) + b_y*matern_kernel(x, y, l)
vmap_L_operator_y = jit(vmap(vmap(L_operator_y, in_axes=(None, 0, None, None, 0)), in_axes=(0, None, None, None, None)))

def LL_operator(x, y, l, epsilon,b_x, b_y):
    return epsilon**2*double_neg_laplacian(x, y, l) + epsilon*b_y*neg_laplacian_x(x, y, l) + epsilon*b_x*neg_laplacian_y(x, y, l)+ b_x*b_y*matern_kernel(x, y, l)

vmap_LL_operator = vmap(vmap(LL_operator, in_axes=(0, None, None, None,0, None)), in_axes = (None, 0, None, None, None, 0))
