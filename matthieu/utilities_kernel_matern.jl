


import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import hessian
import math

# Utilities for the kernel

# This is actually the squared exponential kernel
def matern_kernel(x, y, length_scale):
    r = jnp.sum((x - y) ** 2)
    #factor =r / length_scale
    return jnp.exp(-r/(2*length_scale**2))


def matern_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    #factor =r / length_scale
    return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)


vmap_kernel_row = vmap(matern_kernel, in_axes=(None, 0, None))
# Now we apply vmap to the result to vectorize over the rows of the first argument
vmap_kernel = vmap(vmap_kernel_row, in_axes=(0, None, None))

# Define a function that produces the kernel matrix evaluated at each pair of entries
# First, we vectorize the matern_kernel function over the entries of the first vector
vmapped_matern_kernel_first_vector = vmap(matern_kernel, in_axes=(0, None, None))
# Now, we vectorize the result over the entries of the second vector
vmapped_matern_kernel_matrix = vmap(vmapped_matern_kernel_first_vector, in_axes=(None, 0, None))

# Define the function that computes the kernel matrix for two vectors
def compute_K_pairwise(vector1, vector2, length_scale):
    # The resulting matrix will have shape (d, d)
    return vmapped_matern_kernel_matrix(vector1, vector2, length_scale)

# Define a function that computes negative laplacian of the kernel
@jit
def neg_laplacian_x(x,y, l ):
    hess = -hessian(matern_kernel, argnums = 0)(x,y,l)
    nu = 5/2
    hess = jnp.where(jnp.allclose(x,y), nu/(l**2*(nu-1)), hess)
    return jnp.sum(hess)

@jit
def neg_laplacian_y(x,y, l ):
    hess = -hessian(matern_kernel, argnums = 1)(x,y,l)
    nu = 5/2
    hess = jnp.where(jnp.allclose(x,y), nu/(l**2*(nu-1)), hess)
    return jnp.sum(hess)

@jit
def double_neg_laplacian(x,y,l):
    hess = -hessian(neg_laplacian_x, argnums = 1)(x,y,l)

    nu = 5/2
    hess = jnp.where(jnp.allclose(x,y), nu**2/(8*(2-3*nu+nu**2))*math.factorial(4)/l**4, hess)
    return jnp.sum(hess)

# Vectorize the gradient computation over the second argument y
vmap_hess_one_kernel_row = jit(vmap(neg_laplacian_x, in_axes=(None, 0, None)))
# Vectorize the above result over the first argument x
vmap_kernel_laplacian = jit(vmap(vmap_hess_one_kernel_row, in_axes=(0, None, None)))


# Vectorize the gradient computation over the second argument y
vmap_hess_kernel_row = jit(vmap(double_neg_laplacian, in_axes=(None, 0, None)))
# Vectorize the above result over the first argument x
vmap_kernel_double_laplacian = jit(vmap(vmap_hess_kernel_row, in_axes=(0, None, None)))


# Define a function that produces the kernel matrix evaluated at each pair of entries
vmap_matern_laplacian_first_vector = jit(vmap(double_neg_laplacian, in_axes=(0, None, None)))
# Now, we vectorize the result over the entries of the second vector
vmap_matern_laplacian_kernel_matrix = jit(vmap(vmap_matern_laplacian_first_vector, in_axes=(None, 0, None)))
#  Define the function that computes the kernel matrix for two vectors
def compute_K_double_laplacian_pairwise(vector1, vector2, length_scale):
    # The resulting matrix will have shape (d, d)
    return vmap_matern_laplacian_kernel_matrix(vector1, vector2, length_scale)