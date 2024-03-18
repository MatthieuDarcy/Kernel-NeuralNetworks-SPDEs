import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import hessian


########################################################################################################################

# Utilities for the kernel

# This is actually the squared exponential kernel
def matern_kernel(x, y, length_scale):
    r = jnp.sum((x - y) ** 2)
    #factor =r / length_scale
    return jnp.exp(-r/(2*length_scale**2))


# def matern_kernel(x, y, length_scale):
#     r = jnp.sqrt(jnp.sum((x - y) ** 2))
#     #factor =r / length_scale
#     return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)


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
    return jnp.sum(hess)

@jit
def neg_laplacian_y(x,y, l ):
    hess = -hessian(matern_kernel, argnums = 1)(x,y,l)
    return jnp.sum(hess)

@jit
def double_neg_laplacian(x,y,l):
    hess = -hessian(neg_laplacian_x, argnums = 1)(x,y,l)
    return jnp.sum(hess)

# Vectorize the gradient computation over the second argument y
vmap_hess_one_kernel_row = jit(vmap(neg_laplacian_x, in_axes=(None, 0, None)))
# Vectorize the above result over the first argument x
vmap_kernel_laplacian = jit(vmap(vmap_hess_one_kernel_row, in_axes=(0, None, None)))


# Vectorize the gradient computation over the second argument y
vmap_hess_one_kernel_row_y = jit(vmap(neg_laplacian_y, in_axes=(None, 0, None)))
# Vectorize the above result over the first argument x
vmap_kernel_laplacian_y = jit(vmap(vmap_hess_one_kernel_row_y, in_axes=(0, None, None)))



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


########################################################################################################################

# Utilities for the quadrature

def root_interval(x_q, w_q, interval):
    # Defines the roots of the interval [a,b]
    a= interval[0]
    b= interval[1]
    return (b-a)/2*x_q + (b+a)/2, (b - a) / 2 * w_q

vmap_root_interval = vmap(root_interval, in_axes=(None,None,  0))

@jit
def bilinear_form_K(x, y, points_1, points_2, length_scale):
    # Create the kernel matrix 
    K = compute_K_double_laplacian_pairwise(points_1, points_2, length_scale)
    return jnp.dot(x, K @ y)
# Vectorize bilinear_form_K over the rows of B for fixed rows of A
vmapped_bilinear_form_K_over_B = vmap(bilinear_form_K, in_axes=(None, 0, None, 0, None))
# Now, vectorize the result over the rows of A
vmapped_bilinear_form_K_over_A_and_B = vmap(vmapped_bilinear_form_K_over_B, in_axes=(0, None, 0, None, None))


# Define the function that applies vmapped_bilinear_form_K_over_A_and_B to compute the NxN result
@jit
def construct_theta_integral(A, B, length_scale):
    # A and B have shape (N, d)
    # The resulting matrix will have shape (N, N), where each (i, j) element is the result of
    # bilinear_form_K(A[i], A[j], B[i], B[j])
    return vmapped_bilinear_form_K_over_A_and_B(A, A, B, B, length_scale)

# Now we compute the kernel matrix between the measurements and the boundary
kernel_laplacian_vmap1 = jit(vmap(neg_laplacian_y, in_axes=(None, 0, None)))
vmap_laplacian_kernel_quad = jit(vmap(vmap(kernel_laplacian_vmap1, in_axes=(None, 0, None)), in_axes=(0,None, None)))

def construct_theta(boundary,psi_matrix, root_psi, length_scale):
    theta_11 = vmap_kernel(boundary, boundary, length_scale)
    theta_22 = construct_theta_integral(psi_matrix, root_psi, length_scale)

    K_quad = vmap_laplacian_kernel_quad(boundary, root_psi[:, :, None], length_scale)
    theta_12 = jnp.einsum('nmk,mk->nm', K_quad, psi_matrix)
    

    theta = jnp.block([[theta_11, theta_12], [theta_12.T, theta_22]])

    return theta

def evaluate_prediction(x, c, length_scale, root_psi, psi_matrix, boundary):
    K_boundary = vmap_kernel(x,boundary, length_scale)
    K_interior = jnp.einsum('nmk,mk->nm',  vmap_laplacian_kernel_quad(x, root_psi[:, :, None], length_scale), psi_matrix)
    K_evaluate = jnp.block([[K_boundary, K_interior]])

    return K_evaluate@c
