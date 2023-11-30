
import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import hessian

from jax.config import config
config.update("jax_enable_x64", True)

# Utilities for bump functions
def bump(x, epsilon, loc):
    condition = (loc - epsilon < x) & (x < loc + epsilon)
    return jnp.where(
        condition,
        jnp.exp(-1 / (1 - (x - loc) ** 2 / epsilon ** 2))/epsilon/0.4439938,
        0.0
    )

# First vectorization over 'loc'
vectorized_over_loc = vmap(bump, in_axes=(None, None, 0))
# Second vectorization over 'epsilon'
vectorized_over_epsilon = vmap(vectorized_over_loc, in_axes=(None, 0, None))
# Final vectorization over 'x_values'
vmap_bump = vmap(vectorized_over_epsilon, in_axes=(0, None, None))

bump_vector = vmap(bump, in_axes=(0, None, 0))



########################################################################################################################

# Utilities for the kernel

# This is actually the squared exponential kernel
def matern_kernel(x, y, length_scale):
    r = jnp.sum((x - y) ** 2)
    #factor =r / length_scale
    return jnp.exp(-r/(2*length_scale**2))

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

    #hess = jnp.where(jnp.isnan(hess), 0.0, hess)
    return jnp.sum(hess)


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



########################################################################################################################



if __name__=="__main__":
    # Define the parameters of the bump functions
    epsilon_values =  jnp.array([0.05])

    # Boundary of the domain
    lower, upper = -1.0, 1.0
    L = 2.0 # Lenght of the domain
    loc_values = jnp.linspace(lower + epsilon_values[0], upper - epsilon_values[0],  int(L/(2*epsilon_values[0])))
    support = jnp.array([loc_values - epsilon_values[0], loc_values + epsilon_values[0]]).T
    vol = support[:,1] - support[:,0]
    N_test_functions = loc_values.shape[0]

    # Define the parameters of the kernel
    length_scale = 0.5
    # Import the quadrature rules
    from scipy.special import roots_legendre
    n_order = 1000
    x_q, w_q = roots_legendre(n_order)

    root_psi, w_psi = vmap_root_interval(x_q, w_q, support)



    # Construct the matrix of psi functions
    psi_matrix = bump_vector(root_psi, epsilon_values, loc_values)
    psi_matrix = psi_matrix * w_psi

    theta = construct_theta_integral(psi_matrix, root_psi, length_scale)

    print(theta.shape)

    eigenvalues, eigenvectors = jnp.linalg.eigh(theta)
    print("The smallest eigenvalue is ", jnp.min(eigenvalues))

    # These values should be roughly equal
    print("These values (along the diagonal) should be roughly equal")
    # Pause here until user presses enter
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[i,i], double_neg_laplacian(loc_values[i], loc_values[i], length_scale))




    print("These values should also be roughly equal (kernel at different points)")
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[0,i], double_neg_laplacian(loc_values[0], loc_values[i], length_scale))

    print("Now we try with a Monte Carlo approximation")
    input("Press enter to view...")

    # Instead of using Gauss quadrature points, we will use random samples 
    from jax import random

    N_samples = 1000
    key = random.PRNGKey(0)
    x_q = random.uniform(key, (N_test_functions, N_samples))*2*epsilon_values[0] +  loc_values[:, None] - epsilon_values[0]
    root_psi = x_q

    # The weights are not defined by the quadrature rule anymore but instead by the volume of integration and the number of samples
    w_psi = jnp.ones((N_test_functions, N_samples))/N_samples*vol[:, None]


    psi_matrix = bump_vector(root_psi, epsilon_values, loc_values)
    psi_matrix = psi_matrix * w_psi
    theta = construct_theta_integral(psi_matrix, root_psi,  length_scale)


    eigenvalues, eigenvectors = jnp.linalg.eigh(theta)
    print("The smallest eigenvalue is ", jnp.min(eigenvalues))

    # These values should be roughly equal
    print("These values (along the diagonal) should be roughly equal")
    # Pause here until user presses enter
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[i,i], double_neg_laplacian(loc_values[i], loc_values[i], length_scale))

    print("These values should also be roughly equal (kernel at different points)")
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[0,i], double_neg_laplacian(loc_values[0], loc_values[i], length_scale))




    
    


    


    
    