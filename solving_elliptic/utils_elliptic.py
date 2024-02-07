import jax.numpy as jnp
from jax import vmap, jit, hessian, grad
import math

# Defining the elliptic operators 

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

def L_operator_x(x, y, l, epsilon,):
    return epsilon*neg_laplacian_x(x, y, l) - matern_kernel(x, y, l)

vmap_L_operator_x = jit(vmap(vmap(L_operator_x, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None)))

def L_operator_y(x, y, l, epsilon):
    return epsilon*neg_laplacian_y(x, y, l) - matern_kernel(x, y, l)
vmap_L_operator_y = jit(vmap(vmap(L_operator_y, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None)))


# def LL_operator(x, y, l, epsilon):
#     return -epsilon**2*double_neg_laplacian(x, y, l) + 2*epsilon*neg_laplacian_x(x, y, l)+ matern_kernel(x, y, l)

def LL_operator(x, y, l, epsilon):
    return epsilon**2*double_neg_laplacian(x, y, l) - epsilon*neg_laplacian_x(x, y, l) - epsilon*neg_laplacian_y(x, y, l)+ matern_kernel(x, y, l)

vmap_LL_operator = vmap(vmap(LL_operator, in_axes=(0, None, None, None)), in_axes = (None, 0, None, None))


# Defining how to compute the integrals against test functions
def bilinear_form_K(x, y, points_1, points_2, length_scale, epsilon):
    # Create the kernel matrix 
    K = vmap_LL_operator(points_1, points_2, length_scale, epsilon)
    return jnp.dot(x, K @ y)

vmap_bilinear_form_K = jit(vmap(vmap(bilinear_form_K,  in_axes=(None, 0, None, 0, None, None)), in_axes=(0, None, 0, None, None, None)))

def linear_form_K(x, p, points, length_scale, epsilon):
    # Create the kernel matrix 
    K = vmap_L_operator_y(p, points, length_scale, epsilon)
    return K@x

vmap_linear_form_K = jit(vmap(vmap(linear_form_K, in_axes=(None, 0, None, None, None)), in_axes=(0, None, 0, None, None)))


# Define a function that constructs the various blocks of the matrix
@jit
def theta_blocks(boundary,psi_matrix, root_psi, length_scale, epsilon):
    theta_11 = vmap_kernel(boundary, boundary, length_scale)
    theta_21 = jnp.squeeze(vmap_linear_form_K(psi_matrix, boundary, root_psi, length_scale, epsilon), axis = -1)
    theta_22 = vmap_bilinear_form_K(psi_matrix, psi_matrix, root_psi, root_psi, length_scale, epsilon)
    return theta_11, theta_21, theta_22

@jit
def evaluate_prediction(x, c, length_scale, root_psi, psi_matrix, boundary, epsilon):
    K_boundary = vmap_kernel(x,boundary, length_scale)
    K_interior = jnp.squeeze(vmap_linear_form_K(psi_matrix, x[:, None], root_psi, length_scale, epsilon), axis = -1).T
    K_evaluate = jnp.block([[K_boundary, K_interior]])

    return K_evaluate@c