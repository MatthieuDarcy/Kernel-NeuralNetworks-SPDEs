

import jax.numpy as jnp
from jax import vmap, jit, hessian, grad
import math

# Defining the second order elliptic operators 
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


########################################################################################################################
# L = -Delta u + b(x)u
def L_b_x(x, y, l, epsilon, b_x):
    return epsilon*neg_laplacian_x(x, y, l) + b_x*matern_kernel(x, y, l)

vmap_L_b_x = jit(vmap(vmap(L_b_x, in_axes=(0, None, None, None, 0)), in_axes=(None, 0, None, None, None)))

def L_b_y(x, y, l, epsilon, b_y):
    return epsilon*neg_laplacian_y(x, y, l) + b_y*matern_kernel(x, y, l)
vmap_L_b_y = jit(vmap(vmap(L_b_y, in_axes=(None, 0, None, None, 0)), in_axes=(0, None, None, None, None)))

def L_b_xy(x, y, l, epsilon,b_x, b_y):
    return epsilon**2*double_neg_laplacian(x, y, l) + epsilon*b_y*neg_laplacian_x(x, y, l) + epsilon*b_x*neg_laplacian_y(x, y, l)+ b_x*b_y*matern_kernel(x, y, l)

vmap_L_b_xy = vmap(vmap(L_b_xy, in_axes=(0, None, None, None,0, None)), in_axes = (None, 0, None, None, None, 0))


########################################################################################################################
# This is for the boundary conditions
def grad_matern_kernel_x(x, y, length_scale):
    nabla = grad(matern_kernel, argnums = 0)(x, y, length_scale)
    nabla = jnp.where(jnp.allclose(x,y), 0, nabla)

    return nabla

def double_grad_matern_kernel(x, y, length_scale):
    nabla = grad(grad_matern_kernel_x, argnums = 1)(x, y, length_scale)
    nu = 5/2
    nabla = jnp.where(jnp.allclose(x,y), -2*nu/(2*(1-nu)*length_scale), nabla)

    return nabla

vmap_double_grad_kernel = vmap(vmap(double_grad_matern_kernel, in_axes = (0, None, None)), in_axes = (None, 0, None))


########################################################################################################################

# Defining how to compute the integrals against test functions
def bilinear_form_K(x, y, points_1, points_2, length_scale, epsilon, b_1, b_2):
    # Create the kernel matrix 
    K = vmap_L_b_xy(points_1, points_2, length_scale, epsilon, b_1, b_2)
    return jnp.dot(x, K @ y)

vmap_bilinear_form_K = jit(vmap(vmap(bilinear_form_K,  in_axes=(None, 0, None, 0, None, None, 0, None)), in_axes=(0, None, 0, None, None, None, None, 0)))

def linear_form_K(x, p, points, length_scale, epsilon, b_y):
    # Create the kernel matrix 
    K = vmap_L_b_y(p, points, length_scale, epsilon, b_y)
    return K@x

vmap_linear_form_K = jit(vmap(vmap(linear_form_K, in_axes=(None, 0, None, None, None, None)), in_axes=(0, None, 0, None, None, 0)))

# Define a function that constructs the various blocks of the matrix
@jit
def theta_blocks(boundary,psi_matrix, root_psi, length_scale, epsilon, b_root):
    theta_11 = vmap_kernel(boundary, boundary, length_scale)
    theta_21 = jnp.squeeze(vmap_linear_form_K(psi_matrix, boundary, root_psi, length_scale, epsilon, b_root), axis = -1)
    theta_22 = vmap_bilinear_form_K(psi_matrix, psi_matrix, root_psi, root_psi, length_scale, epsilon, b_root, b_root)
    return theta_11, theta_21, theta_22

@jit
def evaluate_prediction(x, c, length_scale, root_psi, psi_matrix, boundary, epsilon, b_root):
    K_boundary = vmap_kernel(x,boundary, length_scale)
    K_interior = jnp.squeeze(vmap_linear_form_K(psi_matrix, x[:, None], root_psi, length_scale, epsilon, b_root), axis = -1).T
    K_evaluate = jnp.block([[K_boundary, K_interior]])

    return K_evaluate@c

vmap_evaluate_prediction = jit(vmap(evaluate_prediction, in_axes=(0, None, None, None, None, None, None, None)))

def build_K_psi(x, length_scale, root_psi, psi_matrix, boundary, epsilon, b_root):
    K_boundary = vmap_kernel(x,boundary, length_scale)
    K_interior = jnp.squeeze(vmap_linear_form_K(psi_matrix, x[:, None], root_psi, length_scale, epsilon, b_root), axis = -1).T
    K_evaluate = jnp.block([[K_boundary, K_interior]])

    return K_evaluate
vmap_K_psi = jit(vmap(build_K_psi, in_axes=(0, None, None, None, None, None, None)))

def build_K_eval(x, length_scale, root_psi, psi_matrix, boundary, epsilon, b_root):
    K_boundary = vmap_kernel(x,boundary, length_scale)
    K_interior = jnp.squeeze(vmap_linear_form_K(psi_matrix, x[:, None], root_psi, length_scale, epsilon, b_root), axis = -1).T
    K_evaluate = jnp.block([[K_boundary, K_interior]])

    return K_evaluate

vmap_K_eval = jit(vmap(build_K_eval, in_axes=(0, None, None, None, None, None, None)))




