
import jax.numpy as jnp
from jax import grad, jit, vmap
from utils_DST import *


# Defining the second order elliptic operators 
def matern_kernel(x, y, length_scale):
    r = jnp.sqrt(jnp.sum((x - y) ** 2))
    return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)

vmap_kernel_row = vmap(matern_kernel, in_axes=(None, 0, None))
# Now we apply vmap to the result to vectorize over the rows of the first argument
vmap_kernel = jit(vmap(vmap_kernel_row, in_axes=(0, None, None)))

# Defining the second order elliptic operators 
def matern_kernel_2d(x_1, x_2, y_1, y_2, length_scale):
    r = jnp.sqrt((x_1 - y_1) ** 2 + (x_2 - y_2) ** 2)
    return (1 + jnp.sqrt(5)*r/length_scale + (5 / 3) * (r ** 2) / (length_scale ** 2)) * jnp.exp(-jnp.sqrt(5)*r/length_scale)

matern_kernel_tensor = jit(vmap(vmap(vmap(vmap(matern_kernel_2d, in_axes=(None, None,None, 0, None)), in_axes=(None, None, 0, None, None)), in_axes = (None, 0, None, None, None)), in_axes=(0, None, None, None, None)))

def neg_laplacian_x(x_1, x_2, y_1, y_2, length_scale):
    x = jnp.hstack([x_1, x_2])
    y = jnp.hstack([y_1, y_2])

    nu = 5/2
    l_1 = grad(grad(matern_kernel_2d, argnums=0), argnums = 0)(x_1, x_2, y_1, y_2, length_scale)
    l_1 = jnp.where(jnp.allclose(x,y), -nu/(length_scale**2*(nu-1)), l_1)
    l_2 = grad(grad(matern_kernel_2d, argnums=1), argnums = 1)(x_1, x_2, y_1, y_2, length_scale)
    l_2 = jnp.where(jnp.allclose(x,y), -nu/(length_scale**2*(nu-1)), l_2)
    return -(l_1 + l_2)

def neg_laplacian_y(x_1, x_2, y_1, y_2, length_scale):
    x = jnp.hstack([x_1, x_2])
    y = jnp.hstack([y_1, y_2])
    
    nu = 5/2
    l_1 = grad(grad(matern_kernel_2d, argnums=2), argnums = 2)(x_1, x_2, y_1, y_2, length_scale)
    l_1 = jnp.where(jnp.allclose(x,y), -nu/(length_scale**2*(nu-1)), l_1)
    l_2 = grad(grad(matern_kernel_2d, argnums=3), argnums = 3)(x_1, x_2, y_1, y_2, length_scale)
    l_2 = jnp.where(jnp.allclose(x,y), -nu/(length_scale**2*(nu-1)), l_2)
    return -(l_1 + l_2)

def double_neg_laplacian(x_1, x_2, y_1, y_2, length_scale):
    x = jnp.hstack([x_1, x_2])
    y = jnp.hstack([y_1, y_2])
    
    nu = 5/2
    l_1 = grad(grad(neg_laplacian_x, argnums=2), argnums = 2)(x_1, x_2, y_1, y_2, length_scale)
    l_1 =  jnp.where(jnp.allclose(x,y), -4*nu**2/((2-3*nu+nu**2))/length_scale**4, l_1)
    
    l_2 = grad(grad(neg_laplacian_x, argnums=3), argnums = 3)(x_1, x_2, y_1, y_2, length_scale)
    l_2 =  jnp.where(jnp.allclose(x,y), -4*nu**2/((2-3*nu+nu**2))/length_scale**4, l_2)

    return -(l_1 + l_2)

double_neg_laplacian_tensor = jit(vmap(vmap(vmap(vmap(double_neg_laplacian, in_axes=(None, None,None, 0, None)), in_axes=(None, None, 0, None, None)), in_axes = (None, 0, None, None, None)), in_axes = (0, None, None, None, None)))
neg_laplacian_x_tensor = jit(vmap(vmap(vmap(vmap(neg_laplacian_x, in_axes=(None, None,None, 0, None)), in_axes=(None, None, 0, None, None)), in_axes = (None, 0, None, None, None)), in_axes = (0, None, None, None, None)))
neg_laplacian_y_tensor = jit(vmap(vmap(vmap(vmap(neg_laplacian_y, in_axes=(None, None,None, 0, None)), in_axes=(None, None, 0, None, None)), in_axes = (None, 0, None, None, None)), in_axes = (0, None, None, None, None)))
neg_laplacian_y_partial = vmap(vmap(vmap(neg_laplacian_y, in_axes=(None, None,None, 0, None)), in_axes=(None, None, 0, None, None)), in_axes = (0, 0, None, None, None))
matern_kernel_partial = vmap(vmap(vmap(matern_kernel_2d, in_axes=(None, None,None, 0, None)), in_axes=(None, None, 0, None, None)), in_axes = (0, 0, None, None,None))


@jit
def L_b_xy_tensor(x_1,x_2, y_1, y_2,length_scale, epsilon,b_1, b_2):
    return epsilon**2*double_neg_laplacian_tensor(x_1,x_2, y_1, y_2,length_scale) + epsilon*b_2[None, None]*neg_laplacian_x_tensor(x_1,x_2, y_1, y_2, length_scale) + epsilon*b_1[:, :, None, None]*neg_laplacian_y_tensor(x_1,x_2, y_1, y_2, length_scale) + b_1[:, :, None, None]*b_2[None, None]*matern_kernel_tensor(x_1,x_2, y_1, y_2, length_scale)
@jit
def L_b_y_partial(x_1, x_2, y_1, y_2, length_scale, epsilon, b_1):
    return  epsilon*neg_laplacian_y_partial(x_1, x_2, y_1, y_2, length_scale) + jnp.multiply(b_1[None], matern_kernel_partial(x_1, x_2, y_1, y_2, length_scale))

@jit 
def L_b_y_tensor(x_1, x_2, y_1, y_2, length_scale, epsilon, b_1):
    return  epsilon*neg_laplacian_y_tensor(x_1, x_2, y_1, y_2, length_scale) + jnp.multiply(b_1[None], matern_kernel_tensor(x_1, x_2, y_1, y_2, length_scale))

@jit
def build_matrices(boundary_points,x,y,length_scale, b_matrix, nu):
    n_boundary = boundary_points.shape[0]
    n_points = x.shape[0]
    theta_11 = vmap_kernel(boundary_points, boundary_points, length_scale)
    theta_22_tensor = L_b_xy_tensor(x, y,x,y, length_scale, nu, b_matrix, b_matrix)
    theta_12_tensor = L_b_y_partial(boundary_points[:, 0],boundary_points[:, 1],  x, y, length_scale, nu, b_matrix)

    theta_22_dst_tensor = double_compute_sine_coef_2d(theta_22_tensor)
    theta_12_dst_tensor  = vmap_compute_sine_coef_2d(theta_12_tensor)

    theta_12_dst = jnp.reshape(theta_12_dst_tensor, (n_boundary,n_points*n_points))
    theta_22_dst = jnp.reshape(theta_22_dst_tensor, shape= ( n_points*n_points, n_points*n_points))
    theta = jnp.block([[theta_11, theta_12_dst], [theta_12_dst.T, theta_22_dst]])

    return theta, theta_11, theta_12_dst, theta_22_dst

def solve(theta, boundary_condition, coef_forcing, s_decay, reg_bc = 1e-10, reg = 1e-7):
    n_points = coef_forcing.shape[0]
    n_boundary = boundary_condition.shape[0]
    rhs = jnp.hstack([boundary_condition, jnp.reshape(coef_forcing, (n_points)*(n_points))])

    # Adaptive nugget
    decay_nugget =  (jnp.arange(1,n_points+1)[None]**2  + jnp.arange(1,n_points+1)[:,None]**2) 
    decay_nugget = decay_nugget.reshape((n_points)*(n_points))**(s_decay)
    nugget = jnp.hstack([jnp.ones(n_boundary)*reg_bc, decay_nugget*reg])

    alpha = scipy.linalg.solve(theta + jnp.diag(nugget), rhs, assume_a='pos')
    return alpha

def create_interior(x, y):
    # Use meshgrid to create the grid of pairs
    A, B = jnp.meshgrid(x, y, indexing='ij')

    # Combine A and B into pairs
    interior = jnp.stack((A, B), axis=-1).reshape(-1, 2)
    return interior


def predict(alpha, x_eval, y_eval, x,y, boundary_points, length_scale, nu, b_matrix):
    n_eval = x_eval.shape[0]
    n_points =x.shape[0]
    interior_eval = create_interior(x_eval, y_eval)
    K_interior = L_b_y_tensor(x_eval, y_eval, x, y, length_scale, nu, b_matrix).reshape(n_eval*n_eval, n_points, n_points)
    K_interior = vmap_compute_sine_coef_2d(K_interior)
    K_interior = K_interior.reshape(-1, (n_points)*(n_points))

    K_eval_bc = vmap_kernel(interior_eval, boundary_points, length_scale)
    K_eval = jnp.hstack([K_eval_bc, K_interior])


    pred = jnp.dot(K_eval, alpha)
    pred_grid = jnp.reshape(pred, (n_eval, n_eval))

    return pred_grid, pred



#### Pointwise version

@jit
def build_matrices_pointwise(boundary_points,x,y,length_scale, b_matrix, nu):
    n_boundary = boundary_points.shape[0]
    n_points = x.shape[0]
    theta_11 = vmap_kernel(boundary_points, boundary_points, length_scale)
    theta_22_tensor = L_b_xy_tensor(x, y,x,y, length_scale, nu, b_matrix, b_matrix)
    theta_12_tensor = L_b_y_partial(boundary_points[:, 0],boundary_points[:, 1],  x, y, length_scale, nu, b_matrix)

    theta_22_dst_tensor = theta_22_tensor
    theta_12_dst_tensor  = theta_12_tensor

    theta_12_dst = jnp.reshape(theta_12_dst_tensor, (n_boundary,n_points*n_points))
    theta_22_dst = jnp.reshape(theta_22_dst_tensor, shape= ( n_points*n_points, n_points*n_points))
    theta = jnp.block([[theta_11, theta_12_dst], [theta_12_dst.T, theta_22_dst]])

    return theta, theta_11, theta_12_dst, theta_22_dst

def solve_pointwise(theta, boundary_condition, f_observed, s_decay, reg_bc = 1e-10, reg = 1e-7):
    n_points = f_observed.shape[0]
    n_boundary = boundary_condition.shape[0]
    rhs = jnp.hstack([boundary_condition, jnp.reshape(f_observed, (n_points)*(n_points))])

    # Adaptive nugget
    decay_nugget =jnp.ones(n_points*n_points)
    nugget = jnp.hstack([jnp.ones(n_boundary)*reg_bc, decay_nugget*reg])

    alpha = scipy.linalg.solve(theta + jnp.diag(nugget), rhs, assume_a='pos')
    return alpha

def predict_pointwise(alpha, x_eval, y_eval, x,y, boundary_points, length_scale, nu, b_matrix):
    n_eval = x_eval.shape[0]
    n_points =x.shape[0]
    interior_eval = create_interior(x_eval, y_eval)
    K_interior = L_b_y_tensor(x_eval, y_eval, x, y, length_scale, nu, b_matrix).reshape(n_eval*n_eval, n_points, n_points)
    K_interior = K_interior.reshape(-1, (n_points)*(n_points))

    K_eval_bc = vmap_kernel(interior_eval, boundary_points, length_scale)
    K_eval = jnp.hstack([K_eval_bc, K_interior])


    pred = jnp.dot(K_eval, alpha)
    pred_grid = jnp.reshape(pred, (n_eval, n_eval))

    return pred_grid, pred