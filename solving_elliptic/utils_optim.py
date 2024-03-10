
import jax.numpy as jnp
from jax import vmap
from jax import jit

from utils_rough_pde import vmap_integrate_f_test_functions

from jax.config import config
config.update("jax_enable_x64", True)

from jax import scipy


def solve_qp(K, K_interior, K_bc, L_stiff, f_m, bc, reg, reg_stab = 1e-5):
    #print(L_stiff)
    rhs= jnp.hstack([K_interior.T@scipy.linalg.cho_solve(L_stiff, f_m), bc])
    Q = jnp.block([ [K_interior.T@scipy.linalg.cho_solve(L_stiff, K_interior) + reg*K, K_bc.T], [K_bc, jnp.zeros((2, 2))]])

    sol = scipy.linalg.solve(Q + reg_stab*jnp.eye(Q.shape[0]), rhs, assume_a='sym')
    c = sol[:-2]

    return c

def compute_frechet_proj(u_root, psi_matrix, tau_prime):
    root_values =  tau_prime(u_root)*u_root
    return vmap_integrate_f_test_functions(root_values, psi_matrix)

def Gauss_Newton_solver(lr, L_stiff, f_meas, b, tau, tau_prime, psi_matrix, root_psi,linear_solver, reg, n_iter = 10):

    rhs_meas = jnp.copy(f_meas)
    root_b = b(root_psi)

    prev_error = jnp.inf

    for i in range(n_iter):
        # Do a linear solve
        linear_solver.solve_linear_prob(rhs_meas, reg, root_b, L_stiff)
        # Compute the residuals
        l_2_error, l_2_rel, h_s_error, h_s_rel = linear_solver.compute_residuals_nl(f_meas, tau, root_b, L_stiff)
        if h_s_rel > prev_error:
            lr = lr/2
        else:
            prev_error = h_s_rel
            print("Iteration {}, current error: {} H^-s, {} l_2".format(i+1, h_s_rel, l_2_rel)) 

            # Update the rhs 
            u_pred_root = linear_solver.evaluate_solution_parallel(root_psi)
            linear_proj = compute_frechet_proj(u_pred_root, psi_matrix, tau_prime)
            residual = f_meas - linear_solver.non_linear_mes
            rhs_meas = lr*residual + linear_proj

            # Update the operator by updating the roots of b
            root_b = b(root_psi) + tau_prime(u_pred_root)
    
    return linear_solver