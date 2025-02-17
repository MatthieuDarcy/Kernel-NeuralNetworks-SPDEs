
import jax.numpy as jnp
from jax import vmap
from jax import jit

from utils_rough_pde import vmap_integrate_f_test_functions
from utils_elliptic_coef import vmap_bilinear_form_K, vmap_linear_form_K, theta_blocks, evaluate_prediction, vmap_evaluate_prediction, theta_blocks_chunked
from utils_rough_pde import compute_error
from utils_elliptic_coef import vmap_K_psi, build_K_eval, vmap_K_eval


from pdb import set_trace as bp


from jax import scipy


def solve_qp(K, K_interior, K_bc, L_stiff, f_m, bc, reg, reg_stab = 1e-5):
    """
    This solves the QP with a regularization term
    """
    #print(L_stiff)
    rhs= jnp.hstack([K_interior.T@scipy.linalg.cho_solve(L_stiff, f_m), bc])
    Q = jnp.block([ [K_interior.T@scipy.linalg.cho_solve(L_stiff, K_interior) + reg*K, K_bc.T], [K_bc, jnp.zeros((2, 2))]])

    sol = scipy.linalg.solve(Q + reg_stab*jnp.eye(Q.shape[0]), rhs, assume_a='sym')
    c = sol[:-2]

    return c

def solve_linear_system(K, f_m, bc, reg_stab = 1e-8):
    """
    This solves the linear system without regularization
    """
    rhs = jnp.hstack([bc, f_m])
    n_bc = bc.shape[0]
    #n_interior = f_m.shape[0]
    nugget_bc = jnp.ones(n_bc)*1e-10
    nugget_interior = jnp.ones(f_m.shape[0])*reg_stab
    nugget = jnp.hstack([nugget_bc, nugget_interior])
    #print(nugget.shape)
    return scipy.linalg.solve(K + jnp.diag(nugget), rhs, assume_a='pos')

def cho_solve_system(L_K, f_m, bc, reg_stab = 1e-8):
    """
    This solves the linear system without regularization
    """
    rhs = jnp.hstack([bc, f_m])
    
    #print(nugget.shape)
    return scipy.linalg.cho_solve(L_K, rhs)

def compute_frechet_proj(u_root, psi_matrix, tau_prime):
    root_values =  tau_prime(u_root)*u_root
    return vmap_integrate_f_test_functions(root_values, psi_matrix)

def Gauss_Newton_solver_variable(lr, L_stiff, f_meas, b, tau, tau_prime, psi_matrix, root_psi,linear_solver, reg, n_iter = 10):
    """
    This Gauss Newton solver accepts lr rates other than 1.0 and automatically decreases the learning rate if the error increases
    """

    rhs_meas = jnp.copy(f_meas)
    root_b = b(root_psi)
    root_linearization = jnp.copy(root_b)

    prev_error = jnp.inf

    for i in range(n_iter):
        # Do a linear solve
        linear_solver.solve_linear_prob(rhs_meas, reg, root_linearization, L_stiff)
        # Compute the residuals
        l_2_error, l_2_rel, h_s_error, h_s_rel = linear_solver.compute_residuals_nl(f_meas, tau, root_b, L_stiff)
        if h_s_error > prev_error:
            lr = lr/10
            print("Decreasing the learning rate to {}".format(lr))
            rhs_meas = lr*residual +linear_part

        else:
            improvement = jnp.abs(prev_error - h_s_error)/jnp.abs(prev_error)
            prev_error = h_s_error
            print("Iteration {}, current error: {} H^-s, {} l_2. Improvement {}".format(i+1, jnp.round(h_s_rel,5), jnp.round(l_2_rel,5), jnp.round(improvement, 5))) 

            # Update the rhs 
            u_pred_root = linear_solver.evaluate_solution_parallel(root_psi)
            linear_proj = compute_frechet_proj(u_pred_root, psi_matrix, tau_prime)
            residual = f_meas - linear_solver.non_linear_mes
            linear_part = linear_solver.evaluate_solution_L_psi(root_b) + linear_proj
            rhs_meas = lr*residual + linear_part

            # # lr = 1.0 case
            # root_values = tau_prime(u_pred_root)*u_pred_root - tau(u_pred_root)
            # rhs_1 = f_meas +  vmap_integrate_f_test_functions(root_values, psi_matrix)

            # print(jnp.sqrt(jnp.sum((rhs_1 - rhs_meas)**2)))

            # Update the operator by updating the roots of b
            root_linearization = b(root_psi) +  tau_prime(u_pred_root)
        
        if lr < 1e-10:
            print("Learning rate too small, exiting")
            break
        if improvement < 1e-4:
            print("Improvement too small, exiting")
            break

    
    return linear_solver

def Gauss_Newton_solver(L_stiff, f_meas, b, tau, tau_prime, psi_matrix, root_psi,linear_solver, reg, n_iter = 10, verbose = True):
    """
    This Gauss Newton solver only accepts lr rates of 1.0 and breaks if the error increases.
    """
    rhs_meas = jnp.copy(f_meas)
    root_b = b(root_psi)
    root_linearization = jnp.copy(root_b)
    prev_error = jnp.inf


    for i in range(n_iter):
        # Do a linear solve
        linear_solver.solve_linear_prob(rhs_meas, reg, root_linearization, L_stiff)
        # Compute the residuals
        l_2_error, l_2_rel, h_s_error, h_s_rel = linear_solver.compute_residuals_nl(f_meas, tau, root_b, L_stiff)
        if h_s_error > prev_error:
            if verbose:
                print("Increase in error, exiting")
            break
        else:
            improvement = jnp.abs(prev_error - h_s_error)/jnp.abs(prev_error)
            prev_error = h_s_error
            if verbose:
                print("Iteration {}, current error: {} H^-s, {} l_2. Improvement {}".format(i+1, jnp.round(h_s_rel,5), jnp.round(l_2_rel,5), jnp.round(improvement, 5))) 

            # Update the rhs 
            u_pred_root = linear_solver.evaluate_solution_parallel(root_psi)
            root_values = tau_prime(u_pred_root)*u_pred_root - tau(u_pred_root)
            rhs_meas = f_meas +  vmap_integrate_f_test_functions(root_values, psi_matrix)

            root_linearization = b(root_psi) +  tau_prime(u_pred_root)
        
        if improvement < 1e-5:
            if verbose:
                print("Improvement too small, exiting")
            break

    
    return linear_solver

class kernel_linear_solver():
    def __init__(self, length_scale, psi_matrix, root_psi, boundary,boundary_conditions, nu):

        self.length_scale = length_scale

        self.psi_matrix = psi_matrix
        self.root_psi = root_psi
        self.boundary_conditions = boundary_conditions
        self.boundary = boundary

        self.nu = nu
        self.n_meas = psi_matrix.shape[0]

    def solve_linear_prob(self, rhs_meas, reg, root_b, L_stiff):
        self.root_b = root_b

        # Construct the matrices
        if self.n_meas > 2048:
            print(f"Too many measurements. Falling back to batched approach.")
            theta_11, theta_21, theta_22 = theta_blocks_chunked(self.boundary,self.psi_matrix, self.root_psi, self.length_scale, self.nu, root_b)
        else:
            try:
                theta_11, theta_21, theta_22 = theta_blocks(self.boundary,self.psi_matrix, self.root_psi, self.length_scale, self.nu, root_b)
            except Exception as e:
                print(f"vmap failed with error: {e}. Falling back to batched approach.")
                theta_11, theta_21, theta_22 = theta_blocks_chunked(self.boundary,self.psi_matrix, self.root_psi, self.length_scale, self.nu, root_b)
        
        theta_12 = theta_21.T
        K = jnp.block([[theta_11, theta_12], [theta_21, theta_22]])
        K_interior = jnp.vstack([theta_12, theta_22]).T
        K_bc = jnp.vstack([theta_11, theta_21]).T

        self.K, self.K_interior, self.K_bc = K, K_interior, K_bc

        # When there is no regularization, solve the linear system directly with a Cholesky decomposition
        if reg is None:
            c = solve_linear_system(K, rhs_meas, self.boundary_conditions)
        else:
            c = solve_qp(K, K_interior, K_bc, L_stiff, rhs_meas, self.boundary_conditions, reg)
        self.c = c

        #self.residuals= self.compute_residuals(K_interior, rhs_meas, L_stiff)
        self.meas = K_interior@c
    
    def build_matrices(self, root_b):
        self.root_b = root_b

        # Construct the matrices
        theta_11, theta_21, theta_22 = theta_blocks(self.boundary,self.psi_matrix, self.root_psi, self.length_scale, self.nu, root_b)
        theta_12 = theta_21.T
        K = jnp.block([[theta_11, theta_12], [theta_21, theta_22]])
        K_interior = jnp.vstack([theta_12, theta_22]).T
        K_bc = jnp.vstack([theta_11, theta_21]).T

        self.K, self.K_interior, self.K_bc = K, K_interior, K_bc

        # Compute the nugget
        n_bc = self.boundary_conditions.shape[0]
        nugget_bc = jnp.ones(n_bc)*1e-10
        nugget_interior = jnp.ones(K.shape[0]- n_bc)*1e-8
        nugget = jnp.hstack([nugget_bc, nugget_interior])
        self.L_K = scipy.linalg.cho_factor(K + jnp.diag(nugget), lower = True)

    def solve(self, rhs_meas, reg, L_stiff, compute_residuals = True):

        # When there is no regularization, solve the linear system directly with a Cholesky decomposition
        if reg is None:
            c = cho_solve_system(self.L_K, rhs_meas, self.boundary_conditions)
        else:
            c = solve_qp(self.K, self.K_interior, self.K_bc, L_stiff, rhs_meas, self.boundary_conditions, reg)
        self.c = c
        if compute_residuals:
            self.residuals= self.compute_residuals(self.K_interior, rhs_meas, L_stiff)
            self.meas = self.K_interior@c

    

    def evaluate_solution(self, x):
        return evaluate_prediction(x, self.c, self.length_scale, self.root_psi, self.psi_matrix, self.boundary, self.nu, self.root_b)
    
    def evaluate_solution_parallel(self, x):
        return vmap_evaluate_prediction(x, self.c, self.length_scale, self.root_psi, self.psi_matrix, self.boundary, self.nu, self.root_b)
    
    # def evaluate_solution_psi(self):
    #     pred_root = self.evaluate_solution_parallel(self.root_psi)
    #     return vmap_integrate_f_test_functions(pred_root, self.psi_matrix)

    def create_K_eval(self, x):
        self.K_eval = build_K_eval(x, self.length_scale, self.root_psi, self.psi_matrix, self.boundary, self.nu, self.root_b)
    
    def vmap_create_K_eval(self, x):
        self.K_eval_vmap = vmap_K_eval(x, self.length_scale, self.root_psi, self.psi_matrix, self.boundary, self.nu, self.root_b)

    def evaluate_solution_predef(self, x):
        # Check wether K_predef has been created
        if not hasattr(self, 'K_eval_vmap'):
            self.vmap_create_K_eval(x)
        return self.K_eval_vmap@self.c

    
    def create_K_psi(self):
        K_psi = vmap_K_psi(self.root_psi, self.length_scale, self.root_psi, self.psi_matrix, self.boundary, self.nu, self.root_b)
        #bp()
        K_psi = jnp.sum(K_psi*self.psi_matrix[None :, None],axis = 1)
        self.K_psi = K_psi

    
    def evaluate_solution_psi(self):
        # Check wether K_psi has been created
        if not hasattr(self, 'K_psi'):
            self.create_K_psi()

        return self.K_psi@self.c

    
    def evaluate_solution_L_psi(self, b_L):
        K_bc =  jnp.squeeze(vmap_linear_form_K(self.psi_matrix, self.boundary, self.root_psi, self.length_scale, self.nu, b_L), axis = -1)
        # For some reason the arguments are flipped 
        K_interior = vmap_bilinear_form_K(self.psi_matrix, self.psi_matrix, self.root_psi, self.root_psi, self.length_scale, self.nu, self.root_b, b_L)
        #K_interior = vmap_bilinear_form_K(self.psi_matrix, self.psi_matrix, self.root_psi, self.root_psi, self.length_scale, self.nu, b_L, self.root_b)
        K_eval = jnp.hstack([K_bc, K_interior])
        self.K_eval_L_psi = K_eval
        return K_eval@self.c
    
    def compute_residuals_nl(self, rhs_meas, tau, root_b, L_stiff):

        non_linear = tau(self.evaluate_solution_parallel(self.root_psi))
        non_linear = vmap_integrate_f_test_functions(non_linear, self.psi_matrix)

        nl_meas = self.evaluate_solution_L_psi(root_b) + non_linear
        self.non_linear_mes = nl_meas

        residual= nl_meas + non_linear -rhs_meas

        # Compute the l_2 norm of the rhs
        l_2_rhs = jnp.sqrt(jnp.sum(rhs_meas**2))
        l_2_error = jnp.sqrt(jnp.sum(residual**2))
        l_2_rel = l_2_error/l_2_rhs

        # Compute the H_s norm
        h_s_error = jnp.sqrt((residual)@scipy.linalg.cho_solve(L_stiff, residual))
        h_s_rhs = jnp.sqrt((rhs_meas)@scipy.linalg.cho_solve(L_stiff, rhs_meas))
        h_s_rel = h_s_error/h_s_rhs


        return l_2_error, l_2_rel, h_s_error, h_s_rel

    def compute_residuals(self, K_interior, rhs_meas, L_stiff):
        residual = K_interior@self.c -rhs_meas

        # Compute the l_2 norm of the rhs
        l_2_rhs = jnp.sqrt(jnp.sum(rhs_meas**2))
        l_2_error = jnp.sqrt(jnp.sum(residual**2))
        l_2_rel = l_2_error/l_2_rhs

        h_s_error = jnp.sqrt((residual)@scipy.linalg.cho_solve(L_stiff, residual))
        h_s_rhs = jnp.sqrt((rhs_meas)@scipy.linalg.cho_solve(L_stiff, rhs_meas))
        h_s_rel = h_s_error/h_s_rhs

        self.residuals = residual
        return l_2_error, l_2_rel, h_s_error, h_s_rel
    
    def compute_error(self, x_error, w_error, u_error):
        pred_error = evaluate_prediction(x_error, self.c, self.length_scale, self.root_psi, self.psi_matrix, self.boundary, self.nu, self.root_b)
        error = compute_error(pred_error, u_error, w_error)
        return error
