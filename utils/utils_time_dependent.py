import jax.numpy as jnp
from jax import jit, vmap

from tqdm import tqdm
def implicit_EM_solver(time_span, h, linear_solver, y_init,sigma, BM):
    y = y_init
    #noise = []
    c_history = []
    for i,t  in enumerate(time_span):
        # Sample from the space time white noise
        xi = BM[:, i]
        # Create the right hand side
        rhs = y + sigma*xi

        # Solve the linear system
        linear_solver.solve(rhs, None, None, compute_residuals = False)
        # Update the rhs
        y = linear_solver.evaluate_solution_psi()    
        c_history.append(linear_solver.c)    
        #print(y.shape)

    return y, linear_solver, jnp.array(c_history)


# The fundamental element
def tent_element(x, normalization = 1.0):
    element = jnp.maximum(0, 1 + (x - 1) )
    # Set the element to zero outside the interval
    element = jnp.where(x < 0.0, 0.0, element)
    element = jnp.where(x > 1.0, 0.0, element)

    return element/normalization

@jit
def integrate_f_test_functions(f_values, psi):
    # Psi should be a matrix containing the values of the weighted test functions at the quadrature points
    return f_values@psi
vmap_integrate_f_test_functions = jit(vmap(integrate_f_test_functions, in_axes=(0,None)))

from utils_DST import compute_sine_coefficients, discrete_sine_transform

def spectral_implicit_EM_solver(A, h, time_span, y_init, sigma, BM, tau):
    y = y_init
    n = y_init.shape[0]
    A_inv = jnp.diag((h*A + (1-h)*jnp.ones(n))**-1)

    #noise = []
    y_history = []
    print("Number of steps {}".format(time_span.shape[0]))
    for i,t  in tqdm(enumerate(time_span)):
        # Sample from the space time white noise
        xi = BM[:, i]

        # Evaluate the non linearity in the physical domain
        non_linear = tau(discrete_sine_transform(y))
        # Compute the Fourier coefficients
        non_linear = compute_sine_coefficients(non_linear)

        # Create the right hand side
        rhs = y + non_linear*h +jnp.sqrt(h)*sigma*xi
        y =A_inv@rhs
        y_history.append(y)


    return y, jnp.array(y_history)


def implicit_EM_solver_non_linear(time_span, h, linear_solver, y_init,non_linear_init, sigma, BM_proj, tau, k_quad, tent_quad):

    y = y_init
    non_linear = non_linear_init

    #noise = []
    c_history = []
    print("Number of steps {}".format(time_span.shape[0]))
    for i,t  in tqdm(enumerate(time_span)):
        # Sample from the space time white noise
        xi = BM_proj[:, i]


        # Create the right hand side
        #bp()
        rhs = y + h*non_linear+ jnp.sqrt(h)*sigma*xi
        #print(xi[0])

        # Solve the linear system
        linear_solver.solve(rhs, None, None, compute_residuals = False)

    

        c_history.append(linear_solver.c)    

        values_quad = linear_solver.evaluate_solution_predef(k_quad)
        # Project on the tent functions
        non_linear = integrate_f_test_functions(tau(values_quad), tent_quad)

        # Update the rhs
        y = linear_solver.evaluate_solution_psi()#integrate_f_test_functions(values_quad, tent_quad)


    return y, linear_solver, jnp.array(c_history)