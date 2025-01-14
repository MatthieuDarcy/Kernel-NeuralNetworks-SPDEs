import jax.numpy as jnp

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