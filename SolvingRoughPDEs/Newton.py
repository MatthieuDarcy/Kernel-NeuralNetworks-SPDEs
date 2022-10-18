import jax.numpy as jnp

def NewtonSolver(solver, Xquad):

    # initialize solution
    def u_n(x):
        return jnp.zeros(shape=(len(x),1))
    
    # define parameters for iterative solvers
    tol = 1e-9
    max_iter = 10

    # initialize termination criteria and counter
    delta_u  = jnp.inf
    ctr = 0
    
    # run Newton's method with GP
    while(delta_u > tol and ctr < max_iter):
        # solve for new solution
        solver.solve(u_n)
        # evaluate change in solution
        delta_u = jnp.linalg.norm(u_n(Xquad) - solver.MAP(Xquad))
        print('GP iter %d: Delta u = %f' % (ctr, delta_u))
        # update u_n and counter
        u_n = lambda x: solver.MAP(x)
        ctr += 1

    return solver