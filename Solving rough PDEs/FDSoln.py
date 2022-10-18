import numpy as onp
import scipy.sparse
from scipy.sparse import diags
from jax import vmap, grad

def NonlinearElliptic_FDsolver(N, tau, f, u_n_x):

    # define grid: N+1 points with interval size 1/N
    delta_x = 1./N
    x_grid = (onp.arange(0,N+1,1))*delta_x

    # evaluate derivative of tau on grid
    u_n_x = u_n_x[:,0]
    tau_prime = vmap(grad(tau))(u_n_x)

    # diagonal and off-diagonals of A
    a_diag = 2*onp.ones((N-1,))/(delta_x**2) + tau_prime[1:N]
    a_offdiag = -1*onp.ones((N-1,))/(delta_x**2)
    # assemble A
    A = diags([a_offdiag, a_diag, a_offdiag], [-1,0,1], shape=(N-1, N-1))
    
    # assemble right-hand-side
    b = vmap(f)(x_grid) - vmap(tau)(u_n_x) + tau_prime * u_n_x

    # solve for solution
    b = b[1:N, onp.newaxis]
    sol_u = scipy.sparse.linalg.spsolve(A, b)

    # set boundary conditions
    sol_u_plus_bd = onp.zeros((N+1,1))
    sol_u_plus_bd[1:N,:] = sol_u[:,None]

    return sol_u_plus_bd

def NonlinearElliptic_FDresidual(N, tau, f, u_n_x):

    # define grid: N+1 points with interval size 1/N
    delta_x = 1./N
    x_grid = (onp.arange(0,N+1,1))*delta_x

    # diagonal and off-diagonals of A
    a_diag = 2*onp.ones((N+1,))/(delta_x**2)
    a_offdiag = -1*onp.ones((N+1,))/(delta_x**2)
    # assemble A
    A = diags([a_offdiag, a_diag, a_offdiag], [-1,0,1], shape=(N+1, N+1))
    # compute Laplacian of u
    Au = A.dot(u_n_x)

    # evaluate residual
    return Au + vmap(tau)(u_n_x) - vmap(f)(x_grid)[:,None]

def NewtonFDSolver(Xquad, tau, solver):

    # define parameters for iterative solvers
    tol = 1e-9
    max_iter = 10

    # initialize solution
    Nquad = Xquad.shape[0]
    uFD_n = onp.zeros(shape=(Nquad+1,1))

    # initialize termination criteria and counter
    delta_u  = onp.inf
    ctr = 0
    
    # run Newton's method with finite-difference
    while(delta_u > tol and ctr < max_iter):
        # solve for new solution
        uFD_np1 = NonlinearElliptic_FDsolver(Nquad, tau, solver.rhs, uFD_n)
        # evaluate change in solution
        delta_u = onp.linalg.norm(uFD_n - uFD_np1)
        print('FD iter %d: Delta u = %f' % (ctr, delta_u))
        # update uFD_n and counter
        uFD_n = uFD_np1
        ctr += 1

    return uFD_n