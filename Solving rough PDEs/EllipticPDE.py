from jax import grad, jit, vmap, lax
from functools import partial
import jax.numpy as jnp

#from jax import config
#config.update("jax_enable_x64", True)

import kernels as k

class NonlinearEllipticPDE:
    """  PDE: -\Delta u + tau(u) = f on domain \Omega = [0,1] 
              where f(x) = \sum_{j} j^\alpha * xi_j * \phi_j(x)
    """
    def __init__(self, tau, Xquad, Wquad, Ntest, rhs_rv, alpha=-1):
        # set nonlinearity for PDE
        self.tau    = tau
        # assign quadrature points on [0,1]
        self.Xquad  = Xquad
        self.Wquad  = Wquad
        self.Nquad  = Xquad.shape[0]
        # set number of test functions
        self.Ntest  = Ntest
        # set rhs parameters
        self.rhs_rv = rhs_rv
        self.Nrv    = rhs_rv.shape[0]
        self.alpha  = alpha
        # define kernel and nugget parameter
        self.kernel = k.matern_kernel_3 #k.gaussian_kernel
        self.nugget = 0.0001

    def solve(self, u_n):
        # setup and invert kernel matrix
        Xquad_int = self.Xquad[1:-1,:]
        K = self.kernel(Xquad_int, Xquad_int)
        self.Kinv = jnp.linalg.inv(K + self.nugget*jnp.eye(self.Nquad-2))
        # define linear system
        A,b = self.linear_system(u_n)
        # solve for Zquad = v(Xquad) and Lagrange multiplers
        KA = jnp.block([[self.Kinv, A.T],[A, jnp.zeros(shape=(self.Ntest, self.Ntest))]]) 
        bz = jnp.block([[jnp.zeros(shape=(self.Nquad-2,1))], [b]])
        Z_lambda = jnp.linalg.solve(KA, bz)
        # extract Z evaluations
        Zquad_int = Z_lambda[:self.Nquad-2]
        # add boundary conditions
        Zquad_wb  = jnp.zeros(shape=(self.Nquad,1))
        Zquad_wb  = Zquad_wb.at[1:-1].set(Zquad_int)
        # save in Zquad
        self.Zquad = Zquad_wb

    def rhs(self, x):
        rhs_coeff = jnp.arange(1,self.Nrv+1,dtype=float)**self.alpha * self.rhs_rv[:,0]
        basis = self.testfunc(x, self.Nrv)
        return jnp.dot(basis, rhs_coeff)

    def testfunc(self, x, Nmax):
        # define grid of basis functions
        kgrid = jnp.arange(1, Nmax+1)
        # evaluate basis
        return jnp.sqrt(2.) * jnp.sin(jnp.pi*kgrid*x)

    def hess_testfunc(self, x, Nmax):
        # define grid of basis functions
        kgrid = jnp.arange(1, Nmax+1)
        # evaluate hessian of basis
        return -1.* (jnp.pi * kgrid)**2 * jnp.sqrt(2.) * jnp.sin(jnp.pi*kgrid*x)

    def int_rhs_testfunc(self, k):
        # Compute \int f(x) * psi_k(x) dx exactly
        # Inner product is non-zero only for k <= self.Nrv
        return jnp.where(k <= self.Nrv, (k*1.)**self.alpha * self.rhs_rv[k-1,:], 0.0)

    def linear_system(self, u_n):
        # evaluate all test functions and L(psi) at collocation points
        psi = self.testfunc(self.Xquad, self.Ntest)
        Lst_psi = -1*self.hess_testfunc(self.Xquad, self.Ntest)
        Lst_psi = Lst_psi.at[-1,:].set(jnp.zeros(shape=(self.Ntest,)))
        # evaluate derivative of non-linear part of PDE
        tau_prime = vmap(grad(self.tau))(u_n(self.Xquad)[:,0])[:,None]
        # evaluate Taylor series expansion of non-linear part tau at u_n
        linearizedPDE_lhs = tau_prime * u_n(self.Xquad) - self.tau(u_n(self.Xquad))
        # setup linear system for solution v(Xquad)
        A = (Lst_psi.T + (psi.T * tau_prime.T) ) * self.Wquad.T
        b = vmap(self.int_rhs_testfunc)(jnp.arange(1,self.Ntest+1)) + jnp.dot(psi.T * self.Wquad.T, linearizedPDE_lhs)
        # remove boundary conditions (first and last columns of A)
        A = A[:,1:-1]
        return A, b

    @partial(jit, static_argnums=(0,))
    def MAP(self, xtest):
        # evaluate kernel at evaluation points
        Kx = self.kernel(xtest, self.Xquad) 
        # compute inner product with kernel evaluated at test points
        Kinv = jnp.linalg.inv(self.kernel(self.Xquad, self.Xquad) + self.nugget*jnp.eye(self.Nquad))
        #Kinv = self.Kinv
        return jnp.dot(Kx, jnp.dot(Kinv, self.Zquad))

    def residual(self, xtest):
        """ Evaluate residual of strong form on a grid """
        u = lambda x: self.MAP(x)[0,0]
        return -1*vmap(grad(grad(u)))(xtest[:,0]) + vmap(self.tau)(self.MAP(xtest[:,0]))[:,0] - self.rhs(xtest)

if __name__=='__main__':

    # non-linear part of PDE
    @jit
    def tau(u):
        return u**3

    # define parameters
    Nrv    = 200#50
    Nquad  = 200
    Ntest  = 150

    # define inputs
    from jax import random
    rhs_rv = random.normal(random.PRNGKey(0), shape=(Nrv,1))
    Xquad  = jnp.linspace(0,1,Nquad)[:,None]
    Wquad  = jnp.ones(shape=(Nquad,1))/Nquad
    Wquad  = Wquad.at[0].set(1/(2.*Nquad))
    Wquad  = Wquad.at[-1].set(1/(2.*Nquad))

    # setup PDE solver
    solver = NonlinearEllipticPDE(tau, Xquad, Wquad, Ntest, rhs_rv)

    # define parameters for iterative solvers
    tol = 1e-9
    max_iter = 10

    # initialize solution
    def u_n(x):
        return jnp.zeros(shape=(len(x),1))
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
 
    # run FD solver
    import numpy as onp
    from FDSoln import NewtonFDSolver, NonlinearElliptic_FDresidual
    uFD_n = NewtonFDSolver(Xquad, tau, solver)

    # define test grid for solution and residual at test points
    N_pts = 200
    xx = jnp.linspace(0, 1, N_pts)[:,None]

    # plot solution
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(xx, solver.rhs(xx))
    plt.xlim(0,1)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$ ')

    plt.subplot(1,3,2)
    plt.plot(xx, solver.MAP(xx), label='GP')
    plt.plot(onp.linspace(0,1,Nquad+1), uFD_n, label='FD')
    plt.xlim(0,1)
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$ ')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(xx, solver.residual(xx), label='GP')
    plt.plot(onp.linspace(0,1,Nquad+1), NonlinearElliptic_FDresidual(Nquad, tau, solver.rhs, uFD_n), label='FD')
    plt.xlim(0,1)
    plt.ylim(-5,5)
    plt.xlabel('$x$')
    plt.ylabel('$r(x)$ ')
    plt.show()