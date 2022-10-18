from Newton import NewtonSolver
from EllipticPDE import NonlinearEllipticPDE
from jax import grad, jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from FDSoln import NewtonFDSolver

# non-linear part of PDE
@jit
def tau(u):
    return u**3

# define parameters
maxNrv = 200
Nquad  = 250
Ntest  = 200

# define fixed inputs
Xquad  = jnp.linspace(0,1,Nquad)[:,None]
Wquad  = jnp.ones(shape=(Nquad,1))/Nquad
Wquad  = Wquad.at[0].set(1/(2.*Nquad))
Wquad  = Wquad.at[-1].set(1/(2.*Nquad))

# define random input
from jax import random
rhs_rv = random.normal(random.PRNGKey(0), shape=(maxNrv,1))
Nrv_grid = [5,10,20,50,100,200]

plt.figure(figsize=(8,3))
# define test grid for solution and residual at test points
N_pts = 200
xx = jnp.linspace(0, 1, N_pts)[:,None]

for Nrv in Nrv_grid:

    # setup PDE solver
    solver = NonlinearEllipticPDE(tau, Xquad, Wquad, Ntest, rhs_rv[:Nrv,:], alpha=1.)
    # run Newton's method
    solver = NewtonSolver(solver, Xquad)

    # run reference solution
    FDsoln = NewtonFDSolver(Xquad, tau, solver)

    # plot solution
    plt.subplot(1,3,1)
    plt.plot(xx, solver.rhs(xx))
    plt.subplot(1,3,2)
    plt.plot(xx, solver.MAP(xx), label='N = '+str(Nrv))
    plt.subplot(1,3,3)
    Xquad_p = jnp.linspace(0,1,Nquad+1)[:,None]
    plt.plot(Xquad_p, jnp.abs(solver.MAP(Xquad_p) - FDsoln))

# add labels
plt.subplot(1,3,1)
plt.xlim(0,1)
plt.xlabel('$x$')
plt.ylabel('$f(x) = \sum_{n=1}^{N} n^{\\alpha}\\xi_n\\phi_n(x)$')
plt.subplot(1,3,2)
plt.xlim(0,1)
plt.xlabel('$x$')
plt.ylabel('$u^*(x)$')
plt.legend()
plt.subplot(1,3,3)
plt.xlim(0,1)
plt.xlabel('$x$')
plt.ylabel('$|u^*(x) - u^{FD}(x)|$')
# plt.ylabel('$r (x) = -\Delta u(x) + \\tau(u)(x) - f(x)$')
plt.tight_layout()
plt.savefig('alpha1.pdf')
plt.show()