import jax.numpy as jnp
from jax import random, vmap, jit, grad
import jax.scipy as scipy
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from utils_rough_pde import *
from utils_elliptic_coef import *

import time

from utils_optim import kernel_linear_solver

########################################################################################################################

import argparse

parser = argparse.ArgumentParser(description='Compute the convergence rates for the linear and semilinear elliptic PDEs')
parser.add_argument('--n_max', type=int, help='Maximum number of measurements (in log 2 scale)')
parser.add_argument('--n_min', type=int, help='Maximum number of measurements (in log 2 scale)')
parser.add_argument('--n_order', type=int, help='Order of the quadrature rule')
parser.add_argument('--name', type=str, help='Name of the experiment')
n_max = parser.parse_args().n_max
n_min = parser.parse_args().n_min
n_order = parser.parse_args().n_order
name = parser.parse_args().name




meas_exp = jnp.arange(n_min, n_max+1)

print(meas_exp)


# Define the domain of the problem
lower = 0.0
upper = 1.0
L = upper - lower
x = jnp.linspace(0, 1, 1000)
boundary = jnp.array([[lower, upper]]).T
boundary_conditions = jnp.zeros(shape = (2, )) # Dirichlet boundary conditions
domain = jnp.array([lower, upper])


########################################################################################################################
# Linear PDE

# Create the RHS

nu = 1e-3
b = lambda x: jnp.ones_like(x)  # Coefficinet in front of the u in the PDE (constant here)

# draw random values for the coefficients
#key = random.PRNGKey(54)
key = random.PRNGKey(11)
n_coef = 500

s = 1 # H^s
alpha = 2*s+1 + 0.1
decay_u = alpha/2
L = 1 # Lenght of the domain
coef_u =  jnp.ones(shape = (1, ))/(jnp.arange(1, n_coef+1)**(decay_u)) #jnp.ones(shape = (1, ))#random.normal(key, shape=(n_coef,))/(jnp.arange(1, n_coef+1)**(decay_u))
u_lambda = lambda x: evaluate_function(x, coef_u, L=L)

coef_u = random.normal(key, shape=(n_coef,))/(jnp.arange(1, n_coef+1)**(decay_u))
coef_f_constant = coef_u*(nu*jnp.arange(1, n_coef+1)**(2)*jnp.pi**2*L**2)

# Parameters of the kernel
length_scale = 0.1
reg = None




error = []
error_rel = []
for i in meas_exp:
    n_meas = 2**i
    m_tool = measurement_tool(domain, n_meas, n_order=n_order)
    root_b = m_tool.evaluate_at_roots(b)
    u_error = m_tool.evaluate_for_error(u_lambda)

    # Project f against the test functions
    f_quad = evaluate_function(m_tool.root_psi, coef_f_constant, L) + b(m_tool.root_psi)*evaluate_function(m_tool.root_psi, coef_u, L)
    f_meas = m_tool.project(f_quad)

    # Solve the linear problem
    solver = kernel_linear_solver(length_scale, m_tool.psi_matrix, m_tool.root_psi, boundary, boundary_conditions, nu)
    solver.solve_linear_prob(f_meas, reg, root_b, m_tool.L_stiff)

    loss, relative_loss = solver.compute_error(m_tool.x_error, m_tool.w_error, u_error)
    error.append(loss)
    error_rel.append(relative_loss)

error = jnp.array(error)
error_rel = jnp.array(error_rel)


# Estimate the convergence rate by fitting a line to the log-log plot of the error
log_n_meas = jnp.log(2**meas_exp)
log_error = jnp.log(error)

a = jnp.hstack([log_n_meas.reshape(-1, 1), jnp.ones_like(log_n_meas.reshape(-1, 1))])
b = log_error
r, C = jnp.linalg.lstsq(a, b)[0]
r, C = -r.item(), jnp.exp(C).item()

print("L^2 Convergence rate: ", jnp.round(r,3)) 

# Plot the convergence rates 

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(2**meas_exp, error, label = r"$r :{:.03f}$".format(r))
ax[0].scatter(2**meas_exp, error)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlabel("Number of measurements")
ax[0].set_ylabel(r"$||u^\dagger - u^*||_{L^2}$")
ax[0].set_title(r"Convergence of the $L^2$ error")
ax[0].legend()

ax[1].plot(2**meas_exp, error_rel, label = r"$r :{:.03f}$".format(r))
ax[1].scatter(2**meas_exp, error_rel)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel("Number of measurements")
ax[1].set_title(r"Convergence of the relative $L^2$ error")
ax[1].set_ylabel(r"$\frac{||u^\dagger - u^*||_{L^2}}{||u^*||_{L^2}}$")

# Save the plot
plt.savefig("L2_convergence_elliptic_{}.png".format(name))


########################################################################################################################
# Semilinear PDE
tau = lambda x : x**3
# We hardcode the derivative of tau (one could use autograd to compute it)
tau_prime = lambda x : 3*x**2
b = lambda x: jnp.ones_like(x)


nu = 1e-3
# draw random values for the coefficients
n_coef = 500
s = 1 # H^s
alpha = 2*s+1 + 0.1
decay_u = alpha/2
L = 1 # Lenght of the domain
coef_u =  jnp.ones(shape = (1, ))/(jnp.arange(1, n_coef+1)**(decay_u))
coef_u = random.normal(key, shape=(n_coef,))/(jnp.arange(1, n_coef+1)**(decay_u))
coef_f_linear = coef_u*(nu*jnp.arange(1, n_coef+1)**(2)*jnp.pi**2*L**2)

u_lambda = lambda x: evaluate_function(x, coef_u, L=L)
f_lambda = lambda x: evaluate_function(x, coef_f_linear, L=L) + b(x)*evaluate_function(x, coef_u, L) + tau(evaluate_function(x, coef_u, L))


from utils_optim import Gauss_Newton_solver_variable, Gauss_Newton_solver

# Kernel parameters
length_scale = 0.05
reg = None

error = []
error_rel = []
for i in meas_exp:
    n_meas = 2**i
    m_tool = measurement_tool(domain, n_meas, n_order=n_order)
    root_b = m_tool.evaluate_at_roots(b)
    u_error = m_tool.evaluate_for_error(u_lambda)

    # Project f against the test functions
    f_quad = f_lambda(m_tool.root_psi)
    f_meas = m_tool.project(f_quad)

    # Solve the linear problem
    linear_solver = kernel_linear_solver(length_scale, m_tool.psi_matrix, m_tool.root_psi, boundary, boundary_conditions, nu)
    solver_opt = Gauss_Newton_solver(m_tool.L_stiff, f_meas, b, tau, tau_prime, m_tool.psi_matrix, m_tool.root_psi, linear_solver, reg, n_iter = 50, verbose = False)


    loss, relative_loss = solver_opt.compute_error(m_tool.x_error, m_tool.w_error, u_error)
    error.append(loss)
    error_rel.append(relative_loss)


error = jnp.array(error)
error_rel = jnp.array(error_rel)


# Estimate the convergence rate by fitting a line to the log-log plot of the error
log_n_meas = jnp.log(2**meas_exp)
log_error = jnp.log(error)

k = jnp.hstack([log_n_meas.reshape(-1, 1), jnp.ones_like(log_n_meas.reshape(-1, 1))])
l = log_error
r, C = jnp.linalg.lstsq(k, l)[0]
r, C = -r.item(), jnp.exp(C).item()

print("L^2 Convergence rate: ", jnp.round(r,3)) 

# Plot the convergence rates 

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(2**meas_exp, error, label = r"$r :{:.03f}$".format(r))
ax[0].scatter(2**meas_exp, error)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xlabel("Number of measurements")
ax[0].set_ylabel(r"$||u^\dagger - u^*||_{L^2}$")
ax[0].set_title(r"Convergence of the $L^2$ error")
ax[0].legend()

ax[1].plot(2**meas_exp, error_rel, label = r"$r :{:.03f}$".format(r))
ax[1].scatter(2**meas_exp, error_rel)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel("Number of measurements")
ax[1].set_title(r"Convergence of the relative $L^2$ error")
ax[1].set_ylabel(r"$\frac{||u^\dagger - u^*||_{L^2}}{||u^*||_{L^2}}$")

# Save the plot
plt.savefig("L2_convergence_semilinear_{}.png".format(name))

