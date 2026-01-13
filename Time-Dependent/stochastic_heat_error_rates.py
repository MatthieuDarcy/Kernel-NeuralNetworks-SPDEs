# %%
import os
import sys

current_working_directory = os.getcwd()
project_directory = os.path.dirname(current_working_directory)
utils_path = os.path.join(project_directory, 'utils')
plotting_path = os.path.join(project_directory, 'plotting_templates')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# %%
import time
import jax
jax.config.update("jax_enable_x64", True) # Enable 64-bit double precision

# Imports
import jax.numpy as jnp
from jax import random, vmap, scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utilities
from utils_DST import discrete_sine_transform, vmap_dst, vmap_compute_sine_coef
from utils_error import compute_2d_error
from utils_rough_pde import *
from utils_elliptic_coef import *
from utils_optim import kernel_linear_solver

from utils_rough_pde import vmap_root_interval
from scipy.special import roots_legendre

from utils_time_dependent import *


plt.style.use(plotting_path + '/plot_style-Examples_3d.txt')
width = 4
height = width * 3 / 4

# %%
def compute_1d_error(pred, u, x):
    norm_u = jnp.sqrt(scipy.integrate.trapezoid(u**2, x))
    norm_diff = jnp.sqrt(scipy.integrate.trapezoid((pred - u)**2, x))
    return norm_diff, norm_diff/norm_u

# %%
plt.style.use(plotting_path + '/plot_style-Examples_3d.txt')
width = 4
height = width * 3 / 4

# %%
key = random.PRNGKey(12)


n_coef = 2**11
upper = 1.0
lower = 0.0
boundary = jnp.array([lower, upper])
boundary_conditions = jnp.array([0.0, 0.0])

domain_finest = jnp.linspace(lower,upper,n_coef+1, endpoint=False)[1:] # Do not include 0 or 1!
print(n_coef)

frequencies = jnp.arange(0, n_coef) + 1

# This is the initial condition
coef_g = -jnp.hstack([random.normal(key, shape= (n_coef,))])/(0.01*frequencies**(2+0.5)*jnp.pi**2  +1)
g_values = discrete_sine_transform(coef_g)

# %%
# Create 2 plots for the  function u and f
fig = plt.figure(figsize=(width, height))
plt.plot(domain_finest, g_values)
plt.xlabel(r"$x$")
plt.ylabel(r"$g(x)$")
plt.title(r"Initial condition")

plt.gca().autoscale()  # This applies to the current axis

plt.show()

# %%
def convolve(K, Z):
    return scipy.signal.convolve(K, Z, mode='full')[:Z.shape[0]]

def OU_explicit(time_span, h, u_init, beta, sigma, BM):
    """
    time span should not contain the time 0.0
    """

    # Create the convolution kernel
    K = jnp.exp(-beta*time_span)
    u = u_init*jnp.exp(-beta*time_span) + sigma*jnp.sqrt(h)*convolve(K, BM)

    return jnp.hstack([u_init, u])

OU_system = vmap(OU_explicit, in_axes=(None, None, 0, 0, None, 0))

#%%

nu = 0.025
beta = nu*jnp.pi**2*jnp.arange(1, n_coef+1)**2

h_spectral  =(2**(-13))
T = 1.0 //h_spectral * h_spectral #h_spectral*2**13
print(T, h_spectral)
sigma = 0.1

time_span = jnp.arange(0, T +h_spectral, h_spectral)
key = random.PRNGKey(0)
BM = random.normal(key, (n_coef, time_span.shape[0]-1))
#%%
u_hist = OU_system(time_span[1:], h_spectral, coef_g, beta, sigma, BM).block_until_ready()
u_values_T = vmap_dst(u_hist.T).block_until_ready()

print(u_values_T.shape)

#%% Save the fine solution
import pickle
with open('stochastic_heat_equation_fine_solution.pkl', 'wb') as f:
    pickle.dump(u_values_T, f)
    pickle.dump(domain_finest, f)
    pickle.dump(time_span, f)
# %%
coarse_factor = 4 # Coarsen the fine solution by a factor of 4
h = h_spectral*coarse_factor

time_span = jnp.arange(0, T+h, h)
time_span_spectral = jnp.arange(0, T+h_spectral, h_spectral) 
b =lambda x : jnp.ones_like(x)
length_scale = 0.1
cfl_factor = 5

print("We integrate the system with a time step of ", h)

#%%
# Generate our Galerkin basis
n_intervals = int(jnp.sqrt(1/h)*cfl_factor) + 1#2**10
n_meas = n_intervals -1 # We do not include the boundary points

epsilon = (upper- lower)/n_intervals
centers = jnp.linspace(lower, upper, n_intervals + 1)
epsilon = (upper - lower)/n_intervals
epsilon_values  = jnp.ones(n_intervals)*epsilon
intervals = jnp.array([centers[:-1], centers[1:]]).T
print("We use {} measurements (tent functions spaced out by a factor h = {:.2e})".format(n_meas, epsilon))

# %%
tent_values = vmap_tent(domain_finest, epsilon, centers[1:-1]) # Evaluate the tent functions on the fine grid
tent_proj_coef = vmap_compute_sine_coef(tent_values)  # Project the tent functions onto the sine basis
g_tent = tent_proj_coef@coef_g # We can now project the initial condition onto the tent functions

# Create 2 plots for the  function u and f
fig = plt.figure(figsize=(width, height))
plt.plot(domain_finest, g_values, label = r"Exact")
plt.plot(centers[1:-1], g_tent, label = r"Projected")
plt.xlabel(r"$x$")
plt.ylabel(r"$g(x)$")
plt.title(r"Initial condition")
plt.legend(loc = 'upper right')

plt.gca().autoscale()  # This applies to the current axis

plt.show()

#%%
n_quad = 3
x_q, w_q = roots_legendre(n_quad)
# Quadrature node for the tent element
x_element, w_element = root_interval(x_q, w_q, jnp.array([0, 1]))
# Quadrature node for the kernel
x_quad, w_quad = vmap_root_interval(x_q, w_q, intervals)
k_quad = []
for i in range(n_meas):
    k_quad.append(jnp.hstack([x_quad[i], x_quad[i+1]]))
k_quad = jnp.array(k_quad)

root_b= b(k_quad)
element_quad = tent_element(x_element, normalization = 1.0)
tent_quad = jnp.hstack([element_quad, element_quad[::-1]])*jnp.hstack([w_element, w_element[::-1]])

# Coarsen the fine grid solution
BM_coarse = BM.reshape(n_coef, time_span_spectral[1:].shape[0] // coarse_factor, coarse_factor)
BM_coarse = jnp.sum(BM_coarse, axis = -1)
xi_tent_coarse = tent_proj_coef@BM_coarse*jnp.sqrt(h_spectral)
u_coarse = u_values_T[coarse_factor::coarse_factor]


# %% [markdown]
# # Computing error rates

# %%
h_list = jnp.array([2**i*h_spectral for i in range(2,13)])[::-1]
coarse_list = jnp.array([2**i for i in range(2, 10)])[::-1]

# %%

error_list = []
error_list_r = []
error_time = []
for coarse_factor in coarse_list:
    h = coarse_factor*h_spectral
    n_meas = int(jnp.sqrt(1/h)*cfl_factor)
    print("We integrate the system with a time step of ", h)
    print("We use {} measurements (tent functions spaced out by a factor h = {:.2e})".format(n_meas, (upper-lower)/ (n_meas +1)))
    #break
    # Generate our Galerkin basis
    n_intervals = n_meas +1
    n_meas = n_intervals -1 # We do not include the boundary points

    epsilon = (upper- lower)/n_intervals
    centers = jnp.linspace(lower, upper, n_intervals + 1)
    epsilon = (upper - lower)/n_intervals
    epsilon_values  = jnp.ones(n_intervals)*epsilon
    intervals = jnp.array([centers[:-1], centers[1:]]).T

    # Quadrature node for the kernel
    x_quad, w_quad = vmap_root_interval(x_q, w_q, intervals)
    k_quad = []
    for i in range(n_meas):
        k_quad.append(jnp.hstack([x_quad[i], x_quad[i+1]]))
    k_quad = jnp.array(k_quad)
    root_b= b(k_quad)

    # Project the initial condition on the Galerkin basis
    tent_values = vmap_tent(domain_finest, epsilon, centers[1:-1]) # Evaluate the tent functions on the fine grid
    tent_proj_coef = vmap_compute_sine_coef(tent_values)  # Project the tent functions onto the sine basis
    g_tent = tent_proj_coef@coef_g # We can now project the initial condition onto the tent functions

    # Coarsen the Wiener process
    BM_coarse = BM.reshape(n_coef, time_span_spectral[1:].shape[0] // coarse_factor, coarse_factor)
    BM_coarse = jnp.sum(BM_coarse, axis = -1)
    xi_tent_coarse = tent_proj_coef@BM_coarse*jnp.sqrt(h_spectral)
    # Create the linear solver
    print("Building the linear solver...")
    linear_solver = kernel_linear_solver(length_scale, tent_quad, k_quad,  boundary[:, None], boundary_conditions, h*nu)
    linear_solver.build_matrices(root_b)
    linear_solver.create_K_psi()
    linear_solver.create_K_eval(domain_finest)

    # Integrate
    print("Integrating...")
    time_span = jnp.arange(0, T+h, h)
    print(h, n_meas, time_span[1:].shape[0])
    y, linear_solver, c_kernel_history= implicit_EM_solver(time_span[1:], h, linear_solver, g_tent, sigma,xi_tent_coarse)

    pred_kernel = jnp.einsum('ij,kj->ki', linear_solver.K_eval, c_kernel_history)
    u_coarse = u_values_T[coarse_factor::coarse_factor]
    e, e_r = compute_2d_error(pred_kernel, u_coarse, domain_finest, time_span[1:])
    error_list.append(e)
    error_list_r.append(e_r)

    e_t , _  = vmap(compute_1d_error, in_axes=(0, 0, None))(pred_kernel, u_coarse, domain_finest)
    error_time.append(e_t)

# %%
error = jnp.array(error_list)
error_rel = jnp.array(error_list_r)

meas_list = []
for coarse_factor in coarse_list:
        h = coarse_factor*h_spectral
        n_meas = int(jnp.sqrt(1/h)*cfl_factor)
        meas_list.append(T/h*n_meas)
meas_list = jnp.array(meas_list)

# Estimate the convergence rate by fitting a line to the log-log plot of the error
log_n_meas = jnp.log(meas_list)
log_error = jnp.log(error)

a = jnp.hstack([log_n_meas.reshape(-1, 1), jnp.ones_like(log_n_meas.reshape(-1, 1))])
b_2 = log_error
r, C = jnp.linalg.lstsq(a, b_2)[0]
r, C = -r.item(), jnp.exp(C).item()

print("L^2 Convergence rate: ", jnp.round(r,3)) 

# %%
fig = plt.figure(figsize=(width, height))
plt.plot(meas_list, error, label = r"$r :{:.03f}$".format(r))
plt.scatter(meas_list, error)
plt.yscale('log')
plt.xscale('log')

plt.xlabel("Number of measurements")
plt.ylabel(r"$||u^\dagger - u^*||_{L^2([0, T], L^2(\Omega))}$")
plt.title(r"Convergence of the $L^2$ error")
plt.legend()

plt.show()


# %%
# Save the error vs time step data
import pickle
with open('stochastic_heat_equation_error_time.pkl', 'wb') as f:
    pickle.dump(error, f)
    pickle.dump(meas_list, f)

# %%
print("Saved the error vs time step data")
print(error)
print(meas_list)





