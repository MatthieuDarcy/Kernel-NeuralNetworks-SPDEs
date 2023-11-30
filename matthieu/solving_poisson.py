
import jax.numpy as jnp
from jax import random
from jax import scipy 

import matplotlib.pyplot as plt

import utils_rough_pde
from utils_rough_pde import *

import jax 
devices = jax.devices()

# Print the list of devices
print("Available devices:", devices)

#jax.config.update('jax_platform_name', 'cpu')

# Ask the user the name of the folder where to save the results
save_folder = input("Name of the folder where to save the results: ") + "/"

# Ask the user the maximum number of measurements and the decay of the coefficients 
n_meas = int(input("Number of measurements: "))
decay_u = float(input("Decay of the coefficients: "))


seed = 1


import os

# Create the folder for saving the results
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# draw random values for the coefficients
key = random.PRNGKey(seed)

n_coef = 1000

L = 1 # Lenght of the domain
coef_u = random.normal(key, shape=(n_coef,))/(jnp.arange(1, n_coef+1)**(decay_u))

coef_f = coef_u*jnp.arange(1, n_coef+1)**(2)*jnp.pi**2*L**2

x = jnp.linspace(0, L, 1200)
u_values = evaluate_function(x, coef_u, L=L)
f_values = evaluate_function(x, coef_f, L=L)


# Create 2 plots for the  function u and f
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x, u_values)
ax[0].set_xlabel("x")
ax[0].set_ylabel("u(x)")
ax[0].set_title("Function u")

ax[1].plot(x, f_values)
ax[1].set_xlabel("x")
ax[1].set_ylabel("f(x)")
ax[1].set_title("Function f")

plt.savefig(save_folder+"u_f.png")



# Solve the Poisson equation with increasing number of measurements

n_meas_list = jnp.arange(10, n_meas + 10, 10, dtype=int)
# Boundary of the domain
lower, upper = 0.0, 1.0
length_scale = 0.1
boundary = jnp.array([[lower, upper]]).T

nugget_interior = 1e-6
nugget_boundary = 1e-10

error_list = []
relative_error_list = []

pred_list = []

from scipy.special import roots_legendre
n_order = 999
x_q, w_q = roots_legendre(n_order)

for n_meas in n_meas_list:

    # Construct the measurements
    epsilon_values =  jnp.array([1/(n_meas*2)])
    loc_values = jnp.linspace(lower + epsilon_values[0], upper - epsilon_values[0],  int(L/(2*epsilon_values[0])))
    support = jnp.array([loc_values - epsilon_values[0], loc_values + epsilon_values[0]]).T
    vol = support[:,1] - support[:,0]
    N_test_functions = loc_values.shape[0]

    print("Number of test functions: ", N_test_functions)

    # Construct the mmatrix of weighted bump functions 
    root_psi, w_psi = vmap_root_interval(x_q, w_q, support)
    psi_matrix = bump_vector(root_psi, epsilon_values, loc_values)
    psi_matrix = psi_matrix * w_psi

    # Compute the RHS of the linear system
    f_quad = evaluate_function(root_psi, coef_f, L)
    f_meas = vmap_integrate_f_test_functions(f_quad, psi_matrix)

    # Construct the RHS of the linear system
    Y = jnp.block([jnp.zeros(shape = 2), f_meas])

    # Compute the kernel matrix
    print("Constructing the kernel matrix")
    theta = construct_theta(boundary,psi_matrix, root_psi, length_scale)

    # Construct the nugget
    nugget = jnp.block([nugget_boundary*jnp.ones(shape = 2), nugget_interior*jnp.ones(shape = N_test_functions)])

    # Solve the linear system
    print("Solving the linear system")
    c = scipy.linalg.solve(theta + nugget*jnp.eye(theta.shape[0]), Y, assume_a='pos')

    # Compute the numerical solution
    pred = evaluate_prediction(x, c, length_scale, root_psi, psi_matrix, boundary)
    pred_list.append(pred)


    # Compute the error between the true solution and the numerical solution
    loss, relative_loss = compute_error(pred, u_values)
    error_list.append(loss)
    relative_error_list.append(relative_loss)



error_list = jnp.array(error_list)
relative_error_list = jnp.array(relative_error_list)

pred_list = jnp.array(pred_list)

# Plot both the error and the relative error
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(n_meas_list, error_list)
ax[0].scatter(n_meas_list, error_list)
ax[0].set_yscale("log")
ax[0].set_xlabel("Number of measurements")
ax[0].set_ylabel("Error")
ax[0].set_title("Error")

ax[1].plot(n_meas_list, relative_error_list)
ax[1].scatter(n_meas_list, relative_error_list)
ax[1].set_yscale("log")
ax[1].set_xlabel("Number of measurements")
ax[1].set_ylabel("Relative error")
ax[1].set_title("Relative error")

# Save the plot
plt.savefig(save_folder+"error_convergence.png")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x, u_values, label = "True solution", color = "Red")
for i, pred in enumerate(pred_list):
    ax[0].plot(x, pred, label = "{}".format(n_meas_list[i]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel(r"$u(x)$")
    ax[0].set_title("Numerical solution")

    ax[1].plot(x, pred - u_values, label = "{}".format(n_meas_list[i]))
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$u^\dagger(x) - u(x)$")
    ax[1].set_title("Pointwise error")
ax[0].legend()
ax[1].legend()

plt.savefig(save_folder+"numerical_solution.png")

