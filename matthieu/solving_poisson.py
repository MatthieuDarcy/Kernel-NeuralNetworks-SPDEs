
import jax.numpy as jnp
from jax import random
from jax import scipy 

import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)





import jax 
devices = jax.devices()

# Print the list of devices
print("Available devices:", devices)

#jax.config.update('jax_platform_name', 'cpu')


############################################

## Set up argparse
import argparse

parser = argparse.ArgumentParser(description='Solve the Poisson equation with increasing number of measurements')

# The parser should be able to read the following arguments:
# - kernel name (default: matern)
# - measurement type (default: indicator)
# - minimum number of measurements (default: 10)
# - maximum number of measurements (required)
# - regularity (required)
# - save folder (required)

parser.add_argument('--kernel_name', type=str, default="matern", help='Kernel name')
parser.add_argument('--measurement_type', type=str, default="indicator", help='Measurement type')
parser.add_argument('--n_meas_min', type=int, default=10, help='Minimum number of measurements')
parser.add_argument('--n_meas_max', type=int, help='Maximum number of measurements')
parser.add_argument('--s', type=float, help='Regularity of the solution')
parser.add_argument('--save_folder', type=str, help='Folder where to save the results')
parser.add_argument('--nugget_interior', type=float, default=1e-7, help='Regularization parameter for the interior of the domain')
parser.add_argument('--nugget_boundary', type=float, default=1e-12, help='Regularization parameter for the boundary of the domain')

args = parser.parse_args()

kernel_name = args.kernel_name
measurement_type = args.measurement_type
n_meas_min = args.n_meas_min
n_meas_max = args.n_meas_max
s = args.s
save_folder = args.save_folder + "/"
nugget_interior = args.nugget_interior
nugget_boundary = args.nugget_boundary






############################################



if kernel_name == "se":
    from utils_rough_pde_se import *
elif kernel_name == "matern":
    from utils_rough_pde import *
else:
    raise ValueError("Kernel name not recognized")



seed = 54


import os

# Create the folder for saving the results
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# draw random values for the coefficients
key = random.PRNGKey(seed)

n_coef = 500


alpha = 2*s+1 + 0.25
print(s, alpha)
decay_u = alpha/2
L = 1 # Lenght of the domain
coef_u =  jnp.ones(shape = (1, ))/(jnp.arange(1, n_coef+1)**(decay_u)) 

coef_u = random.normal(key, shape=(n_coef,))/(jnp.arange(1, n_coef+1)**(decay_u))
coef_f = coef_u*jnp.arange(1, n_coef+1)**(2)*jnp.pi**2*L**2

x = jnp.linspace(0, L, 999)
u_values = evaluate_function(x, coef_u, L=L)
f_values = evaluate_function(x, coef_f, L=L)

# Plot both the coeffients of u and f in 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(coef_u**2)
ax1.set_yscale('log')
ax1.set_title('Squared coefficients of u')
ax2.plot(coef_f**2)
ax2.set_yscale('log')
ax2.set_title('Squared coefficients of f')

plt.savefig(save_folder+"coefficients.png")


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

n_meas_list = jnp.arange(n_meas_min, n_meas_max + 10, 10, dtype=int)
# Boundary of the domain
lower, upper = 0.0, 1.0
length_scale = 0.1
boundary = jnp.array([[lower, upper]]).T

print("Nugget interior: ", nugget_interior)
print("Nugget boundary: ", nugget_boundary)


error_list = []
relative_error_list = []

pred_list = []

from scipy.special import roots_legendre
n_order = 50
x_q, w_q = roots_legendre(n_order)

error_list = []
relative_error_list = []

pred_list = []
for n_meas in n_meas_list:

    # Construct the measurements
    epsilon_values =  jnp.array([1/(n_meas*2)])
    loc_values = jnp.linspace(lower + epsilon_values[0], upper - epsilon_values[0],  int(L/(2*epsilon_values[0])))
    support = jnp.array([loc_values - epsilon_values[0], loc_values + epsilon_values[0]]).T
    vol = support[:,1] - support[:,0]
    N_test_functions = loc_values.shape[0]

    print("Number of test functions: ", N_test_functions)


    root_psi, w_psi = vmap_root_interval(x_q, w_q, support)

    # Construct the mmatrix of weighted measurement functions 
    if measurement_type == "indicator":
        psi_matrix = indicator_vector(root_psi, epsilon_values, loc_values)
    elif measurement_type == "bump":
        psi_matrix = bump_vector(root_psi, epsilon_values, loc_values)
    else:
        # Raise an error
        print("Measurement type not recognized")
        break

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


print("Best error: ", jnp.min(error_list))
print("Best relative error: ", jnp.min(relative_error_list))


# Plot both the error and the relative error
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(n_meas_list, error_list)
ax[0].scatter(n_meas_list, error_list)
ax[0].set_yscale("log")
ax[0].set_xlabel("Number of measurements")
ax[0].set_ylabel(r"$||u^\dagger - u* ||_{L^2}$")
ax[0].set_title(r"$L^2$ Error")

ax[1].plot(n_meas_list, relative_error_list)
ax[1].scatter(n_meas_list, relative_error_list)
ax[1].set_yscale("log")
ax[1].set_xlabel("Number of measurements")
ax[1].set_ylabel(r"$\frac{||u^\dagger - u* ||_{L^2}}{|| u* ||_{L^2}}$")
ax[1].set_title(r"Relative $L^2$ Error")

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

plt.savefig(save_folder+"all_numerical_solution.png")

#For the final preduction, plot the true solution and the numerical solution and the error
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x, u_values, label = "True solution", color = "Red")
for i, pred in enumerate(pred_list[-1:]):
    ax[0].plot(x, pred, label = "{}".format(n_meas_list[-1]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel(r"$u(x)$")
    ax[0].set_title("Numerical solution")

    ax[1].plot(x, pred - u_values, label = "{}".format(n_meas_list[-1]))
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$u^\dagger(x) - u(x)$")
    ax[1].set_title("Pointwise error")


ax[0].legend()
ax[1].legend()

plt.savefig(save_folder+"final_numerical_solution.png")


