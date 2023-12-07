
import os
from pdb import set_trace as bp

# Set JAX to use CPU
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


import jax.numpy as jnp
from jax import random
from jax import scipy 
import jax
print("Default JAX device:", jax.devices()[0])

import matplotlib.pyplot as plt
from utils_rough_pde import *

from jax.config import config
config.update("jax_enable_x64", True)


from matplotlib.animation import FuncAnimation

import jax 
devices = jax.devices()

# Print the list of devices
print("Available devices:", devices)

jax.config.update('jax_platform_name', 'cpu')


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


# First the mandatory arguments
parser.add_argument('--n_meas_max', type=int, help='Maximum number of measurements')
parser.add_argument('--s', type=float, help='Regularity of the solution')
parser.add_argument('--save_folder', type=str, help='Folder where to save the results')

#Now the optional arguments
parser.add_argument('--kernel_name', type=str, default="matern", help='Kernel name')
parser.add_argument('--measurement_type', type=str, default="indicator", help='Measurement type')
parser.add_argument('--n_meas_min', type=int, default=10, help='Minimum number of measurements')
parser.add_argument('--nugget_interior', type=float, default=1e-7, help='Regularization parameter for the interior of the domain')
parser.add_argument('--nugget_boundary', type=float, default=1e-12, help='Regularization parameter for the boundary of the domain')
# add an argument for the seed
parser.add_argument('--seed', type=int, default=54, help='Seed for the random number generator')
# add an argument for the length scale
parser.add_argument('--length_scale', type=float, default=1.0, help='Length scale of the kernel')
# add an argument for the order of the quadrature rule
parser.add_argument('--n_order', type=int, default=50, help='Order of the quadrature rule')
# add an argument for the number of coefficients
parser.add_argument('--n_coef', type=int, default=1000, help='Number of coefficients')
parser.add_argument('--n_evaluations', type=int, default=2000, help='Number of evaluations for trapzoidal rule')
parser.add_argument('--no_max_min', default = False, action=argparse.BooleanOptionalAction, help='Whether to use max min ordering or not')

# Add an option to create an animation 
parser.add_argument('--animation', default = False, action=argparse.BooleanOptionalAction, help='Whether to create an animation of the max min ordering or not')

args = parser.parse_args()

kernel_name = args.kernel_name
measurement_type = args.measurement_type
n_meas_min = args.n_meas_min
n_meas_max = args.n_meas_max
s = args.s
save_folder = args.save_folder + "/"
nugget_interior = args.nugget_interior
nugget_boundary = args.nugget_boundary
seed = args.seed
length_scale = args.length_scale
n_order = args.n_order
n_coef = args.n_coef
n_evaluations = args.n_evaluations
no_max_min = args.no_max_min
create_animation = args.animation








############################################



if kernel_name == "se":
    print("Using SE kernel")
    from utilities_kernel_se import *
elif kernel_name == "matern":
    print("Using Matern kernel")
    from utilities_kernel_matern import *
else:
    raise ValueError("Kernel name not recognized")




# Create the folder for saving the results
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# draw random values for the coefficients
key = random.PRNGKey(seed)



alpha = 2*s+1 + 0.25
print(s, alpha)
decay_u = alpha/2
L = 1 # Lenght of the domain
coef_u =  jnp.ones(shape = (1, ))/(jnp.arange(1, n_coef+1)**(decay_u)) 

coef_u = random.normal(key, shape=(n_coef,))/(jnp.arange(1, n_coef+1)**(decay_u))
coef_f = coef_u*jnp.arange(1, n_coef+1)**(2)*jnp.pi**2*L**2


x = jnp.linspace(0, L, n_evaluations)
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



# Boundary of the domain
lower, upper = 0.0, 1.0

boundary = jnp.array([[lower, upper]]).T

print("Nugget interior: ", nugget_interior)
print("Nugget boundary: ", nugget_boundary)


error_list = []
relative_error_list = []

pred_list = []

from scipy.special import roots_legendre
x_q, w_q = roots_legendre(n_order)

# These are to compute the error in the L^2 and H^1 norm (we can afford to go much higher as we only evaluate two functions)
# The order needs to be very high to ensure the accuracy of the computation of the error
x_error, w_error = roots_legendre(3000)
x_error, w_error =  root_interval(x_error, w_error, [0,1])
# The true solution evaluated at the quadrature points (only required for computing the error)
u_error = evaluate_function(x_error, coef_u, L=L)
print(u_error.shape)


error_list = []
relative_error_list = []

error_list_h = []
relative_error_list_h = []

pred_list = []


epsilon_values =  jnp.array([1/(n_meas_max*2)])
# Construct the measurements
loc_values = jnp.linspace(lower + epsilon_values[0], upper - epsilon_values[0],  int(L/(2*epsilon_values[0])))

N_test_functions = loc_values.shape[0]
print("Total number of test functions: ", N_test_functions)


    
# Compute the RHS of the linear system
f_pointwise = evaluate_function(loc_values, coef_f, L)

# Create max min ordering
if no_max_min:
    max_min_order = jnp.arange(loc_values.shape[0])
else:
    print("Creating max min ordering")
    loc_values_boundary = jnp.hstack([jnp.array([lower, upper]), loc_values])
    max_min_order = build_max_min_ordering(loc_values_boundary[:, None],[0,1])
    max_min_order = max_min_order[2:] -2




# Reorganize the measurements according to the max min ordering
f_pointwise = f_pointwise[max_min_order]


# Create an animation of the ordering

if create_animation:
    print("Creating animation of the ordering")
    # Initialize the plot
    fig, ax = plt.subplots(figsize = (12,6))
    def update(i):
        ax.clear()
        ax.scatter(loc_values[max_min_order][:i+1], f_pointwise[:i+1], s = 3)
        ax.set_title(f"Max-min ordering at step {i+1}")
        # Set the limits of the plot
        ax.set_xlim([0,1])
        ax.set_ylim([jnp.min(f_pointwise), jnp.max(f_pointwise)])
    animation = FuncAnimation(fig, update, frames=len(max_min_order), interval=10, repeat=False)
    print("Saving animation")
    # Save the animation
    animation.save(save_folder+"max_min_ordering.mp4", writer='ffmpeg', fps=5, dpi = 150)


# Compute the kernel matrix
print("Constructing the kernel matrix")
theta_11 = vmap_kernel(boundary, boundary, length_scale)
theta_12 = vmap_kernel_laplacian_y(boundary, loc_values[max_min_order], length_scale)
theta_22 = compute_K_double_laplacian_pairwise(loc_values[max_min_order], loc_values[max_min_order], length_scale)
theta = jnp.block([[theta_11, theta_12], [theta_12.T, theta_22]])

theta_evaluate_pointwise = jnp.block([vmap_kernel(x, boundary, length_scale), vmap_kernel_laplacian_y(x, loc_values[max_min_order], length_scale)])
theta_error = jnp.block([vmap_kernel(x_error, boundary, length_scale), vmap_kernel_laplacian_y(x_error, loc_values[max_min_order], length_scale)])


increment = 50
n_meas_list = jnp.arange(n_meas_min, n_meas_max + increment, increment, dtype=int)
n_meas_list = n_meas_list.at[-1].set(n_meas_max)




pred_list = []
meas_list = []
for i in n_meas_list:
    f_temp = f_pointwise[:i]

    # Construct the RHS of the linear system
    Y = jnp.block([jnp.zeros(shape = 2), f_temp])
    print("Number of measurements: ", Y.shape[0]-1)

    # Construct the nugget
    nugget = jnp.block([nugget_boundary*jnp.ones(shape = 2), nugget_interior*jnp.ones(shape = f_temp.shape[0])])

    # Solve the linear system
    print("Solving the linear system")
    # Select the submatrix of theta corresponding to the current measurements
    theta_temp = theta[:i+2, :i+2]

    c = scipy.linalg.solve(theta_temp + nugget*jnp.eye(theta_temp.shape[0]), Y, assume_a='pos')

    # Compute the numerical solution
    pred = theta_evaluate_pointwise[:, :i+2]@c
    pred_list.append(pred)

    # Compute the measurements of the numerical solution
    theta_eval = theta[:, :i+2]
    temp = theta_eval@c
    meas_list.append(temp)

    # Compute the numerical solution at the quadrature points (for computing the error)
    pred_error = theta_error[:, :i+2]@c

    # Compute the error between the true solution and the numerical solution
    loss, relative_loss = compute_error(pred_error, u_error, w_error)
    print("L^2 Error, Relative  L^2 error :", loss, relative_loss)

     # Compute the L^2 error using the second method (for sanity check)
    loss_2, relative_loss_2 = compute_error_h(pred_error, u_error,x_error, w_error, 0.0, L = 1.0, n_coef = n_coef)
    print("L^2 Error, Relative  L^2 error (second method):", loss_2, relative_loss_2)

    # Also compute the error in the H^1 norm
    loss_h, relative_loss_h = compute_error_h(pred_error, u_error,x_error, w_error, 1.0, L = 1.0)
    print("H^1 Error, Relative H^1 error :", loss_h, relative_loss_h)

    # Append the errors to the list
    error_list.append(loss)
    relative_error_list.append(relative_loss)
    error_list_h.append(loss_h)
    relative_error_list_h.append(relative_loss_h)

pred_list = jnp.array(pred_list)
meas_list = jnp.array(meas_list)

error_list = jnp.array(error_list)
relative_error_list = jnp.array(relative_error_list)
error_list_h = jnp.array(error_list_h)
relative_error_list_h = jnp.array(relative_error_list_h)

pred_list = jnp.array(pred_list)


print("Best error: ", jnp.min(error_list))
print("Best relative error: ", jnp.min(relative_error_list))

# Estimate the convergence rate by fitting a line to the log-log plot of the error
log_n_meas = jnp.log(n_meas_list)
log_error = jnp.log(error_list)

a = jnp.hstack([log_n_meas.reshape(-1, 1), jnp.ones_like(log_n_meas.reshape(-1, 1))])
b = log_error
r, C = jnp.linalg.lstsq(a, b)[0]
r, C = jnp.round(-r.item(),3), jnp.round(jnp.exp(C).item(), 3)

# Do the same for the H^1 error
log_error_h = jnp.log(error_list_h)

a = jnp.hstack([log_n_meas.reshape(-1, 1), jnp.ones_like(log_n_meas.reshape(-1, 1))])
b = log_error_h
r_h, C_h = jnp.linalg.lstsq(a, b)[0]
r_h, C_h = jnp.round(-r_h.item(),3), jnp.round(jnp.exp(C_h).item(),3)



print("L^2 Convergence rate: ", jnp.round(r,3)) 
print("H^1 Convergence rate: ", jnp.round(r_h,3))

# In a file, save the error and the relative error\
with open(save_folder+"error.txt", "w") as f:
    f.write("Best error: {}\n".format(jnp.min(error_list)))
    f.write("Best relative error: {}\n".format(jnp.min(relative_error_list)))

    f.write("Number of measurements: {}\n".format(n_meas_list[jnp.argmin(error_list)]))
    
    # Save the arguments of the parser
    f.write("Arguments of the parser:\n")
    for arg in vars(args):
        f.write("{}: {}\n".format(arg, getattr(args, arg)))
        


# Plot both the error and the relative error for the L^2 norm and the H^1 norm
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax[0,0].plot(n_meas_list, error_list, label = r"$r$ = {}, C = {}".format(r, C))
ax[0,0].scatter(n_meas_list, error_list)
ax[0,0].set_yscale("log")
ax[0,0].set_xlabel("Number of measurements")
ax[0,0].set_ylabel(r"$||u^\dagger - u* ||_{L^2}$")
ax[0,0].set_title(r"$L^2$ Error")
ax[0,0].legend()

ax[0,1].plot(n_meas_list, relative_error_list,  label = r"$r$ = {}, C = {}".format(r, C))
ax[0,1].scatter(n_meas_list, relative_error_list)
ax[0,1].set_yscale("log")
ax[0,1].set_xlabel("Number of measurements")
ax[0,1].set_ylabel(r"$\frac{||u^\dagger - u* ||_{L^2}}{|| u* ||_{L^2}}$")
ax[0,1].set_title(r"Relative $L^2$ Error")
ax[0,1].legend()

ax[1,0].plot(n_meas_list, error_list_h, label = r"$r$ = {}, C = {}".format(r_h, C))
ax[1,0].scatter(n_meas_list, error_list_h)
ax[1,0].set_yscale("log")
ax[1,0].set_xlabel("Number of measurements")
ax[1,0].set_ylabel(r"$||u^\dagger - u* ||_{H^1}$")
ax[1,0].set_title(r"$H^1$ Error")
ax[1,0].legend()

ax[1,1].plot(n_meas_list, relative_error_list_h, label = r"$r$ = {}, C = {}".format(r_h, C_h))
ax[1,1].scatter(n_meas_list, relative_error_list_h)
ax[1,1].set_yscale("log")
ax[1,1].set_xlabel("Number of measurements")
ax[1,1].set_ylabel(r"$\frac{||u^\dagger - u* ||_{H^1}}{|| u* ||_{H^1}}$")
ax[1,1].set_title(r"Relative $H^1$ Error")
ax[1,1].legend()




# Save the plot
plt.savefig(save_folder+"error_convergence.png")


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x, u_values, label = "True solution", color = "Red")
for i, pred in enumerate(pred_list):
    ax[0].plot(x, pred, label = "{}".format(n_meas_list[i]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel(r"$u^*(x)$")
    ax[0].set_title("Numerical solution")

    ax[1].plot(x, pred - u_values, label = "{}".format(n_meas_list[i]))
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$u^\dagger(x) - u^*(x)$")
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
    ax[0].set_ylabel(r"$u^*(x)$")
    ax[0].set_title("Numerical solution")

    ax[1].plot(x, pred - u_values, label = "{}".format(n_meas_list[-1]))
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$u^\dagger(x) - u^*(x)$")
    ax[1].set_title("Pointwise error")


ax[0].legend()
ax[1].legend()

plt.savefig(save_folder+"final_numerical_solution.png")


if create_animation:
    print("Creating animation of the results")

    fig, ax = plt.subplots(3,1, figsize = (12,10))

    def update(i):
            idx = n_meas_list[i]
            ax[0].clear()
            ax[0].scatter(loc_values[max_min_order], f_pointwise, label = 'true', s = 3, color = "blue")
            ax[0].scatter(loc_values[max_min_order], meas_list[i, 2:], label = 'prediction', s = 3, color = "green")
            ax[0].scatter(loc_values[max_min_order[:idx]], meas_list[i, 2:idx+2], label = 'training points', color = 'red', s = 3)
            ax[0].legend()
            ax[0].set_title(f"Prediction of the measurements with {n_meas_list[i]} data points")

            ax[1].clear()
            ax[1].plot(x, pred_list[i], label = 'pred')
            ax[1].plot(x, u_values, label = 'true')
            ax[1].set_title("Prediction with  {} pointwise evaluation of the RHS".format(n_meas_list[i]))
            ax[1].legend()

            ax[2].clear()
            ax[2].plot(x, pred_list[i] - u_values)
            ax[2].set_title("Pointwise error with  {} pointwise evaluation of the RHS".format(n_meas_list[i]))
    # Create an animation of the the rhs, the solution and the pointwise error

    animation = FuncAnimation(fig, update, frames=len(n_meas_list), interval=500, repeat=False)
    animation.save(save_folder+"results.mp4", writer='ffmpeg', fps=2, dpi = 150)


