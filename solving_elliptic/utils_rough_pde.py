
import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import hessian
from jax import random
import math



from jax.config import config
config.update("jax_enable_x64", True)


########################################################################################################################
# Test function utilities

# Utilities for bump functions
def bump(x, epsilon, loc):
    condition = (loc - epsilon < x) & (x < loc + epsilon)
    return jnp.where(
        condition,
        jnp.exp(-1 / (1 - (x - loc) ** 2 / epsilon ** 2))/epsilon/0.4439938,
        0.0
    )

# First vectorization over 'loc'
vectorized_over_loc = vmap(bump, in_axes=(None, None, 0))
# Second vectorization over 'epsilon'
vectorized_over_epsilon = vmap(vectorized_over_loc, in_axes=(None, 0, None))
# Final vectorization over 'x_values'
vmap_bump = vmap(vectorized_over_epsilon, in_axes=(0, None, None))

bump_vector = vmap(bump, in_axes=(0, None, 0))

# Utilities for indicator functions
def indicator(x, epsilon, loc):
    condition = (loc - epsilon < x) & (x < loc + epsilon)
    return jnp.where(
        condition,
        1/(2*epsilon),
        0.0
    )

# First vectorization over 'loc'
indicator_vectorized_over_loc = vmap(indicator, in_axes=(None, None, 0))
# Second vectorization over 'epsilon'
indicator_vectorized_over_epsilon = vmap(indicator_vectorized_over_loc, in_axes=(None, 0, None))
# Final vectorization over 'x_values'
vmap_indicator = vmap(indicator_vectorized_over_epsilon, in_axes=(0, None, None))

indicator_vector = vmap(indicator, in_axes=(0, None, 0))

# Utilities for the tent functions
def tent_function(x, epsilon, center):
    return jnp.maximum(0, 1 - jnp.abs(x - center) / epsilon)/epsilon

def gradient_tent_function(x, epsilon, center):
    return jnp.where(jnp.abs(x - center) < epsilon, jnp.sign(center -x) , 0)/epsilon**2

vmap_tent = vmap(tent_function, in_axes=(None, None, 0))
vmap_tent_vector = vmap(tent_function, in_axes=(0, None, 0))
vmap_tent_evaluate= vmap(vmap_tent, in_axes=(0, None, None))

def evaluate_tent(c, x, epsilon, center):
    return vmap_tent_evaluate(x, epsilon, center)@c

def compute_l2_ip(epsilon, center_1, center_2):
    condition1 = center_1 == center_2
    condition2 = jnp.allclose(jnp.abs(center_1 - center_2), epsilon)
    return jnp.where(condition1, 2 / (epsilon*3),
                     jnp.where(condition2, 1 / (epsilon*6), 0))

def compute_h1_ip(epsilon, center_1, center_2):
    condition1 = center_1 == center_2
    condition2 = jnp.allclose(jnp.abs(center_1 - center_2), epsilon)
    return jnp.where(condition1, 2 / epsilon,
                     jnp.where(condition2, -1 / epsilon, 0))


def create_banded_matrix(n, a, b):
    # Diagonal filled with `a`
    diagonal = jnp.full(n, a)
    # Immediate off-diagonals filled with `b`
    off_diagonal = jnp.full(n - 1, b)
    # Create the banded matrix
    matrix = jnp.diag(diagonal) + jnp.diag(off_diagonal, k=1) + jnp.diag(off_diagonal, k=-1)
    return matrix



vmap_compute_l2_ip = vmap(vmap(compute_l2_ip, in_axes=(None, 0, None)), in_axes=(None, None, 0))
vmap_compute_energy_ip = vmap(vmap(compute_h1_ip, in_axes=(None, 0, None)), in_axes=(None, None, 0))


###########################################################################################################

# Utilities for the quadrature

def root_interval(x_q, w_q, interval):
    # Defines the roots of the interval [a,b]
    a= interval[0]
    b= interval[1]
    return (b-a)/2*x_q + (b+a)/2, (b - a) / 2 * w_q

vmap_root_interval = vmap(root_interval, in_axes=(None,None,  0))


########################################################################################################################

# Utilities for the data
def evaluate_function(x, coef, L):
    values = 0
    for i, c in enumerate(coef):
        values +=c*jnp.sin((i+1)*jnp.pi*x/L)*jnp.sqrt(2/L)
    return values

@jit
def integrate_f_test_functions(f_values, psi):
    # Psi should be a matrix containing the values of the weighted test functions at the quadrature points
    return f_values.T@psi
vmap_integrate_f_test_functions = jit(vmap(integrate_f_test_functions, in_axes=(0,0)))

def evaluate_chi(x,k, L):
    return jnp.sin((k)*jnp.pi*x/L)*jnp.sqrt(2/L)

vmap_chi_quad = jit(vmap(vmap(evaluate_chi, in_axes=(0, None, None)), in_axes = (None, 0, None)))

def construct_f_meas(coef,psi_matrix, root_psi, L):
    n_coef = coef.shape[0]
    freq = jnp.arange(1, n_coef+1)

    chi_quad = vmap_chi_quad(root_psi,freq, L)
    B = jnp.sum(chi_quad*psi_matrix[None], axis = -1)
    f_m = jnp.sum(coef[:, None]*B, axis = 0)

    return f_m


########################################################################################################################
# Utilities for the error
# The following uses a Gauss quadrature rule to compute the integral instead of the trapezoidal rule (this is more accurate)
@jit
def compute_error(pred_q, true_q, w_q):
    loss = jnp.sqrt(jnp.sum((pred_q - true_q)**2*w_q))
    relative_loss = loss/jnp.sqrt(jnp.sum(true_q**2*w_q))
    return loss, relative_loss

def compute_projection_sin(x_q, w_q, f_values, n_coef, L = 1.0):
    sin_basis = jnp.sqrt(2/L)*jnp.array([jnp.sin(jnp.pi*k/L*x_q) for k in range(1, n_coef+1)])
    return jnp.sum(f_values[None]*sin_basis*w_q, axis = -1)

def compute_h_norm(coef, s, L = 1.0):
    eigenvalues = jnp.arange(1, coef.shape[0]+1)**2*jnp.pi**2/L**2
    return jnp.sqrt(jnp.sum(coef**2*eigenvalues**s))


def compute_error_h(pred_q, true_q,x_q, w_q, s, L = 1.0, n_coef = 1000):

    # Compute the preojection of the prediction on the sine basis
    coef_error = compute_projection_sin(x_q, w_q, pred_q - true_q, n_coef , L = 1.0)
    coef_true = compute_projection_sin(x_q, w_q, true_q, n_coef, L = 1.0)
    # Compute the error between the true coefficients and the predicted coefficients
    error = compute_h_norm(coef_error, s, L=L)
    norm_true = compute_h_norm(coef_true, s, L= L)

    return error, error/norm_true




########################################################################################################################
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def build_max_min_ordering(X, initial_points):
    dist_matrix = squareform(pdist(X))

    # We include a set of initial points (can be boundary points, chosen at random or whatever). These should be indices of the points in X
    idx_order = initial_points
    #idx_left = jnp.setdiff1d(jnp.arange(X.shape[0]), jnp.array(idx_order)).tolist()
    key = random.PRNGKey(23)



    for i in range(X.shape[0] - len(initial_points)):
        key, subkey = random.split(key)
        # Compute the current max min distance
        dist_temp = dist_matrix[idx_order, :]
        dist_temp = jnp.min(dist_temp, axis = 0)
        #print(dist_temp)

        # Find all the points that maximizes the min distance
        max_values = dist_temp.max()
        max_indices = jnp.where(dist_temp == max_values)[0]
        # Select one at random
        best_j = random.choice(subkey, max_indices).item()
        idx_order.append(best_j)

        

    return jnp.array(idx_order)


########################################################################################################################
# Class to compute measurements
from scipy.special import roots_legendre
from jax import scipy
class measurement_tool():
    def __init__(self, domain, N, n_order = 10):
        lower = domain[0]
        upper = domain[1]

        epsilon = (upper- lower)/N
        centers = jnp.linspace(lower + epsilon, upper -epsilon, N)

        self.N = N
        self.centers = centers
        self.epsilon = epsilon
        self.n_order = n_order  

        self.build_matrices()
        self.build_measurements_matrices()

        n_error = 100
        x_error, w_error = roots_legendre(n_error)
        x_error, w_error = root_interval(x_error, w_error, jnp.array([lower, upper]))
        self.x_error = x_error
        self.w_error = w_error

    def build_matrices(self):
        self.stiffess_matrix = create_banded_matrix(self.N, 2/(self.epsilon*3), 1/(self.epsilon*6))#vmap_compute_energy_ip(self.epsilon, self.centers, self.centers) 
                    
        self.L_2 =create_banded_matrix(self.N, 2/(self.epsilon), -1/(self.epsilon))  #vmap_compute_l2_ip(self.epsilon, self.centers, self.centers) 
        self.L_stiff = scipy.linalg.cho_factor(self.stiffess_matrix + 1e-10*jnp.eye(self.N))

    def build_measurements_matrices(self):
        support = jnp.array([self.centers - self.epsilon
                     , self.centers + self.epsilon]).T
        
        x_q, w_q = roots_legendre(self.n_order)
        self.x_interval, self.w_interval = vmap_root_interval(x_q, w_q, support)

        root_psi, w_psi = vmap_root_interval(x_q, w_q, support)
        psi_matrix = vmap_tent_vector(root_psi, self.epsilon, self.centers)

        self.psi_matrix = psi_matrix * w_psi
        self.root_psi = root_psi

    def project(self, f_quad):
        return vmap_integrate_f_test_functions(f_quad, self.psi_matrix)
    
    def evaluate_at_roots(self, function):
        return function(self.root_psi)
    
    def evaluate_for_error(self, function):
        return function(self.x_error)




if __name__=="__main__":
    # Define the parameters of the bump functions
    epsilon_values =  jnp.array([0.05])

    # Boundary of the domain
    lower, upper = -1.0, 1.0
    L = 2.0 # Lenght of the domain
    loc_values = jnp.linspace(lower + epsilon_values[0], upper - epsilon_values[0],  int(L/(2*epsilon_values[0])))
    support = jnp.array([loc_values - epsilon_values[0], loc_values + epsilon_values[0]]).T
    vol = support[:,1] - support[:,0]
    N_test_functions = loc_values.shape[0]

    # Define the parameters of the kernel
    length_scale = 0.5
    # Import the quadrature rules
    from scipy.special import roots_legendre
    n_order = 1000
    x_q, w_q = roots_legendre(n_order)

    root_psi, w_psi = vmap_root_interval(x_q, w_q, support)



    # Construct the matrix of psi functions
    psi_matrix = bump_vector(root_psi, epsilon_values, loc_values)
    psi_matrix = psi_matrix * w_psi

    theta = construct_theta_integral(psi_matrix, root_psi, length_scale)

    print(theta.shape)

    eigenvalues, eigenvectors = jnp.linalg.eigh(theta)
    print("The smallest eigenvalue is ", jnp.min(eigenvalues))

    # These values should be roughly equal
    print("These values (along the diagonal) should be roughly equal")
    # Pause here until user presses enter
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[i,i], double_neg_laplacian(loc_values[i], loc_values[i], length_scale))




    print("These values should also be roughly equal (kernel at different points)")
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[0,i], double_neg_laplacian(loc_values[0], loc_values[i], length_scale))

    print("Now we try with a Monte Carlo approximation")
    input("Press enter to view...")

    # Instead of using Gauss quadrature points, we will use random samples 
    from jax import random

    N_samples = 1000
    key = random.PRNGKey(0)
    x_q = random.uniform(key, (N_test_functions, N_samples))*2*epsilon_values[0] +  loc_values[:, None] - epsilon_values[0]
    root_psi = x_q

    # The weights are not defined by the quadrature rule anymore but instead by the volume of integration and the number of samples
    w_psi = jnp.ones((N_test_functions, N_samples))/N_samples*vol[:, None]


    psi_matrix = bump_vector(root_psi, epsilon_values, loc_values)
    psi_matrix = psi_matrix * w_psi
    theta = construct_theta_integral(psi_matrix, root_psi,  length_scale)


    eigenvalues, eigenvectors = jnp.linalg.eigh(theta)
    print("The smallest eigenvalue is ", jnp.min(eigenvalues))

    # These values should be roughly equal
    print("These values (along the diagonal) should be roughly equal")
    # Pause here until user presses enter
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[i,i], double_neg_laplacian(loc_values[i], loc_values[i], length_scale))

    print("These values should also be roughly equal (kernel at different points)")
    input("Press enter to view...")
    for i in range(theta.shape[0]):
        print(theta[0,i], double_neg_laplacian(loc_values[0], loc_values[i], length_scale))




    
    


    


    
    