from SolvingRoughPDEs.utilities.kernels import *
from SolvingRoughPDEs.gp.NonlinearElliptic2D_Iteratively import *
from SolvingRoughPDEs.utilities.domain import *
import jax.numpy as jnp
import numpy as np
from jax.config import config
from jax import vmap
import munch
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

cfg = munch.munchify({
    'N' : 20,
    'M': 224,
    'M_Omega': 24,
    'alpha': 1,
    'beta': 1,
    'mT' : 100,
    'm': 5,
    's': 1,
    'gamma': 1e-12,
    'lenghscale' : 0.09,
    'nugget': 1e-12,
    'epoch' : 200,
    'tol': 1e-5,
    'deg': 90,
    'load_current': False,
    'load_pre': True,
})

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 26

plt.rcParams['axes.labelpad'] = 10
#plt.rcParams["figure.figsize"] = (8, 6)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=21)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

domain = Square(0, 1, 0, 1)
kernel = Gaussian_Kernel(cfg.lenghscale)

eq = NonlinearElliptic2D(kernel, domain, cfg.alpha, cfg.beta, cfg.m, cfg.N, cfg.s, cfg.gamma, cfg.deg)

u0_weights = jnp.array([])
u0_hs = jnp.array([])
u0_z = jnp.array([])

if cfg.load_pre:
    prefix = "./results/NonlinearElliptic2D/"
    u_weights_filename = prefix + "u_weights_m_" + str(cfg.m-1) + ".txt"
    u_hs_filename = prefix + "u_hs_m_" + str(cfg.m-1) + ".txt"
    xi_filename = prefix + "xi.txt"
    u_z_filename = prefix + "u_z_m_" + str(cfg.m-1) + ".txt"
    samples_file = prefix + "samples.txt"

    u0_weights = np.loadtxt(u_weights_filename)
    u0_hs = np.loadtxt(u_hs_filename)
    xi = np.loadtxt(xi_filename)
    u_z = np.loadtxt(u_z_filename)
    samples = np.loadtxt(samples_file)

    eq.set_data(samples, cfg.M, cfg.M_Omega, xi)
elif cfg.load_current:
    prefix = "./results/NonlinearElliptic2D/"
    u_weights_filename = prefix + "u_weights_m_" + str(cfg.m) + ".txt"
    u_hs_filename = prefix + "u_hs_m_" + str(cfg.m) + ".txt"
    xi_filename = prefix + "xi.txt"
    u_z_filename = prefix + "u_z_m_" + str(cfg.m) + ".txt"
    samples_file = prefix + "samples.txt"

    u0_weights = np.loadtxt(u_weights_filename)
    u0_hs = np.loadtxt(u_hs_filename)
    xi = np.loadtxt(xi_filename)
    u_z = np.loadtxt(u_z_filename)
    samples = np.loadtxt(samples_file)

    eq.set_data(samples, cfg.M, cfg.M_Omega, xi)

else:
    eq.sample_xi(cfg.mT)
    eq.sampling(cfg)

eq.train(cfg, u0_weights, u0_hs, u0_z)

prefix = "./results/NonlinearElliptic2D/"
u_weights_filename = prefix + "u_weights_m_" + str(cfg.m) + ".txt"
u_hs_filename = prefix + "u_hs_m_" + str(cfg.m) + ".txt"
xi_filename = prefix + "xi.txt"
samples_file = prefix + "samples.txt"
u_z_filename = prefix + "u_z_m_" + str(cfg.m) + ".txt"

np.savetxt(u_weights_filename, eq.weights, delimiter='\t', fmt='%.64f')
np.savetxt(u_hs_filename, eq.hs, delimiter='\t', fmt='%.64f')
np.savetxt(xi_filename, eq.xi, delimiter='\t', fmt='%.64f')
np.savetxt(samples_file, eq.samples, delimiter='\t', fmt='%.64f')
np.savetxt(u_z_filename, eq.z, delimiter='\t', fmt='%.64f')


N_pts = 60
xx= jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(0, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy, indexing='ij')
X_test = jnp.concatenate((XX.reshape(-1,1), YY.reshape(-1,1)), axis=1)

test_truth = vmap(eq.u)(X_test[:, 0], X_test[:, 1])

print("start to resample")
u_r = eq.fit(X_test)
print("finish resampling")
all_errors = np.abs(test_truth.flatten() - u_r.flatten())
print("The final error is {}".format(np.max(all_errors)))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_surface(XX, YY, test_truth.reshape(XX.shape), rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('$x_1$', labelpad=15)
ax.set_ylabel('$x_2$', labelpad=15)
ax.set_zlabel(r'$u^*$', labelpad=20)
ax.tick_params(axis='z', which='major', pad=10)
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_surface(XX, YY, u_r.reshape(XX.shape), rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('$x_1$', labelpad=15)
ax.set_ylabel('$x_2$', labelpad=15)
ax.set_zlabel(r'$u^\dagger$', labelpad=20)
ax.tick_params(axis='z', which='major', pad=10)
fig.tight_layout()

fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
fig = plt.figure()
ax = fig.add_subplot(111)
err_contourf = ax.contourf(XX, YY, all_errors.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Contour of errors')
fig.colorbar(err_contourf, format=fmt)

plt.show()

















