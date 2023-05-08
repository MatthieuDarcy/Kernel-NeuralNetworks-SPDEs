from SolvingRoughPDEs.utilities.kernels import *
from SolvingRoughPDEs.gp.NonlinearElliptic2D import *
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
    'N' : 8,
    'M': 224,
    'M_Omega': 100,
    'alpha': 1,
    'beta': 1,
    'm': 5,
    's': 1,
    'gamma': 1e-5,
    'lenghscale' : 0.1,
    'nugget': 1e-12,
    'epoch' : 200,
    'tol': 1e-5,
    'deg': 80,
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
eq.sampling(cfg)
eq.train(cfg)

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
































