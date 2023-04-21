from SolvingRoughPDEs.utilities.kernels import *
from SolvingRoughPDEs.gp.NonlinearElliptic1D import *
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
    'M': 402,
    'M_Omega': 400,
    'alpha': 1,
    'm': 20,
    's': 2,
    'gamma': 1e-20,
    'lenghscale' : 0.1,
    'nugget': 1e-12,
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

domain = Interval(0, 1)
kernel = Gaussian_Kernel_1D(cfg.lenghscale)

eq = NonlinearElliptic1D(kernel, domain, cfg.alpha, cfg.m, cfg.N, cfg.s, cfg.gamma)
eq.sampling(cfg)
eq.train(cfg)

N_pts = 60
xx = jnp.linspace(0, 1, N_pts)

test_truth = vmap(eq.u)(xx)

print("start to resample")
u_r = eq.fit(xx)
print("finish resampling")
all_errors = np.abs(test_truth - u_r)
print("The final error is {}".format(np.max(all_errors)))

fig = plt.figure()
plt.plot(xx, u_r, 'b', label=r'$u^\dagger$')
plt.plot(xx, test_truth, 'r-.', label=r'$u^*$')
plt.legend()
plt.show()
































