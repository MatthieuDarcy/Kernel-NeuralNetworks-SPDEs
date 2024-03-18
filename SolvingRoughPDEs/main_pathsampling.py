from SolvingRoughPDEs.FD.PathSampling import *
import jax.numpy as jnp
import numpy as np
from jax.config import config
from jax import vmap
import munch
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

cfg = munch.munchify({
    'T' : 1000,
    'a' : 0,
    'b': 100,
    'xa': -1,
    'xb': 1,
    'Deltau': 1e-2,
    'Deltat': 1e-4,
    'sigma': 1,
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

eq = PathSampling(cfg.a, cfg.b, cfg.T, cfg.Deltau, cfg.Deltat, cfg.sigma, cfg.xa, cfg.xb)
sols, itv, start_index = eq.solve()

tt = jnp.linspace(0, cfg.T, eq.nT + 1)
xx = jnp.linspace(cfg.a, cfg.b, eq.nN + 1)


fig, ax = plt.subplots()
def animate(i):
    index = (start_index + i) % itv
    t = tt[index]
    sol = sols[index, :]
    ax.clear()
    ax.plot(xx, sol, 'b', linewidth=3, markersize=12, label=r"$x$")
    plt.xlabel(r'$x$')
    plt.title(f"Time {t}")
    plt.legend()

ani = FuncAnimation(fig, animate, frames=100, interval=itv, repeat=True)
ani.save('./mygif.gif', fps=10)
plt.show()


































