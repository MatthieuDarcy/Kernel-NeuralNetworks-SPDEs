import jax.numpy as jnp
from jax import grad

class Matern_Kernel_1D(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def kappa(self, x, y):
        d = jnp.sqrt((x - y) ** 2)
        # return (1 + jnp.sqrt(3) * d / self.sigma) * jnp.exp(-jnp.sqrt(3) * d / self.sigma)
        return (1 + jnp.sqrt(5) * d / self.sigma + 5 * d ** 2 / (3 * self.sigma**2)) * jnp.exp(-jnp.sqrt(5) * d / self.sigma)

    def Delta_x_kappa(self, x, y):
        val = grad(grad(self.kappa, 0), 0)(x, y)
        return val

    def Delta_y_kappa(self, x, y):
        val = grad(grad(self.kappa, 1), 1)(x, y)
        return val

    def Delta_x_Delta_y_kappa(self, x, y):
        val = grad(grad(self.Delta_x_kappa, 1), 1)(x, y)
        return val

class Gaussian_Kernel_1D(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def softplus(self, x):
        return jnp.log(1 + jnp.exp(x))

    def softplus_inv(self, x):
        return jnp.log(jnp.exp(x) - 1)

    def kappa(self, x, y):
        return jnp.exp(-(1 / (2 * self.sigma ** 2)) * ((x - y) ** 2))

    def Delta_x_kappa(self, x, y):
        val = grad(grad(self.kappa, 0), 0)(x, y)
        return val

    def Delta_y_kappa(self, x, y):
        val = grad(grad(self.kappa, 1), 1)(x, y)
        return val

    def Delta_x_Delta_y_kappa(self, x, y):
        val = grad(grad(self.Delta_x_kappa, 1), 1)(x, y)
        return val



class Gaussian_Kernel(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        return jnp.exp(-(1 / (2 * self.sigma ** 2)) * ((x1 - y1) ** 2 + (x2 - y2) ** 2))

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

class Anisotropic_Gaussian_kernel(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        scale_t = self.sigma[0]
        scale_x = self.sigma[1]
        r = ((x1 - y1) / scale_t) ** 2 + ((x2 - y2) / scale_x) ** 2
        return jnp.exp(-r)

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val