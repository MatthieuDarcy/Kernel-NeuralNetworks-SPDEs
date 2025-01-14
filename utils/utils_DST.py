import jax.numpy as jnp
import jax.scipy as scipy
from jax import vmap, jit

################### 1D DST ####################
def discrete_sine_transform(y):
    """
    y is the vector of coefficents of size n. It assumes that it only contains the frequencies 1,2,3,...,n. (no zero frequency)
    returns evaluation of the sine series at the points 1,2,3,...,n (no zero point)
    """
    n = y.shape[0]
    y_extended = jnp.concatenate([jnp.zeros(1), y, jnp.zeros(1), -y[::-1]])
    y_fft = jnp.fft.fft(y_extended)/2
    return (-y_fft.imag)[1: n+1]*jnp.sqrt(2)

vmap_dst = jit(vmap(discrete_sine_transform, in_axes=(0,)))

def compute_sine_coefficients(y):
    """
    y is the vector of values of the sine series at the points 1,2,3,...,n (no zero point)
    returns the coefficients of the sine series of size n (no zero frequency)
    """
    n = y.shape[0]
    return discrete_sine_transform(y)/(n+1)

vmap_compute_sine_coef = jit(vmap(compute_sine_coefficients, in_axes=(0,)))

################### 2D DST ####################

# @jit
# def dst_2d(A):
#     # Receives a 2d array and returns the 2d discrete sine transform
#     return vmap_dst(vmap_dst(A).T).T

# vmap_dst_2d = jit(vmap(dst_2d, in_axes = 0))#jit(vmap(vmap(dst_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1))
# @jit
# def double_dst_2d(A):
#     # Transform along the last 2 axes
#     A = vmap(vmap(dst_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
#     # Transpose the axes
#     A = jnp.transpose(A, (2, 3, 0, 1))
#     # Transform along the last 2 axes
#     A = vmap(vmap(dst_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
#     # Transpose the axes (going back to the original order)
#     A = jnp.transpose(A, (2, 3, 0,1))
#     return A

# @jit
# def compute_sine_coef_2d(A):
#     # Receives a 2d array and returns the 2d discrete sine transform
#     n = A.shape[0]
#     return dst_2d(A)/((n+1)**2)

# vmap_compute_sine_coef_2d = jit(vmap(compute_sine_coef_2d, in_axes = 0))
# @jit
# def double_compute_sine_coef_2d(A):
#     # Transform along the last 2 axes
#     A = vmap(vmap(compute_sine_coef_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
#     # Transpose the axes
#     A = jnp.transpose(A, (2, 3, 0, 1))
#     # Transform along the last 2 axes
#     A = vmap(vmap(compute_sine_coef_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
#     # Transpose the axes (going back to the original order)
#     A = jnp.transpose(A, (2, 3, 0,1))
#     return A


# Update

@jit
def dst_2d(A):
    # Receives a 2d array and returns the 2d discrete sine transform
    return vmap_dst(vmap_dst(A).T).T
vmap_dst_2d = jit(vmap(dst_2d, in_axes = 0))#jit(vmap(vmap(dst_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1))

@jit
def double_dst_2d(A):
    # Transform along the last 2 axes
    A = vmap(vmap(dst_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
    # Transpose the axes
    A = jnp.transpose(A, (2, 3, 0, 1))
    # Transform along the last 2 axes
    A = vmap(vmap(dst_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
    # Transpose the axes (going back to the original order)
    A = jnp.transpose(A, (2, 3, 0,1))
    return A

@jit
def compute_sine_coef_2d(A):
    # Receives a 2d array and returns the 2d discrete sine transform
    n = A.shape[0]
    return dst_2d(A)/((n+1)**2)

vmap_compute_sine_coef_2d = jit(vmap(compute_sine_coef_2d, in_axes = 0))
@jit
def double_compute_sine_coef_2d(A):
    # Transform along the last 2 axes
    A = vmap(vmap(compute_sine_coef_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
    # Transpose the axes
    A = jnp.transpose(A, (2, 3, 0, 1))
    # Transform along the last 2 axes
    A = vmap(vmap(compute_sine_coef_2d, in_axes=0, out_axes=0), in_axes=1,out_axes=1)(A)
    # Transpose the axes (going back to the original order)
    A = jnp.transpose(A, (2, 3, 0,1))
    return A
