"""
Module providing simple utilities to deal with Fourier coefficients.
In particular, correctly calculates trigonometric Fourier coefficients.
"""
import numpy as np

def calculate_trigonometric_fourier_coefficients(x):
    """
    Takes input `x` which is some function x sampled evenly across the periodic
    interval.
    Returns An and Bn whose nth entry is the nth cosine or sine Fourier mode
    amplitude respectively.
    """

    FT_x = np.fft.rfft(x)
    N = x.shape[0]
    An = 2 * FT_x.real / N
    Bn = -2 * FT_x.imag / N

    return An, Bn



def calculate_trigonometric_fourier_coefficients_vs_r(X):
    """
    Takes n_r x n_theta array of a field evaluated at a set of polar 
    coordinates (r, theta) and returns trigonometric fourier coefficients at
    each radial point r, and up to n_modes 
    (see numpy rfft for n_modes explanation).

    Parameters
    ----------
    X : array_like
        n_r x n_theta array which holds some field X(r, theta) evaluated at 
        a set of evenly-spaced polar coordinates r and theta.

    Returns
    -------
    An_r : ndarray
        n_r x n_theta array holding cosine Fourier modes up to n_modes for 
        each r.
    Bn_r : ndarray
        n_r x n_modes array holding sine Fourier modes up to n_modes for 
        each `r`.
    """
    n_r, n_theta = X.shape
    n_modes = None
    if (n_theta % 2 == 0):
        n_modes = (n_theta // 2) + 1
    else:
        n_modes = (n_theta + 1) // 2

    An_r = np.zeros((n_r, n_modes))
    Bn_r = np.zeros((n_r, n_modes))

    for i in range(n_r):

        An, Bn = calculate_trigonometric_fourier_coefficients(X[i, :])

        An_r[i, :] = An
        Bn_r[i, :] = Bn

    return An_r, Bn_r
