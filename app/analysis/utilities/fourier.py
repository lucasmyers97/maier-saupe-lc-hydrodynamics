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
