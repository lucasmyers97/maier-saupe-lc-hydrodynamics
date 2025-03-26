"""
This script finds the energy minima for a variety of kappa values and sees
where the various transitions happen.
"""

import sys
sys.path.insert(0,'/home/lucas/Documents/research/py-ball-majumdar-singular-potential')

import ball_majumdar_singular_potential as bmsp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import argrelmin

sp = bmsp.singular_potential_quasi_2D(974, 1.0, 1e-9, 100)

sp = bmsp.singular_potential_quasi_2D(974, 1.0, 1e-9, 100)
S_arr = np.linspace(-0.48, 0.98, 2000)
energy = np.zeros(S_arr.shape)

def calc_energy(kappa, b):
    for i, S in enumerate(S_arr):
        Q = S * np.array([2.0 / 3.0, -1.0 / 3.0, 0.0])
        
        sp.invert_Q(Q)
        Z = sp.return_Z()
        L = sp.return_Lambda()
    
        mean_field_term = -kappa * (Q[0]*Q[0] + Q[1]*Q[1] + Q[0]*Q[1] + Q[2]*Q[2] )
        
        entropy_term = (
                2*Q[0]*L[0] + 2*Q[1]*L[1] + Q[0]*L[1] + Q[1]*L[0] + 2*Q[2]*L[2]
                - np.log(Z) + np.log(4*np.pi)
                )

        cubic_term = b * (
                    -3*Q[0]**2*Q[1] - 3*Q[0]*Q[1]**2 + 3*Q[0]*Q[2]**2 + 3*Q[1]*Q[2]**2
                )
    
        energy[i] = mean_field_term + entropy_term + cubic_term

    return energy

def get_local_minima(energy):
    return S_arr[ argrelmin(energy) ]

# 6.75 superheated
# 7.5 supercooled
# T = 2/kappa
print(2 / 6.75)
print(2 / 7.5)
print(4/15)
energy = calc_energy(6.73, 0.0)
print(get_local_minima(energy))
