"""
This script produces an interactive plot of the modified singular potential
energy landscape based on the reviewer suggestion about including the 3rd 
Virial term. 
"""

import sys
sys.path.insert(0,'/home/lucas/Documents/research/py-ball-majumdar-singular-potential')

import ball_majumdar_singular_potential as bmsp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import argrelmin

sp = bmsp.singular_potential_quasi_2D(974, 1.0, 1e-9, 100)
S_arr = np.linspace(-0.48, 0.98, 200)
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

init_kappa = 8.0
init_b = 0.0

fig, ax = plt.subplots()

ymin, ymax = -0.6, 2.0
line, = ax.plot(S_arr, calc_energy(init_kappa, init_b), lw=2)
vlines = ax.vlines(get_local_minima(energy), ymin, ymax, ls='--')
ax.set_xlabel('S')
ax.set_ylim([ymin, ymax])

ax_kappa = fig.add_axes([0.25, 0.0, 0.65, 0.03])
slider_kappa = Slider(
    ax=ax_kappa,
    label='Kappa',
    valmin=0.0,
    valmax=20.0,
    valinit=init_kappa,
)

ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
slider_b = Slider(
    ax=ax_b,
    label='b',
    valmin=-10.0,
    valmax=10.0,
    valinit=init_b,
    orientation='vertical'
)


def update(val):
    energy = calc_energy(slider_kappa.val, slider_b.val)
    line.set_ydata(energy)
    mins = get_local_minima(energy)
    # vlines.set_segments([[[mins[0], ymin], [mins[0], ymax]],
    #                      [[mins[1], ymin], [mins[1], ymax]]])
    vlines.set_segments([[[min, ymin], [min, ymax]] for min in mins])
    fig.canvas.draw_idle()

slider_kappa.on_changed(update)
slider_b.on_changed(update)

plt.show()
