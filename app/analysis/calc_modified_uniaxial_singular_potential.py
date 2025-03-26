"""
This script produces an interactive plot of the modified singular potential
energy landscape based on the reviewer suggestion about including the 3rd 
Virial term. 
"""
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import erf
from scipy.special import erfi
from scipy.optimize import fsolve

# import scienceplots
# plt.style.use('science')

# mpl.rcParams['figure.dpi'] = 300

def I1(Sigma):
    result = np.zeros(Sigma.shape)
    pos = Sigma > 0
    neg = Sigma < 0
    zero = Sigma == 0
    result[pos] = np.sqrt(np.pi / Sigma[pos]) * erfi(np.sqrt(Sigma[pos]))
    result[neg] = np.sqrt(np.pi / -Sigma[neg]) * erf(np.sqrt(-Sigma[neg]))
    result[zero] = 2

    return result


def I2(Sigma):
    result = np.zeros(Sigma.shape)
    pos = Sigma > 0
    neg = Sigma < 0
    zero = Sigma == 0
    result[pos] = 1/Sigma[pos] * (np.exp(Sigma[pos]) - 0.5 * np.sqrt(np.pi / Sigma[pos]) * erfi(np.sqrt(Sigma[pos])))
    result[neg] = 1/Sigma[neg] * (np.exp(Sigma[neg]) - 0.5 * np.sqrt(np.pi / -Sigma[neg]) * erf(np.sqrt(-Sigma[neg])))
    result[zero] = 2/3

    return result


def I3(Sigma):
    result = np.zeros(Sigma.shape)
    pos = Sigma > 0
    neg = Sigma < 0
    zero = Sigma == 0
    result[pos] = (
            1/(2 * Sigma[pos]**2) 
            * (np.exp(Sigma[pos]) * (2 * Sigma[pos] - 3) 
               + 1.5 * np.sqrt(np.pi / Sigma[pos]) * erfi(np.sqrt(Sigma[pos])))
            )
    result[neg] = (
            1/(2 * Sigma[neg]**2) 
            * (np.exp(Sigma[neg]) * (2 * Sigma[neg] - 3) 
               + 1.5 * np.sqrt(np.pi / -Sigma[neg]) * erf(np.sqrt(-Sigma[neg])))
            )
    result[zero] = 2/5

    return result

def calc_S(Sigma):
    return 1.5 * I2(Sigma) / I1(Sigma) - 0.5

def calc_Sigma(S):
    Sigma = np.zeros(S.shape)
    for i, S_val in enumerate(S):
        f = lambda Sigma: calc_S(Sigma) - S_val
        Sigma[i] = fsolve(f, 0)[0]

    return Sigma


def calc_Z(Sigma):
    return 2 * np.pi * np.exp(-Sigma / 3) * I1(Sigma)


def calc_dS_dSigma(Sigma):
    S = calc_S(Sigma)
    Z = calc_Z(Sigma)
    return (-1/3 * (S + 0.5) * (2*S + 1) 
            + 1/Z * 3 * np.pi * np.exp(-Sigma / 3) * I3(Sigma))


def calc_energy(S, Sigma, Z, kappa, B):
    return (-kappa/3 * S**2 
            + 2/9 * B * S**3 
            + np.log(4 * np.pi) 
            - np.log(Z) 
            + 2/3 * Sigma * S )


def calc_energy_derivative(S, Sigma, kappa, B):
    dS_dSigma = calc_dS_dSigma(Sigma)
    return 2/3 * dS_dSigma * (
            -kappa * S + B * S**2 + Sigma
            )


def energy_minimization_equation(Sigma, kappa, B):
    S = 1.5 * I2(Sigma) / I1(Sigma) - 0.5
    return Sigma - kappa * S + B * S**2


def energy_minimization_jacobian(Sigma, kappa, B):
    dS_dSigma = calc_dS_dSigma(Sigma)
    S = calc_S(Sigma)
    return 1 - kappa * dS_dSigma + 2 * B * S * dS_dSigma


def calc_local_extrema(kappa, B):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        energy_min_eq = lambda x: energy_minimization_equation(x, kappa, B)
        energy_min_jac = lambda x: energy_minimization_jacobian(x, kappa, B)
        min_Sigma = np.array([fsolve(energy_min_eq, 
                                     Sigma0, 
                                     fprime=energy_min_jac, 
                                     xtol=1e-10)[0] 
                              for Sigma0 in [-100, 0, 100]])
        min_S = 1.5 * I2(min_Sigma) / I1(min_Sigma) - 0.5
        min_Z = 2 * np.pi * np.exp(-min_Sigma / 3) * I1(min_Sigma)
        min_energy = calc_energy(min_S, min_Sigma, min_Z, kappa, B)

    return min_S, min_energy


def superheated_equation(Sigma, B):
    S = calc_S(Sigma)
    dSigma_dS = 1 / calc_dS_dSigma(Sigma)
    return Sigma - S * (dSigma_dS + B * S)


def calc_superheated_kappa(B):

    f = lambda x: superheated_equation(x, B)
    Sigma = fsolve(f, 100, xtol=1e-11)[0]
    S = calc_S(Sigma)
    return Sigma / S + B * S


def calc_nematic_kappa(S0, B):

    Sigma = calc_Sigma(S0)
    return (Sigma + B * S0**2) / S0


def main():
    # S = np.array([0.6750865826195644])
    # Sigma = calc_Sigma(S)
    # Z = calc_Z(Sigma)
    # print(np.log(4*np.pi) - np.log(Z) + 2/3 * Sigma * S)


    S = np.linspace(-0.48, 0.98, 1000)
    Sigma = calc_Sigma(S)
    Z = calc_Z(Sigma)

    S0 = calc_local_extrema(8.0, 0.0)[0][-1]
    print('S0 = {}'.format(S0))

    B_vals = [0.0, 2.002, 2.893, 4.204]
    for B_val in B_vals:
        kappa0 = calc_nematic_kappa(np.array([S0]), B_val)
        Sigma0 = calc_Sigma(np.array([S0]))
        Z0 = calc_Z(Sigma0)

        E0 = calc_energy(S0, Sigma0, Z0, kappa0, B_val)

        print('kappa = {}'.format(kappa0))
        print('energy = {}'.format(E0))

    B = np.linspace(0, 10, 1000)
    kappa = np.zeros(B.shape)
    for i, B_val in enumerate(B):
        kappa[i] = calc_superheated_kappa(B_val)

    # b = 2.3
    T_diff = (2 / kappa - 4/15) * 15/4
    plt.plot(B, T_diff)
    plt.xlabel(r'$B$')
    plt.ylabel(r'$\frac{T_I - T^*}{T^*}$')

    T_vals = [1/27, 5/270, 1/270]
    T_labels = ['DSCG Max', 'DSCG Min', 'Thermotropics']
    idxs = [np.argmin( np.abs(T_diff - T) ) for T in T_vals]
    delta_x = 0.1
    delta_y = 0.005
    for idx, T_label in zip(idxs, T_labels):
        plt.plot(B[idx], T_diff[idx], ls='', marker='o', label=T_label)
        plt.text(B[idx] + delta_x, T_diff[idx] + delta_y, 
                 '({:.3f}, {:.3f})'.format(B[idx], T_diff[idx]),
                 horizontalalignment='left',
                 verticalalignment='bottom')
    plt.tight_layout()
    plt.legend(frameon=True)
    plt.show()

    init_kappa = 8.0
    init_b = 0.0
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.tick_params(axis='y', labelcolor='blue')
    fig.subplots_adjust(left=0.2, bottom=0.13, right=0.85)

    ymin, ymax = -2, 2
    
    energy = calc_energy(S, Sigma, Z, init_kappa, init_b)
    dE_dSigma = calc_energy_derivative(S, Sigma, init_kappa, init_b)
    min_S, min_energy = calc_local_extrema(init_kappa, init_b)

    line, = ax.plot(S, energy, lw=2, color='blue')
    line2, = ax2.plot(S, dE_dSigma, lw=2, color='black')
    ax2.hlines(0, S[0], S[-1], color='black', ls='--')
    vlines = ax.vlines(min_S, ymin, ymax, ls='--', color='blue')
    hlines = ax.hlines(min_energy, S[0], S[-1], ls='--', color='blue')
    marker, = ax.plot(min_S, min_energy, ls='', marker='o')
    
    ax.set_xlabel(r'$S$')
    ax.set_ylabel(r'$F$', color='blue')
    ax2.set_ylabel(r'$dF/d\Sigma$')
    ax.set_ylim([ymin, ymax])
    
    ax_kappa = fig.add_axes([0.25, 0.0, 0.65, 0.03])
    slider_kappa = Slider(
        ax=ax_kappa,
        label='Kappa',
        valmin=0.0,
        valmax=20.0,
        valinit=init_kappa,
    )
    
    ax_b = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
    slider_b = Slider(
        ax=ax_b,
        label='b',
        valmin=-1.0,
        valmax=10.0,
        valinit=init_b,
        orientation='vertical'
    )

    def update(_):
        energy = calc_energy(S, Sigma, Z, slider_kappa.val, slider_b.val)
        dE_dSigma = calc_energy_derivative(S, Sigma, slider_kappa.val, slider_b.val)
        line.set_ydata(energy)
        line2.set_ydata(dE_dSigma)
        ymin_p = min(ymin, np.min(energy))
        ymax_p = max(ymax, np.max(energy))
        ax.set_ylim([ymin_p, ymax_p])
    
        min_S, min_energy = calc_local_extrema(slider_kappa.val, slider_b.val)
    
        vlines.set_segments([[[min_S_val, ymin_p], [min_S_val, ymax_p]] 
                             for min_S_val in min_S])
        hlines.set_segments([[[S[0], min_energy_val], [S[-1], min_energy_val]] 
                             for min_energy_val in min_energy])
        marker.set_ydata(min_energy)
        marker.set_xdata(min_S)
        
        fig.canvas.draw_idle()
    
    slider_kappa.on_changed(update)
    slider_b.on_changed(update)
    
    plt.show()

if __name__ == '__main__':
    main()
