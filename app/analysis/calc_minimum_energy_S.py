import argparse

import numpy as np
from scipy.special import dawsn
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():

    desc = ('Plots energy density vs. S for uniform uniaxial configuration. '
            'Also finds minimum-energy S.')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--kappa', type=float, help='free energy alignment strength')
    args = parser.parse_args()

    return args.kappa



def calc_Z(Sigma):

    if Sigma == 0:
        return 4 * np.pi

    sSigma = np.sqrt(Sigma)
    return ( 2 * np.pi * np.exp(-Sigma / 3)
             * 2 * np.exp(Sigma) * dawsn( sSigma ) / sSigma )



def f(S, Sigma):

    sSigma = np.sqrt(Sigma)

    return dawsn(sSigma) * (2 * Sigma / 3 * (2 * S + 1) + 1) - sSigma



def calc_Sigma(S):

    if S == 0:
        return 0

    zero_func = lambda Sigma: f(S, Sigma)
    sol = root_scalar(zero_func, x0=1e-8, bracket=[1e-8, 100])
    return sol.root



def calc_S(Sigma):

    S = np.zeros(Sigma.shape)
    S[Sigma == 0] = 0

    sSigma = np.sqrt(Sigma[Sigma != 0])
    S[Sigma != 0] = 0.75 / Sigma[Sigma != 0] * (sSigma / dawsn(sSigma) - 1) - 0.5

    return S



def calc_Emf(S, kappa):

    return - kappa / 3 * S**2



def calc_Es(S, Sigma, Z):

    return np.log(4 * np.pi) - np.log(Z) + 2/3 * Sigma * S



def g(Sigma, kappa):

    sSigma = np.sqrt(Sigma)
    F = dawsn(sSigma)
    return (
            (1 / kappa) * Sigma**2 * F
            + 0.5 * Sigma * F
            - 0.75 * (sSigma - F)
            )



def main():

    kappa = get_commandline_args()

    # Calc energy curve
    S = np.linspace(0, 0.9, 1000)
    Sigma = np.zeros(S.shape)
    Z = np.zeros(Sigma.shape)

    for i in range(S.shape[0]):
        Sigma[i] = calc_Sigma(S[i])
        Z[i] = calc_Z(Sigma[i])

    E_mf = calc_Emf(S, kappa)
    E_s = calc_Es(S, Sigma, Z)

    # calc minimum-energy Sigma
    zero_func = lambda Sigma: g(Sigma, kappa)
    sol = root_scalar(zero_func, x0=1e-8, bracket=[1e-6, 100])
    print(sol)

    Sigma_min = np.array([sol.root])
    S_min = calc_S(Sigma_min)
    Z_min = calc_Z(Sigma_min)
    E_min = calc_Emf(S_min, kappa) + calc_Es(S_min, Sigma_min, Z_min)
    print('S_min = {}'.format(S_min[0]))

    plt.plot(S, E_mf + E_s)
    plt.plot(S_min, E_min, marker='o')
    plt.show()

    plt.plot(S, zero_func(Sigma))
    plt.plot(S_min, zero_func(Sigma_min), marker='o')
    plt.show()






if __name__ == '__main__':
    main()
