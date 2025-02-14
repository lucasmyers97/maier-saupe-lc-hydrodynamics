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

    desc = ('Calculates Sigma, Z, and nondimensional entropy for uniaxial '
            'nematic configuration, given the scalar order parameter S.')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--S', type=float, help='uniaxial order parameter')
    args = parser.parse_args()

    return args.S



def Z(Sigma):

    sSigma = np.sqrt(Sigma)
    return ( 2 * np.pi * np.exp(-Sigma / 3)
             * 2 * np.exp(Sigma) * dawsn( sSigma ) / sSigma )



def f(S, Sigma):

    sSigma = np.sqrt(Sigma)

    return dawsn(sSigma) * (2 * Sigma / 3 * (2 * S + 1) + 1) - sSigma



def main():

    S = get_commandline_args()

    if S == 0:
        print('Sigma = 0')
        print('Z = 4 pi')
        return

    zero_func = lambda Sigma: f(S, Sigma)
    sol = root_scalar(zero_func, x0=1e-8, bracket=[0.1, 100])

    print(sol)
    print('Z = {}'.format(Z(sol.root)))

    Sigma = np.linspace(0, 2*np.pi, 1000)
    plt.plot(Sigma, zero_func(Sigma))
    plt.plot(sol.root, 0, marker='o')
    plt.show()



if __name__ == '__main__':
    main()
