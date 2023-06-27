import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import binom

from ..utilities import nematics as nu
from ..utilities import fourier

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    descrption = ('Plots n = 2 sin Fourier mode as a function of distance in '
                  'the far field, and tries to fit to the analytic solution. '
                  'This has the form A \ln(r) + B'
                  'Needs director structure from PerturbativeDirectorSystem.')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='folder where director data is held')
    parser.add_argument('--structure_filename',
                        dest='structure_filename',
                        help='h5 file with director data')
    parser.add_argument('--data_key',
                        dest='data_key',
                        help='key in h5 file with the data that will be plotted')

    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of cosine file plot (must be png)')

    parser.add_argument('--r_cutoff',
                        type=float,
                        help='near-field cutoff for r')
    parser.add_argument('--n_terms',
                        type=int,
                        help='number of terms to use in analytic solution')

    args = parser.parse_args()

    structure_filename = os.path.join(args.data_folder, args.structure_filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return structure_filename, args.data_key, plot_filename, args.r_cutoff, args.n_terms



def fit_curve(r, A, B):

    return A * np.log(r) + B



def k1(k, m, n):

    return binom(0.5, k) * binom(-0.5, m) * binom(-0.5, k + m - n) * binom(-1.5, n)

def k2(k, m, n):

    return binom(0.5, k) * binom(0.5, k + m - n) * binom(-2.5, m) * binom(-0.5, n)

def k3(k, m, n):

    return binom(0.5, k) * binom(-0.5, m) * binom(-0.5, k + m - n) * binom(-1.5, n)



def analytic_solution(n_terms):

    A = 0
    B = 0
    for k in range(n_terms):
        for m in range(n_terms):
            for n in range(n_terms):
                if (k + m - n) < 0:
                    continue

                A += ( (0.75 * k1(k, n, m) - 1.25 * k2(k, m, n) + 0.5 * k3(k, m, n))
                      / (2*k + 2*m + 2) )
                B -= ( (0.75 * k1(k, n, m) - 1.25 * k2(k, m, n) + 0.5 * k3(k, m, n))
                      / (2*k + 2*m + 2)**2 )

    return A, B




def main():

    (structure_filename, data_key, 
     plot_filename, r_cutoff, n_terms) = get_commandline_args()

    file = h5py.File(structure_filename, 'r')

    phi_data = file[data_key]

    r0 = phi_data.attrs['r_0']
    rf = phi_data.attrs['r_f']
    n_r = phi_data.attrs['n_r']
    n_theta = phi_data.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)

    phi = np.array(phi_data[:]).reshape((n_r, n_theta))

    B2_r = np.zeros(n_r)
    for i in range(n_r):
        phi_r = phi[i, :]
        _, B2_phi = fourier.calculate_trigonometric_fourier_coefficients(phi_r)

        # Only care about the n = 2 mode
        B2_r[i] = B2_phi[2]

    farfield_idx = r > r_cutoff

    fit, cov = curve_fit(fit_curve, r[farfield_idx], B2_r[farfield_idx])
    print(fit)

    A, B = analytic_solution(n_terms)
    print(A, B)

    # regular plots
    fig_B2, ax_B2 = plt.subplots()

    ax_B2.plot(r[farfield_idx], B2_r[farfield_idx], label=r'numerical solution')
    ax_B2.plot(r[farfield_idx], fit_curve(r[farfield_idx], fit[0], fit[1]), 
               label=r'$A \ln(r) + B \\ A = {:.2f}, B = {:.2f}$'.format(fit[0], fit[1]))

    ax_B2.set_title(r'$\sin(2\varphi)$ coefficient vs. $r$')
    ax_B2.set_xlabel(r'$r / \xi$')
    ax_B2.set_ylabel(r'$\sin(2\varphi)$ coefficient')
    ax_B2.legend()

    fig_B2.tight_layout()
    fig_B2.savefig(plot_filename)

    plt.show()




if __name__ == '__main__':
    main()
