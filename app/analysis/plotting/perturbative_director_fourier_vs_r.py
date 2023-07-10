import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utilities import nematics as nu
from ..utilities import fourier

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    descrption = ('Plots director angle as a function of polar angle at '
                  'different distances away from the domain center. '
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

    parser.add_argument('--n_modes',
                        dest='n_modes',
                        type=int,
                        help='number of Fourier modes to plot')
    parser.add_argument('--r_cutoff',
                        dest='r_cutoff',
                        type=float,
                        default=float('inf'),
                        help='number of Fourier modes to plot')

    parser.add_argument('--cos_plot_filename',
                        dest='cos_plot_filename',
                        help='name of cosine file plot (must be png)')
    parser.add_argument('--sin_plot_filename',
                        dest='sin_plot_filename',
                        help='name of sine file plot (must be png)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--core_structure', 
                       action='store_true',
                       help='denotes the analysis is for a core structure')
    group.add_argument('--outer_structure', 
                       action='store_false',
                       help=('denotes the analysis is for the outer ' 
                             'structure (away from defects)'))

    args = parser.parse_args()

    structure_filename = os.path.join(args.data_folder, 
                                      args.structure_filename)
    cos_plot_filename = os.path.join(args.data_folder, 
                                     args.cos_plot_filename)
    sin_plot_filename = os.path.join(args.data_folder, 
                                     args.sin_plot_filename)
    log_cos_plot_filename = os.path.join(args.data_folder, 
                                         'log_' + args.cos_plot_filename)
    log_sin_plot_filename = os.path.join(args.data_folder, 
                                         'log_' + args.sin_plot_filename)

    return (structure_filename, args.data_key, 
            args.n_modes, args.r_cutoff,
            args.core_structure,
            cos_plot_filename, sin_plot_filename,
            log_cos_plot_filename, log_sin_plot_filename)



def main():

    (structure_filename, data_key,
     n_modes, r_cutoff,
     core_structure,
     cos_plot_filename, sin_plot_filename,
     log_cos_plot_filename, log_sin_plot_filename) = get_commandline_args()

    file = h5py.File(structure_filename, 'r')

    phi_data = file[data_key]

    r0 = phi_data.attrs['r_0']
    rf = phi_data.attrs['r_f']
    n_r = phi_data.attrs['n_r']
    n_theta = phi_data.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    phi = np.array(phi_data[:]).reshape((n_r, n_theta))

    An_r = np.zeros((n_r, n_modes))
    Bn_r = np.zeros((n_r, n_modes))
    for i in range(n_r):
        phi_r = phi[i, :]
        An_phi, Bn_phi = fourier.calculate_trigonometric_fourier_coefficients(phi_r)

        An_r[i, :] = An_phi[:n_modes]
        Bn_r[i, :] = Bn_phi[:n_modes]

    # regular plots
    fig_An, ax_An = plt.subplots()
    fig_Bn, ax_Bn = plt.subplots()

    x_axis = None
    x_label = None
    idx = r < r_cutoff
    if core_structure:
        x_axis = r
        x_label = r'$r / \xi$'
    else:
        x_axis = 1/r
        x_label = r'$\xi / r$'

    # for i in range(n_modes):
    #     ax_An.plot(x_axis, An_r[:, i], label=r'$n = {}$'.format(i))

    # for i in range(1, n_modes):
    #     ax_Bn.plot(x_axis, Bn_r[:, i], label=r'$n = {}$'.format(i))

    degree = 1
    for i in range(n_modes):
        coef = np.polynomial.polynomial.polyfit(x_axis[idx], An_r[idx, i], degree)
        ax_An.plot(x_axis[idx], An_r[idx, i], label=r'$n = {}$'.format(i))
        # ax_An.plot(x_axis[idx], np.polynomial.polynomial.polyval(x_axis[idx], coef), linestyle='--',
        #            # label=r'${:.2e} + {:.2e}x + {:.2e}x^2$'.format(coef[0], coef[1], coef[2]))
        #            label=r'${:.2e} + {:.2e}x$'.format(coef[0], coef[1]))

    for i in range(1, n_modes):
        coef = np.polynomial.polynomial.polyfit(x_axis[idx], Bn_r[idx, i], degree)
        ax_Bn.plot(x_axis[idx], Bn_r[idx, i], label=r'$n = {}$'.format(i))
        # ax_Bn.plot(x_axis[idx], np.polynomial.polynomial.polyval(x_axis[idx], coef), linestyle='--',
        #            # label=r'${:.2e} + {:.2e}x + {:.2e}x^2$'.format(coef[0], coef[1], coef[2]))
        #            label=r'${:.2e} + {:.2e}x$'.format(coef[0], coef[1]))

    ax_An.set_title(r'$\cos$ Fourier coefficients vs. $r$')
    ax_An.set_xlabel(x_label)
    ax_An.set_ylabel(r'$\cos$ Fourier coeffs')
    ax_An.legend()

    ax_Bn.set_title(r'$\sin$ Fourier coefficients vs. $r$')
    ax_Bn.set_xlabel(x_label)
    ax_Bn.set_ylabel(r'$\sin$ Fourier coeffs')
    ax_Bn.legend()

    fig_An.tight_layout()
    fig_An.savefig(cos_plot_filename)

    fig_Bn.tight_layout()
    fig_Bn.savefig(sin_plot_filename)

    # log plots
    fig_An_log, ax_An_log = plt.subplots()
    fig_Bn_log, ax_Bn_log = plt.subplots()

    for i in range(n_modes):
        ax_An_log.plot(np.log(r), np.log(np.abs(An_r[:, i])), label=r'$n = {}$'.format(i))

    for i in range(1, n_modes):
        ax_Bn_log.plot(np.log(r), np.log(np.abs(Bn_r[:, i])), label=r'$n = {}$'.format(i))

    ax_An_log.set_title(r'$\log$ of $\cos$ Fourier coefficients vs. $\log(r)$')
    ax_An_log.set_xlabel(r'$\log(r)$')
    ax_An_log.set_ylabel(r'$\log$ of $\cos$ Fourier coeffs')
    ax_An_log.legend()

    ax_Bn_log.set_title(r'$\log$ of $\sin$ Fourier coefficients vs. $\log(r)$')
    ax_Bn_log.set_xlabel(r'$\log(r)$')
    ax_Bn_log.set_ylabel(r'$\log$ of $\sin$ Fourier coeffs')
    ax_Bn_log.legend()

    fig_An_log.tight_layout()
    # fig_An.savefig(cos_plot_filename)

    fig_Bn_log.tight_layout()
    # fig_Bn.savefig(sin_plot_filename)

    plt.show()




if __name__ == '__main__':
    main()
