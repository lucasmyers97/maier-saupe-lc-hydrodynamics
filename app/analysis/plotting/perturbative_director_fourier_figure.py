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
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'lines.linewidth': 2})

markers = ['o', 's', '+', 'x', '^']
colors = ['b', 'r', 'g', 'm', 'k']
linestyles = [':', '--', '-.', (0, (1, 10))]

def get_commandline_args():

    descrption = ('Plots theta_c Fourier coefficients as a function of 1/r. '
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
    parser.add_argument('--r_range',
                        dest='r_range',
                        nargs='*',
                        type=float,
                        default=float('inf'),
                        help='2-component list containing fitted r-range')

    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of Fourier mode file plot (must be png)')

    args = parser.parse_args()

    structure_filename = os.path.join(args.data_folder, args.structure_filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return (structure_filename, args.data_key, 
            args.n_modes, args.r_range, plot_filename)



def get_evenly_spaced_point_idx(points, min, max, n_points):

    spaced_pts = np.linspace(min, max, n_points)
    idx = np.zeros(spaced_pts.shape, dtype=np.int64)
    for i, pt in enumerate(spaced_pts):
        idx[i] = np.argmin( np.abs(points - pt) )

    return idx



def main():

    (structure_filename, data_key,
     n_modes, r_range, plot_filename) = get_commandline_args()

    file = h5py.File(structure_filename, 'r')

    phi_data = file[data_key]

    r0 = phi_data.attrs['r_0']
    rf = phi_data.attrs['r_f']
    n_r = phi_data.attrs['n_r']
    n_theta = phi_data.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)

    phi = np.array(phi_data[:]).reshape((n_r, n_theta))

    Bn_r = np.zeros((n_r, n_modes))
    for i in range(n_r):
        phi_r = phi[i, :]
        _, Bn_phi = fourier.calculate_trigonometric_fourier_coefficients(phi_r)

        Bn_r[i, :] = Bn_phi[:n_modes]

    fig_Bn, ax_Bn = plt.subplots(figsize=(4.5, 3))

    idx = np.logical_and(r > r_range[0], r < r_range[1])
    sparse_idx = get_evenly_spaced_point_idx(1 / r[idx], 1 / r_range[1], 1/r_range[0], 25)

    coef1 = np.polynomial.polynomial.polyfit(1/r[idx], Bn_r[idx, 1], [1])
    coef2 = np.polynomial.polynomial.polyfit(1/r[idx], Bn_r[idx, 2], [2])
    coef3 = np.polynomial.polynomial.polyfit(1/r[idx], Bn_r[idx, 3], [1, 3])
    coef4 = np.polynomial.polynomial.polyfit(1/r[idx], Bn_r[idx, 4], [4])

    print(coef1)
    print(coef2)
    print(coef3)
    print(coef4)

    ax_Bn.plot(1/r[idx], 
               np.polynomial.polynomial.polyval(1/r[idx], coef1), 
               linestyle='-',
               c='y')

    ax_Bn.plot(1/r[idx], 
               np.polynomial.polynomial.polyval(1/r[idx], coef2), 
               linestyle='-',
               c='y')

    ax_Bn.plot(1/r[idx], 
               np.polynomial.polynomial.polyval(1/r[idx], coef3),
               linestyle='-',
               c='y')

    ax_Bn.plot(1/r[idx], 
               np.polynomial.polynomial.polyval(1/r[idx], coef4),
               linestyle='-',
               c='y')


    for i in range(1, n_modes):
        ax_Bn.plot(1/r[idx][sparse_idx], Bn_r[idx, i][sparse_idx], label=r'$n = {}$'.format(i), c=colors[i - 1], ls=linestyles[i - 1])


    ax_Bn.set_xlabel(r'$\xi / r$')
    ax_Bn.set_ylabel(r'$A_n$')
    ylims = ax_Bn.get_ylim()
    ax_Bn.set_ylim(bottom=ylims[0]*1.3)
    ax_Bn.legend(loc='lower left', borderpad=0.2, labelspacing=0.2, borderaxespad=0.1)

    fig_Bn.tight_layout()
    fig_Bn.savefig(plot_filename)

    plt.show()




if __name__ == '__main__':
    main()
