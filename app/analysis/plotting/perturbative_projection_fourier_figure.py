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

colors = ['b', 'r', 'g', 'm', 'k']
linestyles = [':', '--', '-.', (0, (1, 10))]

def get_commandline_args():

    descrption = ('Plots director angle as a function of polar angle at '
                  'different distances away from the core center. '
                  'Needs core structure from `get_points_around_defects`. '
                  'This expects Q-components to be arranged in an n x 5 '
                  'where n is n_r_points x n_theta_points')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='folder where core structure data is held')
    parser.add_argument('--structure_filenames',
                        nargs=2,
                        help='h5 file with structure data')
    parser.add_argument('--defect_distance',
                        dest='defect_distance',
                        type=float,
                        help=('distance between defects if you are fixing the '
                              'defects in place'))
    parser.add_argument('--data_keys',
                        nargs=2,
                        help=('keys in h5 file with the structure data that '
                              'will be plotted.'))

    parser.add_argument('--timestep',
                        dest='timestep',
                        type=int,
                        help='timestep number in h5 file to plot')
    parser.add_argument('--dt',
                        dest='dt',
                        type=float,
                        help='timestep size')
    parser.add_argument('--n_modes',
                        dest='n_modes',
                        type=int,
                        help='number of Fourier modes to plot')
    parser.add_argument('--r_range',
                        type=float,
                        nargs=2,
                        help='limits on the r range to plot')
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of file plot (must be png)')

    args = parser.parse_args()

    structure_filenames = [os.path.join(args.data_folder, args.structure_filenames[0]),
                           os.path.join(args.data_folder, args.structure_filenames[1])]
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return (structure_filenames, args.data_keys,
            args.timestep, args.dt,
            args.n_modes, plot_filename, args.defect_distance, args.r_range)



def main():

    (structure_filenames, data_keys, time, dt, n_modes,
     plot_filename, defect_distance, r_range) = get_commandline_args()

    file = h5py.File(structure_filenames[0], 'r')
    d = defect_distance

    Q_vec = file[data_keys[0].format(time)]
    Q_vec_data = np.array(Q_vec[:])
    Q_data = nu.Q_vec_to_mat(Q_vec_data)

    _, _, n, _ = nu.eigensystem_from_Q(Q_data)

    r0 = Q_vec.attrs['r0']
    rf = Q_vec.attrs['rf']
    n_r = Q_vec.attrs['n_r']
    n_theta = Q_vec.attrs['n_theta']

    file.close()

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    n_x = n[:, 0].reshape((n_r, n_theta))
    n_y = n[:, 1].reshape((n_r, n_theta))
    phi = np.arctan2(n_y, n_x)

    Bn_r = np.zeros((n_r, n_modes))
    for i in range(n_r):
        phi_offset = nu.pairwise_defect_director_at_midpoint(theta, 
                                                             r[i], 
                                                             d, 
                                                             -0.5, 
                                                             0.5)
        # get correct isomorph
        phi_offset -= np.pi/2
        phi_r = nu.sanitize_director_angle(phi[i, :]) 
        phi_offset = nu.sanitize_director_angle(phi_offset)

        _, Bn_phi = fourier.calculate_trigonometric_fourier_coefficients(phi_r - phi_offset)

        Bn_r[i, :] = Bn_phi[:n_modes]

    # Get perturbative director data
    file = h5py.File(structure_filenames[1], 'r')

    phi_data = file[data_keys[1]]

    r0 = phi_data.attrs['r_0']
    rf = phi_data.attrs['r_f']
    n_r = phi_data.attrs['n_r']
    n_theta = phi_data.attrs['n_theta']

    r_dir = np.linspace(r0, rf, num=n_r)

    phi = np.array(phi_data[:]).reshape((n_r, n_theta))

    Bn_phi_r = np.zeros((n_r, n_modes))
    for i in range(n_r):
        phi_r = phi[i, :]
        _, Bn_phi = fourier.calculate_trigonometric_fourier_coefficients(phi_r)

        Bn_phi_r[i, :] = Bn_phi[:n_modes]

    idx = np.logical_and(r_dir > r_range[0], r_dir < r_range[1])

    # make plots
    fig_Bn, ax_Bn = plt.subplots(figsize=(4.5, 3))

    for i in range(1, n_modes):
        ax_Bn.plot(1/r_dir[idx], Bn_phi_r[idx, i], c='c')

    eps = 0.1
    for i in range(1, n_modes):
        ax_Bn.plot(1/r, (1 / eps) * Bn_r[:, i], label=r'$n = {}$'.format(i), c=colors[i - 1], ls=linestyles[i - 1])

    ax_Bn.set_xlabel(r'$1 / r$')
    ax_Bn.set_ylabel(r'$A_n$')
    ylims = ax_Bn.get_ylim()
    ax_Bn.set_ylim(bottom=ylims[0]*1.3)
    ax_Bn.legend(loc='lower left', borderpad=0.2, labelspacing=0.2, borderaxespad=0.1)

    print("Axes are: {}, {}", ax_Bn.get_xlim(), ax_Bn.get_ylim())

    fig_Bn.tight_layout()
    fig_Bn.savefig(plot_filename.format(time))

    plt.show()




if __name__ == '__main__':
    main()
