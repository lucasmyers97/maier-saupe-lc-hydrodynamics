import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utilities import nematics as nu

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

linestyles = ['--', '-.', ':']

def get_commandline_args():

    descrption = ('Plots director angle as a function of polar angle at '
                  'different distances away from the core center. '
                  'Needs core structure from `get_points_around_defects`. '
                  'This expects Q-components to be arranged in an n x 5 '
                  'where n is n_r_points x n_theta_points. '
                  'Also plots Dzyaloshinskii, but centered at a given '
                  'displacement')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='folder where core structure data is held')
    parser.add_argument('--structure_filename',
                        dest='structure_filename',
                        help='h5 file with structure data')
    parser.add_argument('--data_key',
                        dest='data_key',
                        help=('key in h5 file with the core data that will be '
                              'plotted. Note that this will be expected to '
                              'be of form `timestep_{}/something` where the '
                              'string will be later formatted with the '
                              'timesteps'))
    parser.add_argument('--dists_from_center',
                        dest='dists_from_center',
                        nargs='+',
                        type=float,
                        help=('distances away from core at which to plot '
                              'director angle'))
    parser.add_argument('--dzyaloshinskii_offset',
                        dest='dzyaloshinskii_offset',
                        type=float,
                        help=('where to center Dzyaloshinskii solution '
                              'relative to disclination center'))
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of file plot (must be png)')

    parser.add_argument('--dzyaloshinskii_filename',
                        dest='dzyaloshinskii_filename',
                        help='name of h5 file with dzyaloshinskii solution')

    args = parser.parse_args()

    structure_filename = os.path.join(args.data_folder, 
                                      args.structure_filename)
    plot_filename = os.path.join(args.data_folder, 
                                 args.plot_filename)
    dzyaloshinskii_filename = os.path.join(args.data_folder,
                                           args.dzyaloshinskii_filename)

    return (structure_filename, args.data_key, args.dists_from_center,
            args.dzyaloshinskii_offset, plot_filename, dzyaloshinskii_filename)



def offset_polar_coords(polar_coord, radius, offset):

    x = radius * np.cos(polar_coord)
    y = radius * np.sin(polar_coord)

    x -= offset

    return np.arctan2(y, x)



def main():

    (structure_filename,data_key, dists_from_center, dzyaloshinskii_offset,
     plot_filename, dzyaloshinskii_filename) = get_commandline_args()

    file = h5py.File(structure_filename, 'r')

    fig, ax = plt.subplots()
    Q_vec = file[data_key]
    Q_vec_data = np.array(Q_vec[:])
    Q_data = nu.Q_vec_to_mat(Q_vec_data)

    _, _, n, _ = nu.eigensystem_from_Q(Q_data)

    r0 = Q_vec.attrs['r0']
    rf = Q_vec.attrs['rf']
    n_r = Q_vec.attrs['n_r']
    n_theta = Q_vec.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    n_x = n[:, 0].reshape((n_r, n_theta))
    n_y = n[:, 1].reshape((n_r, n_theta))
    phi = np.arctan2(n_y, n_x)

    d_file = h5py.File(dzyaloshinskii_filename)
    d_theta = np.array(d_file['theta'][:])
    d_phi = np.array(d_file['phi'][:])
    ax.plot(d_theta, d_phi, label='Dzyaloshinskii solution')

    dzyaloshinskii_func = lambda polar_coord: np.interp(polar_coord, d_theta, d_phi)

    for dist_from_center in dists_from_center:
        # find idx where r is closest to dist_from_center
        r_idx = np.argmin(np.abs(r - dist_from_center))
        phi_r = nu.sanitize_director_angle(phi[r_idx, :])
        phi_r -= np.min(phi_r)

        dzyaloshinskii_polar = offset_polar_coords(theta, dist_from_center, dzyaloshinskii_offset)

        ax.plot(theta, dzyaloshinskii_func(dzyaloshinskii_polar), label=r'$r_d$: {}'.format(dist_from_center))
        ax.plot(theta, phi_r, label='$r$: {}'.format(dist_from_center), linestyle='--')

    ax.set_xlabel(r'polar angle $\phi$')
    ax.set_ylabel(r'director angle $\theta$')
    ax.set_title(r'Director angle of defect')
    ax.legend()

    fig.tight_layout()

    plt.show()



if __name__ == '__main__':
    main()
