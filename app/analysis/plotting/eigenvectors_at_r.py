import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utilities import nematics as nu

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'lines.linewidth': 2})

linestyles = ['--', '-.', ':']

def get_commandline_args():

    descrption = ('Plots director angle as a function of polar angle at '
                  'different distances away from the core center. '
                  'Needs core structure from `get_points_around_defects`. '
                  'This expects Q-components to be arranged in an n x 5 '
                  'where n is n_r_points x n_theta_points')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--data_folder',
                        help='folder where core structure data is held')
    parser.add_argument('--input_filename',
                        help='h5 file with structure data')
    parser.add_argument('--group_key',
                        help='key in h5 file with the core data')
    parser.add_argument('--dists_from_center',
                        nargs='+',
                        type=float,
                        help=('distances away from core at which to plot '
                              'director angle'))
    parser.add_argument('--domain_center_offset',
                        type=float,
                        help='position of disclination center relative to domain center')
    parser.add_argument('--plot_filename',
                        help='name of file plot (must be png)')
    parser.add_argument('--defect_charge',
                        type=float,
                        help='charge of defect for core_structure choice')

    parser.add_argument('--dzyaloshinskii_filename',
                        help='name of h5 file with dzyaloshinskii solution')

    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)
    dzyaloshinskii_filename = None
    if args.dzyaloshinskii_filename:
        dzyaloshinskii_filename = os.path.join(args.data_folder,
                                               args.dzyaloshinskii_filename)

    return (input_filename, args.group_key, args.dists_from_center,
            args.domain_center_offset,
            plot_filename, args.defect_charge, dzyaloshinskii_filename)



def get_dzyaloshinskii_offset(polar_angle, director_angle, offset, r):

    def dzyaloshinskii_offset(input_polar_angle):
        x1 = r * np.cos(input_polar_angle)
        x2 = x1 + offset
        y = r * np.sin(input_polar_angle)

        input_polar_offset = np.arctan2(y, x2)
        input_polar_offset[input_polar_offset < 0] += 2 * np.pi

        return np.interp(input_polar_offset, polar_angle, director_angle)

    return dzyaloshinskii_offset



def main():

    (input_filename, group_key, dists_from_center, domain_center_offset,
     plot_filename, defect_charge, dzyaloshinskii_filename) = get_commandline_args()

    file = h5py.File(input_filename, 'r')
 
    eigensystem = file[group_key]
    n = np.array(eigensystem['n'][:])

    r0 = eigensystem.attrs['r0']
    rf = eigensystem.attrs['rf']
    n_r = eigensystem.attrs['n_r']
    n_theta = eigensystem.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    half_interval = theta <= np.pi

    n_x = n[:, 0].reshape((n_r, n_theta))
    n_y = n[:, 1].reshape((n_r, n_theta))
    phi = np.arctan2(n_y, n_x)

    shift = np.pi if defect_charge == -0.5 else 0

    fig, ax = plt.subplots(figsize=(4, 3))
    if dzyaloshinskii_filename:
        d_file = h5py.File(dzyaloshinskii_filename)
        d_theta = np.array(d_file['theta'][:])
        d_phi = np.array(d_file['phi'][:])

        dzyaloshinskii_offset = get_dzyaloshinskii_offset(d_theta, 
                                                          d_phi, 
                                                          domain_center_offset, 
                                                          dists_from_center[-1])

        ax.plot(theta[half_interval], 
                dzyaloshinskii_offset(theta)[half_interval] + shift)
                # label='Dzyaloshinskii')

    for (i, dist_from_center) in enumerate(dists_from_center):

        # find idx where r is closest to dist_from_center
        r_idx = np.argmin(np.abs(r - dist_from_center))

        phi_r = nu.sanitize_director_angle(phi[r_idx, :])
        phi_r -= np.min(phi_r)

        ax.plot(theta[half_interval], phi_r[half_interval], linestyle=linestyles[i], label="$r'$: {}".format(dist_from_center))

    ax.set_xlabel(r"$\varphi'$")
    ax.set_ylabel(r'$\theta$')

    if defect_charge == 0.5:
        ax.xaxis.set_ticks([0, np.pi / 2, np.pi],
                           labels=[r'$0$', r'$\pi / 2$', r'$\pi$'])
        ax.yaxis.set_ticks([0, np.pi / 4, np.pi / 2],
                           labels=[r'$0$', r'$\pi / 4$', r'$\pi / 2$'])
        ax.legend(loc='lower right')
    elif defect_charge == -0.5:
        ax.xaxis.set_ticks([0, np.pi / 2, np.pi],
                           labels=[r'$0$', r'$\pi / 2$', r'$\pi$'])
        ax.yaxis.set_ticks([np.pi / 2, 3 * np.pi / 4, np.pi],
                           labels=[r'$\pi / 2$', r'$3 \pi / 4$', r'$\pi$'])
        ax.legend(loc='lower left')

    fig.tight_layout()

    fig.savefig(plot_filename)

    plt.show()




if __name__ == '__main__':
    main()
