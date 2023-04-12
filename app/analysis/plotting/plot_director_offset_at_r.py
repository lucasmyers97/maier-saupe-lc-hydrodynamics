import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utilities import nematics as nu

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

linestyles = ['--', '-.', '-', ':']

def get_commandline_args():

    descrption = ('Plots deviation of director angle from two-defect isotropic '
                  'configuration as a function of polar angle at '
                  'different distances away from the core center. '
                  'Needs core structure from `get_points_around_defects`. '
                  'This expects Q-components to be arranged in an n x 5 '
                  'where n is n_r_points x n_theta_points')
    parser = argparse.ArgumentParser(description=descrption)

    # files
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='folder where core structure data is held')
    parser.add_argument('--structure_filename',
                        dest='structure_filename',
                        help='h5 file with structure data')
    parser.add_argument('--defect_positions',
                        dest='defect_positions',
                        default='defect_positions.h5',
                        help='h5 file with defect position data')
    parser.add_argument('--data_key',
                        dest='data_key',
                        help=('key in h5 file with the core data that will be '
                              'plotted. Note that this will be expected to '
                              'be of form `timestep_{}/something` where the '
                              'string will be later formatted with the '
                              'timesteps'))

    # simulation info
    parser.add_argument('--timestep',
                        dest='timestep',
                        type=int,
                        help='timestep number in h5 file to plot')
    parser.add_argument('--dt',
                        dest='dt',
                        type=float,
                        help='timestep size')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--core_structure', 
                       action='store_true',
                       help='denotes the analysis is for a core structure')
    group.add_argument('--outer_structure', 
                       action='store_false',
                       help=('denotes the analysis is for the outer ' 
                             'structure (away from defects)'))

    parser.add_argument('--defect_charge',
                        dest='defect_charge',
                        type=float,
                        help='charge of defect for core_structure choice')

    # plot specifications 
    parser.add_argument('--dists_from_center',
                        dest='dists_from_center',
                        nargs='+',
                        type=float,
                        help=('distances away from core at which to plot '
                              'director angle'))
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of file plot (must be png)')

    args = parser.parse_args()

    structure_filename = os.path.join(args.data_folder, 
                                      args.structure_filename)
    defect_positions = os.path.join(args.data_folder,
                                    args.defect_positions)
    plot_filename = os.path.join(args.data_folder, 
                                 args.plot_filename)

    return (structure_filename, defect_positions, args.data_key,
            args.timestep, args.dt,
            args.dists_from_center,
            plot_filename, args.core_structure, args.defect_charge)



def get_defect_distances(filename, defect_charge, time, dt):

    defect_file = h5py.File(filename, 'r')
    x = np.array(defect_file['x'][:])
    y = np.array(defect_file['y'][:])
    t = np.array(defect_file['t'][:])
    charge = np.array(defect_file['charge'][:])

    return nu.get_d_from_defect_positions(x, y, t, charge, defect_charge, time, dt)



def main():

    (structure_filename, defect_positions, data_key,
     time, dt,
     dists_from_center,
     plot_filename, core_structure, defect_charge) = get_commandline_args()

    d = get_defect_distances(defect_positions, defect_charge, time, dt)
    print("Distance between defects is: {}".format(d))

    file = h5py.File(structure_filename, 'r')
    Q_vec = file[data_key.format(time)]
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

    fig, ax = plt.subplots()
    fig_no_offset, ax_no_offset = plt.subplots()
    for (i, dist_from_center) in enumerate(dists_from_center):

        # find idx where r is closest to dist_from_center
        r_idx = np.argmin(np.abs(r - dist_from_center))

        phi_r_no_offset = nu.sanitize_director_angle(phi[r_idx, :])
        phi_r = nu.sanitize_director_angle(phi[r_idx, :])
        phi_offset = 0
        if core_structure:
            phi_offset = nu.pairwise_defect_director_near_defect(theta, 
                                                                 dist_from_center, 
                                                                 d, 
                                                                 defect_charge)
        else:
            phi_offset = nu.pairwise_defect_director_at_midpoint(theta, 
                                                                 dist_from_center, 
                                                                 d, 
                                                                 -0.5, 
                                                                 0.5)

        # gets proper isomorph
        phi_offset -= np.pi/2
        phi_offset = nu.sanitize_director_angle(phi_offset)


        ax.plot(theta, phi_r - phi_offset, label='r: {:.2f}'.format(r[r_idx]))
        ax_no_offset.plot(theta, phi_r_no_offset, 
                          linestyle=linestyles[i],
                          label='r: {:.2f}'.format(r[r_idx]))

    ax.set_xlabel(r'polar angle $\theta$')
    ax.set_ylabel(r'director angle $\phi$')
    ax.set_title(r'$\phi$ deviation from isolated, isotropic')
    ax.legend()

    ax_no_offset.set_xlabel(r'polar angle $\theta$')
    ax_no_offset.set_ylabel(r'director angle $\phi$')
    ax_no_offset.set_title(r'Director angle of defect')


    ax_no_offset.legend()
    fig_no_offset.tight_layout()

    plt.show()




if __name__ == '__main__':
    main()
