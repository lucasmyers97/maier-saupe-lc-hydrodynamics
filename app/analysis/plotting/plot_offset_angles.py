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
                  'Note that this just plots the expected isotropic angle')
    parser = argparse.ArgumentParser(description=descrption)
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
    parser.add_argument('--timestep',
                        dest='timestep',
                        type=int,
                        help='timestep number in h5 file to plot')
    parser.add_argument('--dt',
                        dest='dt',
                        type=float,
                        help='timestep size')
    parser.add_argument('--dists_from_center',
                        dest='dists_from_center',
                        nargs='+',
                        type=float,
                        help=('distances away from core at which to plot '
                              'director angle'))
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of file plot (must be png)')

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



def get_d_from_defect_positions(filename, time, dt):

    # get d at this time
    defect_file = h5py.File(filename, 'r')
    x = np.array(defect_file['x'][:])
    y = np.array(defect_file['y'][:])
    t = np.array(defect_file['t'][:])
    charge = np.array(defect_file['charge'][:])

    (pos_t, neg_t, 
     pos_centers, neg_centers) = nu.split_defect_centers_by_charge(charge, 
                                                                   t, 
                                                                   x, 
                                                                   y)
    pos_center = nu.match_times_to_points(np.array([time * dt]), pos_t, 
                                          pos_centers[:, 0], pos_centers[:, 1])
    neg_center = nu.match_times_to_points(np.array([time * dt]), neg_t, 
                                          neg_centers[:, 0], neg_centers[:, 1])

    return np.linalg.norm(pos_center[0, :] - neg_center[0, :])



def main():

    (structure_filename, defect_positions, data_key,
     time, dt,
     dists_from_center,
     plot_filename, core_structure, defect_charge) = get_commandline_args()

    file = h5py.File(structure_filename, 'r')

    defect_file = h5py.File(defect_positions, 'r')
    x = np.array(defect_file['x'][:])
    y = np.array(defect_file['y'][:])
    t = np.array(defect_file['t'][:])
    charge = np.array(defect_file['charge'][:])

    d = nu.get_d_from_defect_positions(x, y, t, charge, defect_charge, time, dt)
    print("Distance between defects is: {}".format(d))

    Q_vec = file[data_key.format(time)]
    r0 = Q_vec.attrs['r0']
    rf = Q_vec.attrs['rf']
    n_r = Q_vec.attrs['n_r']
    n_theta = Q_vec.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)


    fig, ax = plt.subplots()
    fig_no_offset, ax_no_offset = plt.subplots()
    for (i, dist_from_center) in enumerate(dists_from_center):

        phi = None
        if core_structure:
            phi = nu.pairwise_defect_director_near_defect(theta, 
                                                          dist_from_center, 
                                                          d, 
                                                          defect_charge)
        else:
            phi = nu.pairwise_defect_director_at_midpoint(theta, 
                                                          dist_from_center, 
                                                          d, 
                                                          -0.5, 
                                                          0.5)
        phi_no_offset = nu.sanitize_director_angle(phi)
        phi_no_offset += np.pi/2 # gets proper isomorph

        ax.plot(theta, phi, label='r: {}'.format(dist_from_center))
        ax_no_offset.plot(theta, phi_no_offset, 
                          linestyle=linestyles[i],
                          label='r: {}'.format(dist_from_center))

    ax.set_xlabel(r'polar angle $\theta$')
    ax.set_ylabel(r'director angle $\phi$')
    ax.set_title(r'$\phi$ deviation from isolated, isotropic')
    ax.legend()

    ax_no_offset.set_xlabel(r'polar angle $\theta$')
    ax_no_offset.set_ylabel(r'director angle $\phi$')
    ax_no_offset.set_title(r'Director angle of defect')

    # ax_no_offset.legend()
    fig_no_offset.tight_layout()

    plt.show()




if __name__ == '__main__':
    main()
