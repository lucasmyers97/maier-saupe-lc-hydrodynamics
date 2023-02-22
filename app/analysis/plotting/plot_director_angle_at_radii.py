import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utilities import nematics as nu

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

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
    parser.add_argument('--structure_filename',
                        dest='structure_filename',
                        help='h5 file with structure data')
    parser.add_argument('--defect_positions',
                        dest='defect_positions',
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
    parser.add_argument('--r0',
                        dest='r0',
                        type=float,
                        help='inner distance from defect core')
    parser.add_argument('--rf',
                        dest='rf',
                        type=float,
                        help='outer distance from defect core')
    parser.add_argument('--n_r',
                        dest='n_r',
                        type=int,
                        help='number of points in the radial direction')
    parser.add_argument('--n_theta',
                        dest='n_theta',
                        type=int,
                        help='number of points in the radial direction')
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



def get_director_offset(core_structure, theta, defect_charge, r=1, d=1):

    if (core_structure):
        return defect_charge * theta
    else:
        theta1 = np.arcsin(r * np.sin(theta) / 
                           np.sqrt(r**2 + d**2 / 4 + r * d * np.cos(theta)))
        theta2 = np.arcsin(r * np.sin(theta) /
                           np.sqrt(r**2 + d**2 / 4 - r * d * np.cos(theta)))
        # return 0.5 * theta1 - 0.5 * theta2
        return 0



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
 
    d = get_d_from_defect_positions(defect_positions, time, dt)
    print("Distance between defects is: {}".format(d))

    fig, ax = plt.subplots()
    cos_fig, cos_ax = plt.subplots()
    sin_fig, sin_ax = plt.subplots()
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

    for dist_from_center in dists_from_center:
        # find idx where r is closest to dist_from_center
        r_idx = np.argmin(np.abs(r - dist_from_center))
        phi_r = (nu.sanitize_director_angle(phi[r_idx, :]) 
                 - get_director_offset(core_structure, theta, defect_charge, dist_from_center, d))
        phi_r -= np.mean(phi_r)

        FT_phi = np.fft.rfft(phi_r)
        N = phi_r.shape[0]
        An_phi = FT_phi.real / N
        Bn_phi = FT_phi.imag / N

        ax.plot(theta, phi_r, label='r: {}'.format(dist_from_center))
        cos_ax.plot(An_phi, label='r: {}'.format(dist_from_center))
        sin_ax.plot(Bn_phi, label='r: {}'.format(dist_from_center))

        print('r = {}'.format(dist_from_center))
        print('sin(theta) coeff: {}'.format(Bn_phi[1]))
        print('sin(2 theta) coeff: {}'.format(Bn_phi[2]))
        print('sin(3 theta) coeff: {}'.format(Bn_phi[3]))
        print('sin(4 theta) coeff: {}'.format(Bn_phi[4]))
        print()

    ax.set_xlabel(r'polar angle $\theta$')
    ax.set_ylabel(r'director angle $\phi$')
    ax.set_title(r'$\phi$ deviation from isolated, isotropic')
    ax.legend()

    cos_ax.set_xlabel(r'Fourier mode')
    cos_ax.set_ylabel(r'Amplitude')
    cos_ax.set_title(r'$\cos$ Fourier mode for $\phi$ deviation')
    cos_ax.set_xlim(-1, 6)
    cos_ax.legend()

    sin_ax.set_xlabel(r'Fourier mode')
    sin_ax.set_ylabel(r'Amplitude')
    sin_ax.set_title(r'$\sin$ Fourier mode for $\phi$ deviation')
    sin_ax.set_xlim(-1, 6)
    sin_ax.legend()

    fig.tight_layout()
    fig.savefig(plot_filename)

    plt.show()




if __name__ == '__main__':
    main()
