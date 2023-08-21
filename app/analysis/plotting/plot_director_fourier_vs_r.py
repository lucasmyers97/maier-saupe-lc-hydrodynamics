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
                        default='defect_positions.h5',
                        help='h5 file with defect position data')
    parser.add_argument('--defect_distance',
                        dest='defect_distance',
                        type=float,
                        help=('distance between defects if you are fixing the '
                              'defects in place'))
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
    parser.add_argument('--n_modes',
                        dest='n_modes',
                        type=int,
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

    parser.add_argument('--defect_charge',
                        dest='defect_charge',
                        type=float,
                        help='charge of defect for core_structure choice')

    args = parser.parse_args()

    structure_filename = os.path.join(args.data_folder, 
                                      args.structure_filename)
    defect_positions = os.path.join(args.data_folder,
                                    args.defect_positions)
    cos_plot_filename = os.path.join(args.data_folder, 
                                     args.cos_plot_filename)
    sin_plot_filename = os.path.join(args.data_folder, 
                                     args.sin_plot_filename)

    return (structure_filename, defect_positions, args.data_key,
            args.timestep, args.dt,
            args.n_modes,
            cos_plot_filename, sin_plot_filename, args.core_structure,
            args.defect_charge, args.defect_distance)



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
     n_modes,
     cos_plot_filename, sin_plot_filename,
     core_structure, defect_charge, defect_distance) = get_commandline_args()

    file = h5py.File(structure_filename, 'r')
    d = None
    if (defect_distance):
        d = defect_distance
    else:
        d = get_defect_distances(defect_positions, defect_charge, time, dt)
    print(d)

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

    An_r = np.zeros((n_r, n_modes))
    Bn_r = np.zeros((n_r, n_modes))
    for i in range(n_r):
        phi_offset = np.zeros(phi[i, :].shape)
        if core_structure:
            phi_offset = nu.pairwise_defect_director_near_defect(theta, 
                                                                 r[i], 
                                                                 d, 
                                                                 defect_charge)
        else:
            phi_offset = nu.pairwise_defect_director_at_midpoint(theta, 
                                                                 r[i], 
                                                                 d, 
                                                                 -0.5, 
                                                                 0.5)
        # get correct isomorph
        phi_offset -= np.pi/2
        phi_r = nu.sanitize_director_angle(phi[i, :]) 
        phi_offset = nu.sanitize_director_angle(phi_offset)

        An_phi, Bn_phi = fourier.calculate_trigonometric_fourier_coefficients(phi_r - phi_offset)

        An_r[i, :] = An_phi[:n_modes]
        Bn_r[i, :] = Bn_phi[:n_modes]

    # regular plots
    fig_An, ax_An = plt.subplots()
    fig_Bn, ax_Bn = plt.subplots()

    x_axis = None
    x_label = None
    if core_structure:
        x_axis = r
        x_label = r'$r / \xi$'
    else:
        x_axis = 1/r
        x_label = r'$\xi / r$'

    for i in range(n_modes):
        ax_An.plot(x_axis, An_r[:, i], label=r'$n = {}$'.format(i))

    eps = 0.1
    for i in range(1, n_modes):
        ax_Bn.plot(x_axis, (1 / eps) * Bn_r[:, i], label=r'$n = {}$'.format(i))

    ax_An.set_title(r'$\theta_c$ Fourier, t = {}'.format(time*dt))
    ax_An.set_xlabel(x_label)
    ax_An.set_ylabel(r'$\cos$ Fourier coeffs')
    ax_An.legend()

    ax_Bn.set_title(r'$\theta_c$ Fourier, t = {}'.format(time*dt))
    ax_Bn.set_xlabel(x_label)
    ax_Bn.set_ylabel(r'$\sin$ Fourier coeffs')
    # ax_Bn.set_xlim(0.0005499999999999999, 0.010450000000000001)
    # ax_Bn.set_ylim(-0.009422038905869975, 0.010011987892399009)
    ax_Bn.legend()

    print("Axes are: {}, {}", ax_Bn.get_xlim(), ax_Bn.get_ylim())

    fig_An.tight_layout()
    fig_An.savefig(cos_plot_filename.format(time))

    fig_Bn.tight_layout()
    fig_Bn.savefig(sin_plot_filename.format(time))

    # log plots
    fig_An_log, ax_An_log = plt.subplots()
    fig_Bn_log, ax_Bn_log = plt.subplots()

    for i in range(n_modes):
        ax_An_log.plot(np.log(r), np.log(An_r[:, i]), label=r'$n = {}$'.format(i))

    for i in range(1, n_modes):
        ax_Bn_log.plot(np.log(r), np.log(Bn_r[:, i]), label=r'$n = {}$'.format(i))

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
