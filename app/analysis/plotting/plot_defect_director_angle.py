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

    descrption = ('Plots director angle as a function of polar angle given '
                  'core structure from `get_points_around_defects`. This '
                  'expects Q-components to be arranged in an n x 5 '
                  'where n is n_r_points x n_theta_points')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='folder where core structure data is held')
    parser.add_argument('--core_structure_filename',
                        dest='core_structure_filename',
                        help='h5 file with core structure data')
    parser.add_argument('--data_key',
                        dest='data_key',
                        help=('key in h5 file with the core data that will be '
                              'plotted. Note that this will be expected to '
                              'be of form `timestep_{}/something` where the '
                              'string will be later formatted with the '
                              'timesteps'))
    parser.add_argument('--timesteps',
                        dest='timesteps',
                        nargs='+',
                        type=int,
                        help='names of timesteps in h5 file to plot')
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
    parser.add_argument('--dist_from_center',
                        dest='dist_from_center',
                        type=float,
                        help=('distance away from core at which to plot '
                              'director angle'))
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of file plot (must be png)')
    args = parser.parse_args()

    core_structure_filename = os.path.join(args.data_folder, 
                                           args.core_structure_filename)
    plot_filename = os.path.join(args.data_folder, 
                                 args.plot_filename)

    return (core_structure_filename, args.data_key,
            args.timesteps, args.dt,
            args.r0, args.rf, args.n_r, args.n_theta,
            args.dist_from_center,
            plot_filename)



def main():

    (core_structure_filename, data_key,
     timesteps, dt,
     r0, rf, n_r, n_theta,
     dist_from_center,
     plot_filename) = get_commandline_args()

    file = h5py.File(core_structure_filename, 'r')

    fig, ax = plt.subplots()
    cos_fig, cos_ax = plt.subplots()
    sin_fig, sin_ax = plt.subplots()
    for time in timesteps:
        Q_vec_data = np.array(file[data_key.format(time)][:])
        Q_data = nu.Q_vec_to_mat(Q_vec_data)

        _, _, n, _ = nu.eigensystem_from_Q(Q_data)

        r = np.linspace(r0, rf, num=n_r)
        theta = np.linspace(0, 2*np.pi, num=n_theta)

        n_x = n[:, 0].reshape((n_r, n_theta))
        n_y = n[:, 1].reshape((n_r, n_theta))
        phi = np.arctan2(n_y, n_x)

        # find idx where r is closest to dist_from_center
        r_idx = np.argmin(np.abs(r - dist_from_center))
        phi_r = nu.sanitize_director_angle(phi[r_idx, :]) - 0.5 * theta
        phi_r -= np.mean(phi_r)

        FT_phi = np.fft.rfft(phi_r)
        N = phi_r.shape[0]
        An_phi = FT_phi.real / N
        Bn_phi = FT_phi.imag / N

        ax.plot(theta, phi_r, label='time: {}'.format(time*dt))
        cos_ax.plot(An_phi, label='time: {}'.format(time*dt))
        sin_ax.plot(Bn_phi, label='time: {}'.format(time*dt))

    ax.set_xlabel(r'polar angle $\theta$')
    ax.set_ylabel(r'director angle $\phi$')
    ax.set_title(r'$\phi$ deviation from isolated, isotropic')
    ax.legend()

    cos_ax.set_xlabel(r'Fourier mode')
    cos_ax.set_ylabel(r'Amplitude')
    cos_ax.set_title(r'$\cos$ Fourier mode for $\phi$ deviation')
    cos_ax.legend()

    sin_ax.set_xlabel(r'Fourier mode')
    sin_ax.set_ylabel(r'Amplitude')
    sin_ax.set_title(r'$\sin$ Fourier mode for $\phi$ deviation')
    sin_ax.legend()

    fig.tight_layout()
    fig.savefig(plot_filename)

    plt.show()




if __name__ == '__main__':
    main()
