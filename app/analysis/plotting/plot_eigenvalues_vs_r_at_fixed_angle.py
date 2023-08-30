import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from ..utilities import nematics as nu

plt.style.use('science')

mpl.rcParams['figure.dpi'] = 300

linestyles = ['-', ':', '--', '-.']

def get_commandline_args():
    

    description = ('Get core structure eigenvalues at fixed angle with points from '
                   '`get_points_around_defect` program, plot vs r')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--input_filename',
                        dest='input_filename',
                        help='input hdf5 filename containing cores structure')
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='filename of output plot')
    parser.add_argument('--fixed_angles',
                        dest='fixed_angles',
                        nargs='+',
                        default=[0.0],
                        type=float,
                        help='angles at which to plot radial eigenvalues')
    parser.add_argument('--data_key',
                        dest='data_key',
                        help='key name of timestep in hdf5 file')
    parser.add_argument('--equilibrium_S',
                        dest='equilibrium_S',
                        type=float,
                        help='equilibrium S-value which will be plotted as a line')
    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    plot_filename = None
    if args.output_folder:
        plot_filename = os.path.join(args.output_folder, args.plot_filename)
    else:
        plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return (input_filename, plot_filename, args.fixed_angles, args.data_key, 
            args.equilibrium_S)



def main():

    (input_filename, plot_filename, fixed_angles, data_key, 
     equilibrium_S) = get_commandline_args()

    file = h5py.File(input_filename)
    data = file[data_key]

    r0 = data.attrs['r0']
    rf = data.attrs['rf']
    n_r = data.attrs['n_r']
    n_theta = data.attrs['n_theta']
    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    Q_vec = np.array(data[:])
    Q_mat = nu.Q_vec_to_mat(Q_vec)
    q1, q2, _, _ = nu.eigensystem_from_Q(Q_mat)

    q1 = q1.reshape((n_r, n_theta))
    q2 = q2.reshape((n_r, n_theta))

    theta_idxs = [np.argmin(np.abs(theta - fixed_angle)) for fixed_angle in fixed_angles]
    print(theta_idxs)

    fig, ax = plt.subplots()
    for i in range(len(theta_idxs)):

        ax.plot(r, q1[:, theta_idxs[i]], 
                label=r'$q_1, \varphi = {:.3f}$'.format(theta[theta_idxs[i]]),
                linestyle=linestyles[i])
        ax.plot(r, q2[:, theta_idxs[i]], 
                label=r'$q_2, \varphi = {:.3f}$'.format(theta[theta_idxs[i]]),
                linestyle=linestyles[i])
        ax.set_xlabel(r'$r / \xi$')
        ax.set_title(r'Eigenvalues at angles')

    max_eigenvalue = (2/3) * equilibrium_S
    x_lims = ax.get_xlim()
    ax.hlines(max_eigenvalue, x_lims[0], x_lims[1], label=r'equilibrium $q_1$', linestyle='--')
    ax.hlines(-0.5 * max_eigenvalue, x_lims[0], x_lims[1], label=r'equilibrium $q_2$', linestyle='--')

    fig.legend()
    fig.savefig(plot_filename)

    plt.show()


if __name__ == "__main__":

    main()
