import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .utilities import nematics as nu

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'lines.linewidth': 2})

linestyles = ['--', '-.', ':']

dim = 3

def get_commandline_args():

    descrption = ('Calculates theta_c from Q-tensor at many points. '
                  'Does this by rotating the Q-tensor by a rotation matrix '
                  'representing the isotropic solution at every point, and then '
                  'calculating the eigenvectors, and thereby the angle.'
                  'This expects Q-components to be arranged in an n x 5 '
                  'where n is n_r_points x n_theta_points')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--data_folder',
                        help='folder where core structure data is held')
    parser.add_argument('--input_filename',
                        help='h5 file with structure data')
    parser.add_argument('--data_key',
                        help='key in h5 file with the core data')
    parser.add_argument('--defect_centers',
                        nargs=4,
                        type=float,
                        help=('defect center coordinates, arranged as x1 y1 x2 y2'))
    parser.add_argument('--defect_charges',
                        nargs=2,
                        type=float,
                        help='charge of defects, in order q1 q2')

    parser.add_argument('--output_filename',
                        help='name of h5 output file which will contain theta_c')

    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    output_filename = os.path.join(args.data_folder, args.output_filename)

    defect_centers = [[args.defect_centers[0], args.defect_centers[1]],
                      [args.defect_centers[2], args.defect_centers[3]]]

    return (input_filename, args.data_key, defect_centers, args.defect_charges,
            output_filename)



def get_rotation_matrices(defect_charges, defect_centers, X, Y):
    """
    Given an x-y meshgrid, calculates a list of rotation matrices which will be 
    applied at each point
    """

    x_coords = X.flatten()
    y_coords = Y.flatten()
    R = np.zeros((x_coords.shape[0], dim, dim))

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):

        phi1 = np.arctan2(y - defect_centers[0][1], x - defect_centers[0][0])
        phi2 = np.arctan2(y - defect_centers[1][1], x - defect_centers[1][0])
        theta_iso = defect_charges[0] * phi1 + defect_charges[1] * phi2 + np.pi / 2

        R[i, :, :] = np.array([[np.cos(theta_iso), -np.sin(theta_iso), 0],
                               [np.sin(theta_iso), np.cos(theta_iso), 0],
                               [0, 0, 1]])

    return R




def main():

    (input_filename, data_key, defect_centers, defect_charges,
     output_filename) = get_commandline_args()

    file = h5py.File(input_filename, 'r')
 
    data = file[data_key]

    r0 = data.attrs['r0']
    rf = data.attrs['rf']
    n_r = data.attrs['n_r']
    n_theta = data.attrs['n_theta']

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Rot = get_rotation_matrices(defect_charges, defect_centers, X, Y)

    Q_vec = np.array(data[:])
    # Q_mat = nu.Q_vec_to_mat(Q_vec)

    # for i in range(Q_mat.shape[0]):

    #     Q_mat[i, :, :] = Rot[i, :, :].transpose() @ Q_mat[i, :, :] @ Rot[i, :, :]

    # q1, q2, n, m = nu.eigensystem_from_Q(Q_mat)

    # n_x = n[:, 0].reshape((n_r, n_theta))
    # n_y = n[:, 1].reshape((n_r, n_theta))
    # theta_c = np.arctan2(n_y, n_x)

    # print(Theta.shape)
    # print(R.shape)
    # print(theta_c.shape)

    Q2 = Q_vec[:, 1].reshape((n_r, n_theta))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # pcm = ax.pcolormesh(Theta, R, theta_c)
    pcm = ax.pcolormesh(Theta, R, Q2)
    fig.colorbar(pcm, ax=ax)
    plt.show()

    # for (i, dist_from_center) in enumerate(dists_from_center):

    #     # find idx where r is closest to dist_from_center
    #     r_idx = np.argmin(np.abs(r - dist_from_center))

    #     phi_r = nu.sanitize_director_angle(phi[r_idx, :])
    #     phi_r -= np.min(phi_r)

    #     ax.plot(theta[half_interval], phi_r[half_interval], linestyle=linestyles[i], label='r: {}'.format(dist_from_center))

    # ax.set_xlabel(r'$\varphi$')
    # ax.set_ylabel(r'$\theta$')

    # if defect_charge == 0.5:
    #     ax.xaxis.set_ticks([0, np.pi / 2, np.pi],
    #                        labels=[r'$0$', r'$\pi / 2$', r'$\pi$'])
    #     ax.yaxis.set_ticks([0, np.pi / 4, np.pi / 2],
    #                        labels=[r'$0$', r'$\pi / 4$', r'$\pi / 2$'])
    #     ax.legend(loc='lower right')
    # elif defect_charge == -0.5:
    #     ax.xaxis.set_ticks([0, np.pi / 2, np.pi],
    #                        labels=[r'$0$', r'$\pi / 2$', r'$\pi$'])
    #     ax.yaxis.set_ticks([np.pi / 2, 3 * np.pi / 4, np.pi],
    #                        labels=[r'$\pi / 2$', r'$3 \pi / 4$', r'$\pi$'])
    #     ax.legend(loc='lower left')

    # fig.tight_layout()

    # fig.savefig(plot_filename)

    # plt.show()




if __name__ == '__main__':
    main()
