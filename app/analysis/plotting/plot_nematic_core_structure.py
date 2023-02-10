import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utilities import nematics as nu

def get_commandline_args():

    descrption = ('Plots director as quiver and S-value as colormap given '
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
                              'plotted'))
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

    parser.add_argument('--n_r_sparse',
                        dest='n_r_sparse',
                        type=int,
                        help=('number of director arrows in the radial '
                              'direction'))
    parser.add_argument('--n_theta_sparse',
                        dest='n_theta_sparse',
                        type=int,
                        help=('number of director arrows in the polar '
                              'direction'))
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='name of file plot (must be png)')
    args = parser.parse_args()

    core_structure_filename = os.path.join(args.data_folder, 
                                           args.core_structure_filename)
    plot_filename = os.path.join(args.data_folder, 
                                 args.plot_filename)

    return (core_structure_filename, args.data_key,
            args.r0, args.rf, args.n_r, args.n_theta,
            args.n_r_sparse, args.n_theta_sparse,
            plot_filename)



def main():

    (core_structure_filename, data_key,
     r0, rf, n_r, n_theta,
     n_r_sparse, n_theta_sparse,
     plot_filename) = get_commandline_args()

    file = h5py.File(core_structure_filename, 'r')

    Q_vec_data = np.array(file[data_key][:])
    Q_data = nu.Q_vec_to_mat(Q_vec_data)

    S, P, n, m = nu.eigensystem_from_Q(Q_data)

    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    S_grid = S.reshape((n_r, n_theta))
    n_x = n[:, 0].reshape((n_r, n_theta))
    n_y = n[:, 1].reshape((n_r, n_theta))

    sparse_idx_r = np.arange(0, n_r, np.rint(n_r / n_r_sparse), dtype=np.int64)
    sparse_idx_theta = np.arange(0, n_theta, np.rint(n_theta/ n_theta_sparse), dtype=np.int64)

    sparse_idx = np.ix_(sparse_idx_r, sparse_idx_theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.pcolormesh(Theta, R, S_grid, shading='gourad')
    ax.quiver(Theta[sparse_idx], R[sparse_idx], 
              n_x[sparse_idx], n_y[sparse_idx],
              headwidth=0, headaxislength=0, headlength=0)
    plt.show()




if __name__ == '__main__':
    main()
