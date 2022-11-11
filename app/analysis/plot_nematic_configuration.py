import argparse
import os

import h5py
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import utilities.nematics as nu

def get_commandline_args():

    description = ("Given .h5 file with Q, x, and y data plot S-value and director to png")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where Q-configuration data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--configuration_filename',
                        dest='configuration_filename',
                        help='name of h5 file holding Q-configuration')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='output png filename')
    parser.add_argument('--n_pts',
                        type=int,
                        dest='n_pts',
                        help='number of points')

    args = parser.parse_args()

    configuration_filename = os.path.join(args.data_folder,
                                          args.configuration_filename)
    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    output_filename = os.path.join(output_folder, args.output_filename)

    return configuration_filename, output_filename, args.n_pts



def main():

    configuration_filename, output_filename, n_pts = get_commandline_args()
    file = h5py.File(configuration_filename)
    Q = np.array(file['Q'][:])
    x = np.array(file['x'][:])[0, :]
    y = np.array(file['y'][:])[0, :]

    print(x.shape)

    S, P, n, m = nu.eigensystem_from_Q(Q[0, :, :, :])

    x_grid = np.linspace(np.min(x), np.max(x), num=n_pts)
    y_grid = np.linspace(np.min(y), np.max(y), num=n_pts)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    S_grid = interp.griddata((x, y), S, (X.flatten(), Y.flatten()))
    S_grid = S_grid.reshape(X.shape)

    plt.pcolor(X, Y, S_grid)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
