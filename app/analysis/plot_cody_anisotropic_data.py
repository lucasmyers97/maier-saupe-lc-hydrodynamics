"""
This script reads Cody's data from an anisotropic defect, and then plots
the director and S-valueself.

The input data is an hdf5 file which has two 1D arrays, x and y, corresponding
to the x- and y-components of the support points, as well as a 3x3xn Q-array
corresponding to the Q-tensor values at each of the support points.
They are named `x`, `y`, and `Q` respectively.
"""

import argparse
import os
import h5py
import numpy as np

import matplotlib.pyplot as plt

import comparing_to_cody_data as script

from scipy.interpolate import griddata

if __name__ == "__main__":

    description = "Plots cody's anisotropic data"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cody_folder', dest='cody_folder',
                        help='folder where data from cody is stored')
    parser.add_argument('--cody_filename', dest='cody_filename',
                        help='name of data file from cody')
    args = parser.parse_args() 

    filename = os.path.join(args.cody_folder, args.cody_filename)

    file = h5py.File(filename)
    x = np.array(file['x'][:])
    y = np.array(file['y'][:])
    Q = np.array(file['Q'][:])
    
    points = np.vstack((x, y)).transpose()

    num = 256

    x_lims = (np.min(x), np.max(x))
    y_lims = (np.min(y), np.max(y))
    x = np.linspace(x_lims[0], x_lims[1], num=num)
    y = np.linspace(y_lims[0], y_lims[1], num=num)
    X, Y = np.meshgrid(x, y)

    new_points = np.zeros((num*num, 2))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            new_points[num*i + j, 0] = X[i, j]
            new_points[num*i + j, 1] = Y[i, j]

    grid_Q = np.zeros((3, 3, num*num))
    for i in range(3):
        for j in range(3):
            grid_Q[i, j, :] = griddata(points, Q[:, i, j], new_points)

    new_grid_Q = np.zeros((3, 3, num, num))
    for i in range(3):
        for j in range(3):
            for k in range(X.shape[0]):
                for l in range(X.shape[1]):
                    new_grid_Q[i, j, k, l] = grid_Q[i, j, num*k + l]

    n, S = script.calcDirectorAndS(new_grid_Q)
    fig, ax, q = script.plotDirectorAndS(X, Y, n, S)
    plt.show()
