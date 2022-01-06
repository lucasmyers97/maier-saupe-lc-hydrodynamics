import numpy as np
import h5py
import scipy.interpolate as interp
import argparse
import os


if __name__ == "__main__":

    description = "Find defect centers of two configurations, realign if they are not in the same place"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cody_folder', dest='cody_folder',
                        help='folder where data from cody is stored')
    parser.add_argument('--cody_filename', dest='cody_filename',
                        help='name of data file from cody')
    parser.add_argument('--lucas_folder', dest='lucas_folder',
                        help='folder where data from lucas is stored')
    parser.add_argument('--lucas_filename', dest='lucas_filename',
                        help='name of data file from lucas')
    args = parser.parse_args()

    cody_filename = os.path.join(args.cody_folder, args.cody_filename)
    lucas_filename = os.path.join(args.lucas_folder, args.lucas_filename)

    with h5py.File(cody_filename) as f:
        Q = f['Q'][:]
        points = f['points'][:]

    points = points.transpose()

    with h5py.File(lucas_filename) as f:
        X = f['X'][:]
        Y = f['Y'][:]

    m, n = X.shape
    grid_points = np.zeros((m*n, 2))
    for i in range(m):
        for j in range(n):
            grid_points[i*n + j][0] = X[i, j]
            grid_points[i*n + j][1] = Y[i, j]

    Q1_list = interp.griddata(points, Q[:, 0, 0], grid_points)
    Q2_list = interp.griddata(points, Q[:, 0, 1], grid_points)
    Q3_list = interp.griddata(points, Q[:, 0, 2], grid_points)
    Q4_list = interp.griddata(points, Q[:, 1, 1], grid_points)
    Q5_list = interp.griddata(points, Q[:, 1, 2], grid_points)

    Q1 = np.zeros(X.shape)
    Q2 = np.zeros(X.shape)
    Q3 = np.zeros(X.shape)
    Q4 = np.zeros(X.shape)
    Q5 = np.zeros(X.shape)

    for i in range(m):
        for j in range(n):
            Q1[i, j] = Q1_list[i*n + j]
            Q2[i, j] = Q2_list[i*n + j]
            Q3[i, j] = Q3_list[i*n + j]
            Q4[i, j] = Q4_list[i*n + j]
            Q5[i, j] = Q5_list[i*n + j]

    with h5py.File(cody_filename, 'w') as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Q1", data=Q1)
        f.create_dataset("Q2", data=Q2)
        f.create_dataset("Q3", data=Q3)
        f.create_dataset("Q4", data=Q4)
        f.create_dataset("Q5", data=Q5)
