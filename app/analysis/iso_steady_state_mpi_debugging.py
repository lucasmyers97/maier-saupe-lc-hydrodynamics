import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import os
import h5py

def get_filename(step, rank, num_ranks):

    filename = "assemble_system_output_{:d}_{:d}_{:d}.h5"
    return filename.format(step, rank, num_ranks)

def get_data_from_file(filename):

    file = h5py.File(filename)

    points = file['points'][:]
    Q_vals = file['Q_vals'][:]

    return points, Q_vals

def get_gridded_data(points, Q_vals):

    n, m = Q_vals.shape
    num = 1000
    Q = np.zeros((num, num, 5))

    x_endpoints = (np.min(points[:, 0]), np.max(points[:, 0]))
    y_endpoints = (np.min(points[:, 1]), np.max(points[:, 1]))
    x = np.linspace(x_endpoints[0], x_endpoints[1], num=num)
    y = np.linspace(y_endpoints[0], y_endpoints[1], num=num)
    X, Y = np.meshgrid(x, y)

    for i in range(m):
        Q[:, :, i] = scipy.interpolate.griddata(points, Q_vals[:, i], (X, Y))

    return X, Y, Q

def combine_ranks(points_tuple, Q_vals_tuple):

    points = np.concatenate(points_tuple, axis=0)
    Q_vals = np.concatenate(Q_vals_tuple, axis=0)

    return points, Q_vals

def read_domain_data(step, num_ranks, folder):

    point_array = []
    Q_val_array = []

    for rank in range(num_ranks):
        filename = os.path.join(folder, get_filename(step, rank, num_ranks))
        points, Q_vals = get_data_from_file(filename)
        nonzero_points = points[:, 0] != 0
        points = points[nonzero_points, :]
        Q_vals = Q_vals[nonzero_points, :]
        point_array.append(points)
        Q_val_array.append(Q_vals)

    point_tuple = tuple(point_array)
    Q_val_tuple = tuple(Q_val_array)

    points, Q_vals = combine_ranks(point_tuple, Q_val_tuple)
    return points, Q_vals

def compress_support_points_and_vals(support_points, support_point_vals):

    vec_dim = 5
    dim = 2
    n = support_point_vals.shape[0]
    n_pts = int(n / vec_dim)

    rhs_vals = np.zeros((n_pts, vec_dim))
    points = np.zeros((n_pts, dim))
    for i in range(n_pts):
        for j in range(vec_dim):
            rhs_vals[i, j] = support_point_vals[j + i*vec_dim]
        points[i, :] = support_points[i*vec_dim, :]

    return points, rhs_vals

def get_rhs_data_from_file(filename):

    file = h5py.File(filename)
    support_points = file['support_points'][:]
    support_point_vals = file['support_point_vals'][:]
    is_locally_owned = file['is_locally_owned'][:]

    is_locally_owned = is_locally_owned == 1
    support_points = support_points[is_locally_owned]
    support_point_vals = support_point_vals[is_locally_owned]

    points, rhs_vals = compress_support_points_and_vals(support_points,
                                                        support_point_vals)

    return points, rhs_vals

def get_rhs_domain_data(step, num_ranks, folder):

    support_point_array = []
    rhs_array = []

    for rank in range(num_ranks):
        filename = os.path.join(folder, get_filename(step, rank, num_ranks))
        points, rhs_vals = get_rhs_data_from_file(filename)
        support_point_array.append(points)
        rhs_array.append(rhs_vals)

    support_points, rhs_vals = combine_ranks(tuple(support_point_array),
                                             tuple(rhs_array))
    return support_points, rhs_vals

if __name__ == "__main__":

    folder = "/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/data/simulations/iso-steady-state-mpi/2022-01-24-03"

    step = 2
    rank = 0
    num_ranks = 1

    # points, Q_vals = read_domain_data(step, num_ranks, folder)
    # X, Y, Q1 = get_gridded_data(points, Q_vals)
    points, Q_vals = get_rhs_domain_data(step, num_ranks, folder)
    X, Y, Q1 = get_gridded_data(points, Q_vals)
    # X, Y, Q2 = get_gridded_data(points, Q_vals)

    num_ranks = 2

    # points, Q_vals = read_domain_data(step, num_ranks, folder)
    points, Q_vals = get_rhs_domain_data(step, num_ranks, folder)
    X, Y, Q2 = get_gridded_data(points, Q_vals)

    # for i in range(5):
    #     fig, ax = plt.subplots()
    #     c = ax.pcolormesh(X, Y, Q1[:, :, i], shading='auto')
    #     fig.colorbar(c, ax=ax)
    # for i in range(5):
    #     fig, ax = plt.subplots()
    #     c = ax.pcolormesh(X, Y, Q2[:, :, i], shading='auto')
    #     fig.colorbar(c, ax=ax)
    for i in range(5):
        fig, ax = plt.subplots()
        c = ax.pcolormesh(X, Y, Q1[:, :, i] - Q2[:, :, i], shading='auto')
        fig.colorbar(c, ax=ax)

    plt.show()
