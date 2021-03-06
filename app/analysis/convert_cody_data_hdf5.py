"""
This script reads in a .mat file from Cody which has 2D Q-tensor data on the
steady state solution of a liquid crystal system governed by a Maier-Saupe free
energy and isotropic elasticity. The initial configuration is a perfect -1/2
defect. This script reads it in and then converts the data to an hdf5.
"""


import sys
import argparse
import scipy.io as sio
import h5py
import numpy as np

def rotateQ(Q):

    # rotation matrix
    R = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])

    # rotate Q-tensor at each gridpoint
    Q_rot = np.zeros(Q.shape)
    for i in range(Q.shape[2]):
        for j in range(Q.shape[3]):
            Q_new = np.matmul(Q[:, :, i, j], R)
            Q_new = np.matmul(R.transpose(), Q_new)
            Q_rot[:, :, i, j] = Q_new
            
    # rotate Q-tensors around the grid
    for i in range(Q_rot.shape[0]):
        for j in range(Q_rot.shape[1]):
            Q_rot[i, j, :, :] = np.rot90(Q_rot[i, j, :, :], k=3)

    return Q_rot



if __name__ == "__main__":

    description = "Reads a .mat file from cody and exports data as a .h5 file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_filename', dest='input_filename',
                        help='input data file, should be a .mat')
    parser.add_argument('--output_filename', dest='output_filename',
                        help='output data file, should be a .h5')
    parser.add_argument('--num_rotations', dest='num_rotations',
                        help='number of 90 degree clockwise rotations of data')
    args = parser.parse_args()
    
    input_filename = args.input_filename
    output_filename = args.output_filename
    num_rotations = int(args.num_rotations)
    
    data = sio.loadmat(input_filename)
    with h5py.File(output_filename, "w") as f:

        # Sometimes we get degrees of freedom of Q-tensor -- in that case use
        # these first 3 if-statements
        if 'Q' in data.keys():

            Q = data['Q']
            for i in range(num_rotations):
                Q = rotateQ(Q)

            dset = f.create_dataset("Q1", Q[0, 0].shape, dtype=np.double, data=Q[0, 0])
            dset = f.create_dataset("Q2", Q[0, 1].shape, dtype=np.double, data=Q[0, 1])
            dset = f.create_dataset("Q3", Q[0, 2].shape, dtype=np.double, data=Q[0, 2])
            dset = f.create_dataset("Q4", Q[1, 1].shape, dtype=np.double, data=Q[1, 1])
            dset = f.create_dataset("Q5", Q[1, 2].shape, dtype=np.double, data=Q[1, 2])
        if 'X' in data.keys():
            X = data['X']
            dset = f.create_dataset("X", X.shape, dtype=np.double, data=X)
        if 'Y' in data.keys():
            Y = data['Y']
            dset = f.create_dataset("Y", Y.shape, dtype=np.double, data=Y)

        # If we get eta, mu, nu representation, use these if-statements
        if 'u' in data.keys():
            u = data['u']
            dset = f.create_dataset("eta", u[0].shape, dtype=np.double, data=u[0])
            dset = f.create_dataset("mu", u[1].shape, dtype=np.double, data=u[1])
            dset = f.create_dataset("nu", u[2].shape, dtype=np.double, data=u[2])
        if 'x' in data.keys():
            x = data['x']

            # x happens to be shape (1, 257) so we need to index first element
            X, Y = np.meshgrid(x[0], x[0], indexing='ij')
            dset = f.create_dataset("X", X.shape, dtype=np.double, data=X)
            dset = f.create_dataset("Y", Y.shape, dtype=np.double, data=Y)
