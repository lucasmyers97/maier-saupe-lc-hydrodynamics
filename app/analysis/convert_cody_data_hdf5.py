"""
This script reads in a .mat file from Cody which has 2D Q-tensor data on the
steady state solution of a liquid crystal system governed by a Maier-Saupe free
energy and isotropic elasticity. The initial configuration is a perfect -1/2
defect. This script reads it in and then converts the data to an hdf5.
"""


import sys
import scipy.io as sio
import h5py
import numpy as np

arguments = sys.argv
input_filename = str(arguments[1])
output_filename = str(arguments[2])

data = sio.loadmat(input_filename)
Q = data['Q']
X = data['X']
Y = data['Y']

with h5py.File(output_filename, "w") as f:
    dset = f.create_dataset("Q1", Q[0, 0].shape, dtype=np.double, data=Q[0, 0])
    dset = f.create_dataset("Q2", Q[0, 1].shape, dtype=np.double, data=Q[0, 1])
    dset = f.create_dataset("Q3", Q[0, 2].shape, dtype=np.double, data=Q[0, 2])
    dset = f.create_dataset("Q4", Q[1, 1].shape, dtype=np.double, data=Q[1, 1])
    dset = f.create_dataset("Q5", Q[1, 2].shape, dtype=np.double, data=Q[1, 2])
    
    dset = f.create_dataset("X", X.shape, dtype=np.double, data=X)
    dset = f.create_dataset("Y", Y.shape, dtype=np.double, data=Y)