import paraview.simple as ps
import paraview.servermanager as psm

import time

import numpy as np
from mpi4py import MPI

start = time.time()

filename = ("/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/"
            "temp-data/supercomputer-one-defect-dzyaloshinskii/"
            "single-defect-core-eps-0/"
            "Q_components_dzyaloshinskii_single_defect_49.pvtu")

# Read in raw data
Q_configuration = ps.XMLPartitionedUnstructuredGridReader(FileName=filename)

# Calculate eigenvectors and eigenvalues
programmable_filter = ps.ProgrammableFilter(Input=Q_configuration)
programmable_filter.Script = """
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

Q0 = inputs[0].PointData[\'Q0\']
Q1 = inputs[0].PointData[\'Q1\']
Q2 = inputs[0].PointData[\'Q2\']
Q3 = inputs[0].PointData[\'Q3\']
Q4 = inputs[0].PointData[\'Q4\']

Q_mat = np.zeros((3, 3, Q0.shape[0]))

Q_mat[0, 0, :] = Q0
Q_mat[0, 1, :] = Q1
Q_mat[0, 2, :] = Q2
Q_mat[1, 1, :] = Q3
Q_mat[1, 2, :] = Q4
Q_mat[1, 0, :] = Q_mat[0, 1, :]
Q_mat[2, 0, :] = Q_mat[0, 2, :]
Q_mat[2, 1, :] = Q_mat[1, 2, :]

S = np.zeros(Q0.shape)
P = np.zeros(Q0.shape)
n = np.zeros((Q0.shape[0], 3))
m = np.zeros((Q0.shape[0], 3))

for i in range(S.shape[0]):
    w, v = np.linalg.eig(Q_mat[:, :, i])
    w_idx = np.argsort(w)
    S[i] = w[w_idx[-1]]
    P[i] = w[w_idx[-2]]
    n[i, :] = v[:, w_idx[-1]]
    m[i, :] = v[:, w_idx[-2]]

output.PointData.append(S, "S")
output.PointData.append(P, "P")
output.PointData.append(n, "n")
output.PointData.append(m, "m")
"""

# Generate sample points
n = 1000
m = 100
r0 = 0.025
rmax = 2.5
theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
r = np.linspace(r0, rmax, m)

R, Theta = np.meshgrid(r, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = np.zeros(X.shape)

points = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))

poly_points = ps.PolyPointSource()
poly_points.Points = points.transpose().flatten()

# Query configuration at sample points
resampled_data = ps.ResampleWithDataset(SourceDataArrays=programmable_filter,
                                        DestinationMesh=poly_points)

# Write data to HDF5 file
hdf5_filter = ps.ProgrammableFilter(Input=resampled_data)
hdf5_filter.Script = """
import numpy as np

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD

S = inputs[0].PointData["S"]
P = inputs[0].PointData["P"]
m = inputs[0].PointData["m"]
n = inputs[0].PointData["n"]

poinrt = inputs[0].GetPoints()

num = np.array(S.shape[0], dtype='i')
num_g = np.zeros(comm.Get_size(), dtype='i')
comm.Allgather([num, MPI.INT],
                [num_g, MPI.INT])
assert np.sum(num_g) == num_g[0]

if comm.Get_rank() == 0:
    with h5py.File("single_defect_core.h5", "w") as f:
        f.create_dataset("S", data=S)
        f.create_dataset("P", data=P)
        f.create_dataset("n", data=n)
        f.create_dataset("m", data=m)
        f.create_dataset("points", data=points)

# total_points = np.sum(num_g)
# this_address = 0
# rank = comm.Get_rank()
# for i in range(rank):
#     this_address += num_g[i]
# 
# f = h5py.File("parallel_test.h5", "w", driver="mpio", comm=comm)
# dset = f.create_dataset('test', total_points)
# dset[this_address:(this_address + num_g[rank])] = S
# f.close()
"""

# local_resampled_data = psm.Fetch(resampled_data)
# 
# print(local_resampled_data.GetPointData())
# print(local_resampled_data.GetPointData())

# ps.Show(resampled_data)

# Show hdf5 filter so that it actually executes
ps.Show(hdf5_filter)
ps.Render()
active_view = ps.GetActiveView()
ps.SaveScreenshot("scratch_test.png", active_view)

end = time.time()
print(end - start)
