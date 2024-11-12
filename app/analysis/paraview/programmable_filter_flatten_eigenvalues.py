# Calculate eigenvalues 
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

S = 0.6751
P = 0

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

Q_mat = np.zeros((3, 3, Q0.shape[0]))

Q_mat[0, 0, :] = Q0
Q_mat[0, 1, :] = Q1
Q_mat[0, 2, :] = Q2
Q_mat[1, 1, :] = Q3
Q_mat[1, 2, :] = Q4
Q_mat[1, 0, :] = Q_mat[0, 1, :]
Q_mat[2, 0, :] = Q_mat[0, 2, :]
Q_mat[2, 1, :] = Q_mat[1, 2, :]
Q_mat[2, 2, :] = -(Q0 + Q3)

for i in range(Q0.shape[0]):
    w, v = np.linalg.eigh(Q_mat[:, :, i])
    w[-1] = S * (2.0 / 3.0)
    w[-2] = P - 0.5 * w[-1]
    w[0] = -(w[-1] + w[-2])

    Q_mat[:, :, i] = v @ np.diag(w) @ v.transpose()

Q0 = Q_mat[0, 0, :]
Q1 = Q_mat[0, 1, :]
Q2 = Q_mat[0, 2, :]
Q3 = Q_mat[1, 1, :]
Q4 = Q_mat[1, 2, :]

output.PointData.append(Q0, "Q0")
output.PointData.append(Q1, "Q1")
output.PointData.append(Q2, "Q2")
output.PointData.append(Q3, "Q3")
output.PointData.append(Q4, "Q4")
