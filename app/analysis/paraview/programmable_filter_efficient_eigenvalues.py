# Calculate eigenvalues 
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

S_P = np.zeros(Q0.shape)
for i in range(Q1.shape[0]):

    Q_mat = np.zeros((3, 3))
    
    Q_mat[0, 0] = Q0[i]
    Q_mat[0, 1] = Q1[i]
    Q_mat[0, 2] = Q2[i]
    Q_mat[1, 1] = Q3[i]
    Q_mat[1, 2] = Q4[i]
    Q_mat[1, 0] = Q_mat[0, 1]
    Q_mat[2, 0] = Q_mat[0, 2]
    Q_mat[2, 1] = Q_mat[1, 2]
    Q_mat[2, 2] = -(Q_mat[0, 0] + Q_mat[1, 1])

    w, v = np.linalg.eigh(Q_mat)
    S_P[i] = w[-1] - w[-2]

output.PointData.append(S_P, "S - P")
