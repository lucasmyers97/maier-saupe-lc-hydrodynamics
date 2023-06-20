# Calculate eigenvalues 
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

L2 = 0
L3 = 0.3064680072939386

points = inputs[0].GetPoints()
Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

Q_mat = np.zeros(Q0.shape + (3, 3))

Q_mat[:, 0, 0] = Q0
Q_mat[:, 0, 1] = Q1
Q_mat[:, 0, 2] = Q2
Q_mat[:, 1, 1] = Q3
Q_mat[:, 1, 2] = Q4
Q_mat[:, 1, 0] = Q1
Q_mat[:, 2, 0] = Q2
Q_mat[:, 2, 1] = Q4
Q_mat[:, 2, 2] = -(Q0 + Q3)

dQ0 = algs.gradient(Q0)
dQ1 = algs.gradient(Q1)
dQ2 = algs.gradient(Q2)
dQ3 = algs.gradient(Q3)
dQ4 = algs.gradient(Q4)

dQ_mat = np.zeros(dQ0.shape + (3, 3))

dQ_mat[:, :, 0, 0] = dQ0
dQ_mat[:, :, 0, 1] = dQ1
dQ_mat[:, :, 0, 2] = dQ2
dQ_mat[:, :, 1, 1] = dQ3
dQ_mat[:, :, 1, 2] = dQ4
dQ_mat[:, :, 1, 0] = dQ1
dQ_mat[:, :, 2, 0] = dQ2
dQ_mat[:, :, 2, 1] = dQ4
dQ_mat[:, :, 2, 2] = -(dQ0 + dQ3)

L1_term = sum(dQ_mat[:, k, i, j]*dQ_mat[:, k, i, j]
              for i in range(3)
              for j in range(3)
              for k in range(3))

L2_term = sum(dQ_mat[:, j, i, j]*dQ_mat[:, k, i, k]
              for i in range(3)
              for j in range(3)
              for k in range(3))

L3_term = sum(Q_mat[:, l, k]*dQ_mat[:, l, i, j]*dQ_mat[:, k, i, j]
              for i in range(3)
              for j in range(3)
              for k in range(3)
              for l in range(3))

elastic_energy = L1_term + L2 * L2_term + L3 * L3_term

output.PointData.append(L1_term, 'L1_term')
output.PointData.append(L2_term, 'L2_term')
output.PointData.append(L3_term, 'L3_term')
output.PointData.append(elastic_energy, 'elastic_energy')

