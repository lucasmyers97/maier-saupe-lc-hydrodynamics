inputs = None
output = None

import numpy as np
from paraview.vtk.numpy_interface import algorithms as algs

eps = np.zeros((3, 3, 3))
eps[0, 1, 2] = 1
eps[1, 2, 0] = 1
eps[2, 0, 1] = 1
eps[2, 1, 0] = -1
eps[1, 0, 2] = -1
eps[0, 2, 1] = -1

dQ0 = algs.gradient(inputs[0].PointData['Q0'])
dQ1 = algs.gradient(inputs[0].PointData['Q1'])
dQ2 = algs.gradient(inputs[0].PointData['Q2'])
dQ3 = algs.gradient(inputs[0].PointData['Q3'])
dQ4 = algs.gradient(inputs[0].PointData['Q4'])

dQ_mat = np.zeros((dQ0.shape[0], dQ0.shape[1], 3, 3))

dQ_mat[:, :, 0, 0] = dQ0
dQ_mat[:, :, 0, 1] = dQ1
dQ_mat[:, :, 0, 2] = dQ2
dQ_mat[:, :, 1, 1] = dQ3
dQ_mat[:, :, 1, 2] = dQ4
dQ_mat[:, :, 1, 0] = dQ1
dQ_mat[:, :, 2, 0] = dQ2
dQ_mat[:, :, 2, 1] = dQ4
dQ_mat[:, :, 2, 2] = -(dQ0 + dQ3)

D = np.zeros((dQ0.shape[0], 3, 3))
for gamma in range(3):
    for i in range(3):
        D[:, gamma, i] = sum( eps[gamma, mu,  nu]
                              * eps[i, k, l]
                              * dQ_mat[:, k, mu, alpha]
                              * dQ_mat[:, l, nu, alpha]
                              for mu in range(3)
                              for nu in range(3)
                              for k in range(3)
                              for l in range(3)
                              for alpha in range(3)
                            )

angle = np.zeros((D.shape[0]))
omega = np.zeros((D.shape[0]))

for i in range(3):
    for j in range(3):
        omega += D[:, i, j] * D[:, i, j]
        angle += D[:, i, j] * D[:, j, i]

omega = np.sqrt(omega)
angle = np.arccos( np.sqrt( np.abs(angle) ) / omega )

output.PointData.append(omega, 'omega')
output.PointData.append(angle, 'angle')
