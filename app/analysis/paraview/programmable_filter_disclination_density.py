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

Omega = np.zeros((D.shape[0], 3))
T = np.zeros((D.shape[0], 3))
omega = np.zeros((D.shape[0]))

def non_degenerate(l):
    dists = np.array([np.abs(l[0] - l[1]),
                      np.abs(l[1] - l[2]),
                      np.abs(l[0] - l[2])])
    idx = np.argmin(dists)
    if idx == 0:
        return 2
    elif idx == 1:
        return 0
    else:
        return 1

for i in range(D.shape[0]):
    DDT = D[i, :, :] @ D[i, :, :].transpose()
    DTD = D[i, :, :].transpose() @ D[i, :, :]

    l, n = np.linalg.eigh(DDT)
    idx = non_degenerate(l)
    Omega[i, :] = n[:, idx]

    l, n = np.linalg.eigh(DTD)
    idx = non_degenerate(l)
    if n[0, idx] > 0:
        T[i, :] = n[:, idx]
    else:
        T[i, :] = -n[:, idx]

    if np.sign( np.dot(Omega[i, :], T[i, :]) ) != np.sign( np.trace(D[i, :, :]) ):
        Omega[i, :] *= -1

    omega[i] = np.linalg.norm(D[i, :, :])

output.PointData.append(Omega, "Omega")
output.PointData.append(T, "T")
output.PointData.append(omega, "omega")

output.PointData.append(D[:, 0, 0], "D00")
output.PointData.append(D[:, 0, 1], "D01")
output.PointData.append(D[:, 0, 2], "D02")
output.PointData.append(D[:, 1, 0], "D10")
output.PointData.append(D[:, 1, 1], "D11")
output.PointData.append(D[:, 1, 2], "D12")
output.PointData.append(D[:, 2, 0], "D20")
output.PointData.append(D[:, 2, 1], "D21")
output.PointData.append(D[:, 2, 2], "D22")
