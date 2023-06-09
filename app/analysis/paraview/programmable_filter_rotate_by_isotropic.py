# Calculate eigenvalues 
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

points = inputs[0].GetPoints()
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

q1 = np.zeros(Q0.shape)
q2 = np.zeros(Q0.shape)
n = np.zeros((Q0.shape[0], 3))
m = np.zeros((Q0.shape[0], 3))

c1 = np.array([-30.0, 0])
c2 = np.array([30.0, 0])
charge_1 = 0.5
charge_2 = -0.5
ref_vec = np.array([1, 0, 0])
for i in range(q1.shape[0]):
    phi_1 = np.arctan2(points[i, 1] - c1[1],
                       points[i, 0] - c1[0])
    phi_2 = np.arctan2(points[i, 1] - c2[1],
                       points[i, 0] - c2[0])

    theta_iso = charge_1 * phi_1 + charge_2 * phi_2 + np.pi/2
    R = np.array([[np.cos(theta_iso), -np.sin(theta_iso), 0],
                  [np.sin(theta_iso), np.cos(theta_iso), 0],
                  [0, 0, 1]])

    Q = R.transpose() @ Q_mat[:, :, i] @ R
    w, v = np.linalg.eigh(Q)
    q1[i] = w[-1]
    q2[i] = w[-2]
    n[i, :] = v[:, -1]
    if (np.dot(n[i, :], ref_vec) < 0):
        n[i, :] = -n[i, :]
    m[i, :] = v[:, -2]

output.PointData.append(q1, "q1")
output.PointData.append(q2, "q2")
output.PointData.append(n, "n")
output.PointData.append(m, "m")
