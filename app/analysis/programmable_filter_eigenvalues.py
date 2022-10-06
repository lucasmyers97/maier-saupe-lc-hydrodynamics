import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

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


n = inputs[0].PointData['n']
n_prime = inputs[0].PointData['n_prime']

grad_n = algs.gradient(n)
grad_n_prime = algs.gradient(n_prime)

final_grad_n = np.zeros(grad_n.shape)

for i in range(grad_n.shape[0]):
    grad_n_norm = np.linalg.norm(grad_n[i, :, :])
    grad_n_prime_norm = np.linalg.norm(grad_n_prime[i, :, :])
    if grad_n_norm < grad_n_prime_norm:
        final_grad_n[i, :, :] = grad_n[i, :, :]
    else:
        final_grad_n[i, :, :] = grad_n_prime[i, :, :]

output.PointData.append(final_grad_n, "grad_n")

splay = np.zeros(final_grad_n.shape[0])
for i in range(3):
    splay += final_grad_n[:, i, i]**2

bend = (final_grad_n[:, 0, 1] - final_grad_n[:, 1, 0])**2

output.PointData.append(final_grad_n, "grad_n")
output.PointData.append(bend, "bend")
output.PointData.append(splay, "splay")
