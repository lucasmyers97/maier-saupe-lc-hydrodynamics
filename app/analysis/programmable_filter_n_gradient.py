import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np
    
from mpi4py import MPI
comm = MPI.COMM_WORLD
    
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

a = 0.8775825618903728
b = 0.479425538604203
c = 1 / np.sqrt(2)
R = np.array([[a, -b, 0],
              [b, a,  0],
              [0, 0,  1]])
n_prime = np.zeros((Q0.shape[0], 3))
    
for i in range(S.shape[0]):
    w, v = np.linalg.eig(Q_mat[:, :, i])
    w_idx = np.argsort(w)
    S[i] = 1.5 * w[w_idx[-1]]
    P[i] = 0.5 * w[w_idx[-1]] + w[w_idx[-2]]
    n[i, :] = v[:, w_idx[-1]]
    m[i, :] = v[:, w_idx[-2]]

    Q_rot = R @ Q_mat[:, :, i] @ R.transpose()
    w, v = np.linalg.eig(Q_rot)
    w_idx = np.argsort(w)
    n_prime[i, :] = R.transpose() @ v[:, w_idx[-1]]
    
output.PointData.append(S, "S")
output.PointData.append(P, "P")
output.PointData.append(n, "n")
output.PointData.append(n_prime, "n_prime")
output.PointData.append(m, "m")
# --------------------------------
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np
    
from mpi4py import MPI
comm = MPI.COMM_WORLD
    
n = inputs[0].PointData["n"]
n_prime = inputs[0].PointData["n_prime"]

grad_n = algs.gradient(n)
grad_n_prime = algs.gradient(n_prime)

new_grad_n = np.zeros(grad_n.shape)
print(grad_n.shape)

for i in range(grad_n.shape[0]):
    norm_grad_n = np.linalg.norm(grad_n[i, :, :])
    norm_grad_n_prime = np.linalg.norm(grad_n_prime[i, :, :])
    if norm_grad_n < norm_grad_n_prime:
        new_grad_n[i, :, :] = grad_n[i, :, :]
    else:
        new_grad_n[i, :, :] = grad_n_prime[i, :, :]

output.PointData.append(new_grad_n, "grad_n")
