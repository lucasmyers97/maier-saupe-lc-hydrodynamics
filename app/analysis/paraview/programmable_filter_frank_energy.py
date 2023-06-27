from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

eps = 0.1

theta_c = inputs[0].PointData['theta_c']
grad_theta_c = algs.gradient(theta_c)

points = inputs[0].GetPoints()

x = points[:, 0]
y = points[:, 1]
x1 = -30
x2 = 30

r1 = np.sqrt((x - x1)**2 + y**2)
r2 = np.sqrt((x - x2)**2 + y**2)
phi1 = np.arctan2(y, x - x1)
phi2 = np.arctan2(y, x - x2)
sin_phi1 = y / r1
sin_phi2 = y / r2
cos_phi1 = (x - x1) / r1
cos_phi2 = (x - x2) / r2

q1 = 0.5
q2 = -0.5

theta_iso = q1 * phi1 + q2 * phi2
theta = theta_iso + eps * theta_c

grad_theta_iso = np.zeros(grad_theta_c.shape)
grad_theta_iso[:, 0] = -q1*(1/r1)*sin_phi1 - q2*(1/r2)*sin_phi2
grad_theta_iso[:, 1] = q1*(1/r1)*cos_phi1 + q2*(1/r2)*cos_phi2

grad_theta = grad_theta_iso + eps * grad_theta_c

output.PointData.append(grad_theta, "grad_theta")

grad_theta_2 = sum(grad_theta[:, i]*grad_theta[:, i]
                   for i in range(3))

S = (0.5 * (grad_theta[:, 0]**2 + grad_theta[:, 1]**2)
     + 0.5 * np.cos(2*theta) * (grad_theta[:, 1]**2 - grad_theta[:, 0]**2)
     - np.sin(2*theta) * grad_theta[:, 0] * grad_theta[:, 1])

B = (0.5 * (grad_theta[:, 0]**2 + grad_theta[:, 1]**2)
     + 0.5 * np.cos(2*theta) * (grad_theta[:, 0]**2 - grad_theta[:, 1]**2)
     + np.sin(2*theta) * grad_theta[:, 0] * grad_theta[:, 1])

E = (1 - eps)*S + (1 + eps)*B

output.PointData.append(S, 'splay')
output.PointData.append(B, 'bend')
output.PointData.append(E, 'elastic energy')
output.PointData.append(grad_theta_2, 'grad_theta_2')
