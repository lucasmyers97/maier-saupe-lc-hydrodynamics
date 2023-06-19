from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

theta_c = inputs[0].PointData['theta_c']

points = inputs[0].GetPoints()

x = points[:, 0]
y = points[:, 1]
x1 = -30
x2 = 30

r1 = np.sqrt((x - x1)**2 + y**2)
r2 = np.sqrt((x - x2)**2 + y**2)
phi1 = np.arctan2(y, x - x1)
phi2 = np.arctan2(y, x - x2)

q1 = 0.5
q2 = -0.5

theta_iso = q1 * phi1 + q2 * phi2
theta = theta_iso + theta_c

output.PointData.append(theta, 'theta')
