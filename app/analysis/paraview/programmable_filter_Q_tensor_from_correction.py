from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

S = 0.6751
eps = 0.1

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
theta = theta_iso + eps * theta_c

Q0 = 0.5 * S * ( 1.0/3.0 + np.cos(2*theta) );
Q1 = 0.5 * S * np.sin(2*theta);
Q2 = 0.0;
Q3 = 0.5 * S * ( 1.0/3.0 - np.cos(2*theta) );
Q4 = 0.0;

output.PointData.append(Q0, 'Q0')
output.PointData.append(Q1, 'Q1')
output.PointData.append(Q2, 'Q2')
output.PointData.append(Q3, 'Q3')
output.PointData.append(Q4, 'Q4')
