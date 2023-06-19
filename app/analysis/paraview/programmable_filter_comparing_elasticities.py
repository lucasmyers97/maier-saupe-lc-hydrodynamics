from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

grad_theta_2 = inputs[0].PointData['grad_theta_2']
Splay = inputs[0].PointData['splay']
Bend = inputs[0].PointData['bend']

L1_term = inputs[1].PointData['L1_term']
L2_term = inputs[1].PointData['L2_term']
L3_term = inputs[1].PointData['L3_term']

S = 0.6751

L1_term_expected = S**2 * 2 * grad_theta_2
L2_term_expected = S**2 * (Splay + Bend)
L3_term_expected = 2 * S**3 * (Splay - (1/3)*grad_theta_2)
# 
# output.PointData.append(L1_term - L1_term_expected, 'L1_diff')  
# output.PointData.append(L2_term - L2_term_expected, 'L2_diff')  
# output.PointData.append(L3_term - L3_term_expected, 'L3_diff')  
