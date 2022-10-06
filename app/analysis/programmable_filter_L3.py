import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

dQ0 = algs.gradient(Q0)
dQ1 = algs.gradient(Q1)
dQ2 = algs.gradient(Q2)
dQ3 = algs.gradient(Q3)
dQ4 = algs.gradient(Q4)

L3_elastic = (-dQ0[:, 0] - dQ3[:, 0]) * Q1


L3_elastic = ((1.0/2.0)*(2*((-dQ0[:, 0] - dQ3[:, 0])*(-dQ0[:, 1] - dQ3[:, 1]) 
                          + dQ0[:, 0]*dQ0[:, 1] + 2*dQ1[:, 0]*dQ1[:, 1] 
                          + 2*dQ2[:, 0]*dQ2[:, 1] + dQ3[:, 0]*dQ3[:, 1] 
                          + 2*dQ4[:, 0]*dQ4[:, 1])*Q1 
                      + (-dQ0[:, 0] - dQ3[:, 0]*-dQ0[:, 0] - dQ3[:, 0] 
                          + dQ0[:, 0]*dQ0[:, 0] + 2*dQ1[:, 0]*dQ1[:, 0] 
                          + 2*dQ2[:, 0]*dQ2[:, 0] + dQ3[:, 0]*dQ3[:, 0] 
                          + 2*dQ4[:, 0]*dQ4[:, 0])*Q0 
                      + (-dQ0[:, 1] - dQ3[:, 1]*-dQ0[:, 1] - dQ3[:, 1] 
                          + dQ0[:, 1]*dQ0[:, 1] + 2*dQ1[:, 1]*dQ1[:, 1] 
                          + 2*dQ2[:, 1]*dQ2[:, 1] + dQ3[:, 1]*dQ3[:, 1] 
                          + 2*dQ4[:, 1]*dQ4[:, 1])*Q3))

output.PointData.append(L3_elastic, "L3_elastic")
