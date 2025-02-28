# Calculate eigenvalues 
import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

L2 = 10.0
L3 = 0.0

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

dQ0 = algs.gradient(inputs[0].PointData['Q0'])
dQ1 = algs.gradient(inputs[0].PointData['Q1'])
dQ2 = algs.gradient(inputs[0].PointData['Q2'])
dQ3 = algs.gradient(inputs[0].PointData['Q3'])
dQ4 = algs.gradient(inputs[0].PointData['Q4'])


L1_term = (
        0.5*(-dQ0[:, 0] - dQ3[:, 0]) 
        * (-dQ0[:, 0] - dQ3[:, 0]) 
        + 0.5*(-dQ0[:, 1] - dQ3[:, 1]) 
        * (-dQ0[:, 1] - dQ3[:, 1]) 
        + 0.5*(dQ0[:, 0]) 
        * (dQ0[:, 0]) 
        + 0.5*(dQ0[:, 1]) * (dQ0[:, 1]) 
        + (dQ1[:, 0]) * (dQ1[:, 0]) 
        + (dQ1[:, 1]) * (dQ1[:, 1]) 
        + (dQ2[:, 0]) * (dQ2[:, 0]) 
        + (dQ2[:, 1]) * (dQ2[:, 1]) 
        + 0.5*(dQ3[:, 0]) * (dQ3[:, 0]) 
        + 0.5*(dQ3[:, 1]) * (dQ3[:, 1]) 
        + (dQ4[:, 0]) * (dQ4[:, 0]) 
        + (dQ4[:, 1]) * (dQ4[:, 1])
        )

L2_term = (
        0.5*L2
        * ((dQ0[:, 0] + dQ1[:, 1]) * (dQ0[:, 0] + dQ1[:, 1]) 
           + (dQ1[:, 0] + dQ3[:, 1]) * (dQ1[:, 0] + dQ3[:, 1]) 
           + (dQ2[:, 0] + dQ4[:, 1]) * (dQ2[:, 0] + dQ4[:, 1]))
        )


L3_term = (
        0.5*L3
        *(2*((-dQ0[:, 0] - dQ3[:, 0])*(-dQ0[:, 1] - dQ3[:, 1]) 
             + dQ0[:, 0]*dQ0[:, 1] + 2*dQ1[:, 0]*dQ1[:, 1] 
             + 2*dQ2[:, 0]*dQ2[:, 1] + dQ3[:, 0]*dQ3[:, 1] 
             + 2*dQ4[:, 0]*dQ4[:, 1])*Q1 
          + ((-dQ0[:, 0] - dQ3[:, 0]) 
             * (-dQ0[:, 0] - dQ3[:, 0]) 
             + (dQ0[:, 0]) * (dQ0[:, 0]) 
             + 2*(dQ1[:, 0]) * (dQ1[:, 0]) 
             + 2*(dQ2[:, 0]) * (dQ2[:, 0]) 
             + (dQ3[:, 0]) * (dQ3[:, 0]) 
             + 2*(dQ4[:, 0]) * (dQ4[:, 0]))*Q0 
          + ((-dQ0[:, 1] - dQ3[:, 1]) 
             * (-dQ0[:, 1] - dQ3[:, 1]) 
             + (dQ0[:, 1]) * (dQ0[:, 1]) 
             + 2*(dQ1[:, 1]) * (dQ1[:, 1]) 
             + 2*(dQ2[:, 1]) * (dQ2[:, 1]) 
             + (dQ3[:, 1]) * (dQ3[:, 1]) 
             + 2*(dQ4[:, 1]) * (dQ4[:, 1]))*Q3)
          )

output.PointData.append(L1_term, "L1_term")
output.PointData.append(L2_term, "L2_term")
output.PointData.append(L3_term, "L3_term")
output.PointData.append(L1_term + L2_term + L3_term, "total elastic energy")
