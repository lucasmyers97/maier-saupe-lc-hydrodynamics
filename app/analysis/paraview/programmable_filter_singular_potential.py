alpha = 8.0

import sys

ball_path = '/home/lucas/Documents/research/py-ball-majumdar-singular-potential'
if ball_path not in sys.path:
    sys.path.append(ball_path)

import ball_majumdar_singular_potential as bmsp
import numpy as np

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

Lambda0 = np.zeros(Q0.shape)
Lambda1 = np.zeros(Q1.shape)
Lambda2 = np.zeros(Q2.shape)
Lambda3 = np.zeros(Q3.shape)
Lambda4 = np.zeros(Q4.shape)

Z = np.zeros(Q0.shape)

singular_potential = bmsp.singular_potential_3D(974, 1.0, 1e-10, 100)

for i in range(Q0.shape[0]):
    Q = np.array([Q0[i], Q3[i], Q1[i], Q2[i], Q4[i]])
    singular_potential.invert_Q(Q)
    Lambda = singular_potential.return_Lambda()
    Lambda0[i] = Lambda[0]
    Lambda1[i] = Lambda[2]
    Lambda2[i] = Lambda[3]
    Lambda3[i] = Lambda[1]
    Lambda4[i] = Lambda[4]
    Z[i] = singular_potential.return_Z()


mean_field_term = alpha*(-Q0*Q0 - Q0*Q3 - Q1*Q1 - Q2*Q2 - Q3*Q3 - Q4*Q4)

entropy_term = (
        2*Q0*Lambda0 + Q0*Lambda3 
        + 2*Q1*Lambda1 + 2*Q2*Lambda2 
        + Q3*Lambda0 + 2*Q3*Lambda3 
        + 2*Q4*Lambda4 - np.log(Z) + np.log(4*np.pi)
        )

output.PointData.append(mean_field_term, "mean_field_term")
output.PointData.append(entropy_term, "entropy_term")

# output.PointData.append(Lambda0, "Lambda0")
# output.PointData.append(Lambda1, "Lambda1")
# output.PointData.append(Lambda2, "Lambda2")
# output.PointData.append(Lambda3, "Lambda3")
# output.PointData.append(Lambda4, "Lambda4")
# output.PointData.append(Z, "Z")
