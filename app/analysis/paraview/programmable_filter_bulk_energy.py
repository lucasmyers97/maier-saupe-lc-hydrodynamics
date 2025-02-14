import numpy as np

kappa = 8.0

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

Lambda0 = inputs[1].PointData['Lambda0']
Lambda1 = inputs[1].PointData['Lambda1']
Lambda2 = inputs[1].PointData['Lambda2']
Lambda3 = inputs[1].PointData['Lambda3']
Lambda4 = inputs[1].PointData['Lambda4']
Z = inputs[1].PointData['Z']


mean_field_term = kappa*(-Q0*Q0 - Q0*Q3 - Q1*Q1 - Q2*Q2 - Q3*Q3 - Q4*Q4)

entropy_term = (2*Q0*Lambda0 + Q0*Lambda3 
                + 2*Q1*Lambda1 + 2*Q2*Lambda2 
                + Q3*Lambda0 + 2*Q3*Lambda3 
                + 2*Q4*Lambda4 - np.log(Z) + np.log(4*np.pi))

output.PointData.append(mean_field_term, "mean_field_term")
output.PointData.append(entropy_term, "entropy_term")
output.PointData.append(mean_field_term + entropy_term, "bulk energy")
