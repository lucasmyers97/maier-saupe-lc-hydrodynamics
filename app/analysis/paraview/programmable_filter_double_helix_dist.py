inputs = None
output = None

import numpy as np

d = 56.91
alpha = 0.01257

p = inputs[0].GetPoints()

Q0 = inputs[0].PointData['Q0']
Q1 = inputs[0].PointData['Q1']
Q2 = inputs[0].PointData['Q2']
Q3 = inputs[0].PointData['Q3']
Q4 = inputs[0].PointData['Q4']

dist = np.minimum( np.sqrt( (p[:, 1] - d * np.cos(alpha * p[:, 0]))**2
                       + (p[:, 2] - d * np.sin(alpha * p[:, 0]))**2 ),
               np.sqrt( (p[:, 1] + d * np.cos(alpha * p[:, 0]))**2
                       + (p[:, 2] + d * np.sin(alpha * p[:, 0]))**2 )
                 )

output.PointData.append(dist, "dist")
output.PointData.append(Q0, "Q0")
output.PointData.append(Q1, "Q1")
output.PointData.append(Q2, "Q2")
output.PointData.append(Q3, "Q3")
output.PointData.append(Q4, "Q4")
