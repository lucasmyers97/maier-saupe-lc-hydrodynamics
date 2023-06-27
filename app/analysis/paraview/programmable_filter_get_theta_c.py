import numpy as np

n = inputs[0].PointData['n']
theta_c = np.arctan2(n[:, 1], n[:, 0])

output.PointData.append(theta_c, "theta_c")
