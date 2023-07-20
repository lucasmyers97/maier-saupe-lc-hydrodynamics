import numpy as np

points_1 = inputs[0].GetPoints()
points_2 = inputs[1].GetPoints()

idx_1 = np.lexsort((points_1[:, 0], points_1[:, 1]))
idx_2 = np.lexsort((points_2[:, 0], points_2[:, 1]))

theta_1 = inputs[0].PointData['Q0'][idx_1]
theta_2 = inputs[1].PointData['theta_c'][idx_2]

error = np.zeros(theta_1.shape)
error[idx_1] = theta_1 - theta_2

output.PointData.append(error, 'theta_diff')
