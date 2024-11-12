inputs = None
output = None

import numpy as np

axis = 'x'
director = 'n'

n = inputs[0].PointData[director]
p = inputs[0].GetPoints()

axis_list = ['x', 'y', 'z']
axis_list.remove( axis )
axis_idx_dict = {'x': 0, 'y': 1, 'z': 2}
a = [axis_idx_dict[axis_list[0]], axis_idx_dict[axis_list[1]]]

n_dot_p = n[:, a[0]] * p[:, a[0]] + n[:, a[1]] * p[:, a[1]]
p_mag = np.sqrt( p[:, a[0]]**2 + p[:, a[1]]**2 )
n_mag = np.sqrt( n[:, a[0]]**2 + n[:, a[1]]**2 )
mag_prod = p_mag * n_mag
mag_prod_zero = mag_prod == 0
mag_prod[mag_prod_zero] = 1
n_proj_p = n_dot_p / mag_prod

# Make all positive, in case director is oriented incorrectly
n_proj_p[n_proj_p < 0] *= -1

# If projection is somehow greater than 1, just squish it to 1
n_proj_p[n_proj_p > 1] = 1

angle = np.arccos( n_proj_p )
angle[mag_prod_zero] = np.nan

# add to output
output.PointData.append(angle, "angle");
