import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

n = inputs[0].PointData['director']
dn = algs.gradient(n)

div_n = np.zeros(dn.shape[0])
for i in range(3):
    for j in range(3):
        div_n[:] += dn[:, i, j]

print(dn.shape)

output.PointData.append(dn, "dn")
output.PointData.append(div_n, "div n")
