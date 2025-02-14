# Code for 'Script'

#This script generates a helix curve.
#This is intended as the script of a 'Programmable Source'
import math
import numpy as np
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa

numPts = 100  # Points along Helix
x0 = -200.0 # initial point
xf = 200.0 # final point
r = 52.4913 
alpha = 0.00314 # twist angular velocity

# Compute the point coordinates for the helix.
x = np.linspace(x0, xf, numPts);
y = r * np.cos(alpha * x)
z = r * np.sin(alpha * x)

# Create a (x,y,z) coordinates array and associate that with
# points to pass to the output dataset.
coordinates = algs.make_vector(x, y, z)
pts = vtk.vtkPoints()
pts.SetData(dsa.numpyTovtkDataArray(coordinates, 'Points'))
output.SetPoints(pts)

# Next, we need to define the topology i.e.
# cell information. This helix will be a single
# polyline connecting all the  points in order.
ptIds = vtk.vtkIdList()
ptIds.SetNumberOfIds(numPts)
for i in range(numPts):
   #Add the points to the line. The first value indicates
   #the order of the point on the line. The second value
   #is a reference to a point in a vtkPoints object. Depends
   #on the order that Points were added to vtkPoints object.
   #Note that this will not be associated with actual points
   #until it is added to a vtkPolyData object which holds a
   #vtkPoints object.
   ptIds.SetId(i, i)

# Allocate the number of 'cells' that will be added. We are just
# adding one vtkPolyLine 'cell' to the vtkPolyData object.
output.Allocate(1, 1)

# Add the poly line 'cell' to the vtkPolyData object.
output.InsertNextCell(vtk.VTK_POLY_LINE, ptIds)
