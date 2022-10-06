# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Partitioned Unstructured Grid Reader'
q_components_0 = XMLPartitionedUnstructuredGridReader(registrationName='Q_components_0*', FileName=['/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect/Q_components_00.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect/Q_components_01.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect/Q_components_02.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect/Q_components_03.pvtu'])
q_components_0.PointArrayStatus = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'subdomain']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on q_components_0
q_components_0.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
q_components_0Display = Show(q_components_0, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
q_components_0Display.Representation = 'Surface'
q_components_0Display.ColorArrayName = [None, '']
q_components_0Display.SelectTCoordArray = 'None'
q_components_0Display.SelectNormalArray = 'None'
q_components_0Display.SelectTangentArray = 'None'
q_components_0Display.OSPRayScaleArray = 'Q0'
q_components_0Display.OSPRayScaleFunction = 'PiecewiseFunction'
q_components_0Display.SelectOrientationVectors = 'None'
q_components_0Display.ScaleFactor = 46.6
q_components_0Display.SelectScaleArray = 'None'
q_components_0Display.GlyphType = 'Arrow'
q_components_0Display.GlyphTableIndexArray = 'None'
q_components_0Display.GaussianRadius = 2.33
q_components_0Display.SetScaleArray = ['POINTS', 'Q0']
q_components_0Display.ScaleTransferFunction = 'PiecewiseFunction'
q_components_0Display.OpacityArray = ['POINTS', 'Q0']
q_components_0Display.OpacityTransferFunction = 'PiecewiseFunction'
q_components_0Display.DataAxesGrid = 'GridAxesRepresentation'
q_components_0Display.PolarAxes = 'PolarAxesRepresentation'
q_components_0Display.ScalarOpacityUnitDistance = 8.66410494065387
q_components_0Display.OpacityArrayName = ['POINTS', 'Q0']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
q_components_0Display.ScaleTransferFunction.Points = [-0.22502918541431427, 0.0, 0.5, 0.0, 0.4499047100543976, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
q_components_0Display.OpacityTransferFunction.Points = [-0.22502918541431427, 0.0, 0.5, 0.0, 0.4499047100543976, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.CameraFocalPoint = [0.0, 0.0, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=q_components_0)
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# Properties modified on programmableFilter1
programmableFilter1.Script = """# Code for 'Script'

#This script generates a helix curve.
#This is intended as the script of a 'Programmable Source'
import math
import numpy as np
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa

numPts = 80  # Points along Helix
length = 8.0 # Length of Helix
rounds = 3.0 # Number of times around

# Compute the point coordinates for the helix.
index = np.arange(0, numPts, dtype=np.int32)
scalars = index * rounds * 2 * math.pi / numPts
x = index * length / numPts;
y = np.sin(scalars)
z = np.cos(scalars)

# Create a (x,y,z) coordinates array and associate that with
# points to pass to the output dataset.
coordinates = algs.make_vector(x, y, z)
pts = vtk.vtkPoints()
pts.SetData(dsa.numpyTovtkDataArray(coordinates, 'Points'))
output.SetPoints(pts)

# Add scalars to the output point data.
output.PointData.append(index, 'Index')
output.PointData.append(scalars, 'Scalars')

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
output.InsertNextCell(vtk.VTK_POLY_LINE, ptIds)"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# show data in view
programmableFilter1Display = Show(programmableFilter1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
programmableFilter1Display.Representation = 'Surface'
programmableFilter1Display.ColorArrayName = [None, '']
programmableFilter1Display.SelectTCoordArray = 'None'
programmableFilter1Display.SelectNormalArray = 'None'
programmableFilter1Display.SelectTangentArray = 'None'
programmableFilter1Display.OSPRayScaleArray = 'Index'
programmableFilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
programmableFilter1Display.SelectOrientationVectors = 'None'
programmableFilter1Display.ScaleFactor = 0.79
programmableFilter1Display.SelectScaleArray = 'Index'
programmableFilter1Display.GlyphType = 'Arrow'
programmableFilter1Display.GlyphTableIndexArray = 'Index'
programmableFilter1Display.GaussianRadius = 0.0395
programmableFilter1Display.SetScaleArray = ['POINTS', 'Index']
programmableFilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.OpacityArray = ['POINTS', 'Index']
programmableFilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.DataAxesGrid = 'GridAxesRepresentation'
programmableFilter1Display.PolarAxes = 'PolarAxesRepresentation'
programmableFilter1Display.ScalarOpacityUnitDistance = 8.391066678319271
programmableFilter1Display.OpacityArrayName = ['POINTS', 'Index']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
programmableFilter1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 79.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
programmableFilter1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 79.0, 1.0, 0.5, 0.0]

# hide data in view
Hide(q_components_0, renderView1)

# update the view to ensure updated data information
renderView1.Update()

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1095, 760)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.CameraParallelScale = 329.51176003293114

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
