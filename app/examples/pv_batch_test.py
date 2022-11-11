# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10
import time
start = time.time()

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Partitioned Unstructured Grid Reader'
q_components_two_defect_ = XMLPartitionedUnstructuredGridReader(registrationName='Q_components_two_defect_*', FileName=['/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_00.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_1900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_2900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_3900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_4900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_5900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_6900.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7000.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7100.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7200.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7300.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7400.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7500.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7600.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7700.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7800.pvtu', '/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/supercomputer-two-defect-eps-new/two-defect-eps-0_1/Q_components_two_defect_7900.pvtu'])
q_components_two_defect_.PointArrayStatus = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'subdomain']

# process_id = ProcessIdScalars(Input=q_components_two_defect_)
# process_view = Show(process_id)
# process_view.ColorArrayName = 'ProcessId'
# process_view = Render()
# SaveScreenshot("process_id.png", viewOrLayout=process_view)


# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on q_components_two_defect_
q_components_two_defect_.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
q_components_two_defect_Display = Show(q_components_two_defect_, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
q_components_two_defect_Display.Representation = 'Surface'
q_components_two_defect_Display.ColorArrayName = [None, '']
q_components_two_defect_Display.SelectTCoordArray = 'None'
q_components_two_defect_Display.SelectNormalArray = 'None'
q_components_two_defect_Display.SelectTangentArray = 'None'
q_components_two_defect_Display.OSPRayScaleArray = 'Q0'
q_components_two_defect_Display.OSPRayScaleFunction = 'PiecewiseFunction'
q_components_two_defect_Display.SelectOrientationVectors = 'None'
q_components_two_defect_Display.ScaleFactor = 26.6
q_components_two_defect_Display.SelectScaleArray = 'None'
q_components_two_defect_Display.GlyphType = 'Arrow'
q_components_two_defect_Display.GlyphTableIndexArray = 'None'
q_components_two_defect_Display.GaussianRadius = 1.33
q_components_two_defect_Display.SetScaleArray = ['POINTS', 'Q0']
q_components_two_defect_Display.ScaleTransferFunction = 'PiecewiseFunction'
q_components_two_defect_Display.OpacityArray = ['POINTS', 'Q0']
q_components_two_defect_Display.OpacityTransferFunction = 'PiecewiseFunction'
q_components_two_defect_Display.DataAxesGrid = 'GridAxesRepresentation'
q_components_two_defect_Display.PolarAxes = 'PolarAxesRepresentation'
q_components_two_defect_Display.ScalarOpacityUnitDistance = 4.9456049661243116
q_components_two_defect_Display.OpacityArrayName = ['POINTS', 'Q0']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
q_components_two_defect_Display.ScaleTransferFunction.Points = [-0.22502882778644562, 0.0, 0.5, 0.0, 0.44961023330688477, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
q_components_two_defect_Display.OpacityTransferFunction.Points = [-0.22502882778644562, 0.0, 0.5, 0.0, 0.44961023330688477, 1.0, 0.5, 0.0]

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
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=q_components_two_defect_)
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# Properties modified on programmableFilter1
programmableFilter1.Script = """import paraview.vtk.numpy_interface.algorithms as algs
import numpy as np

Q0 = inputs[0].PointData[\'Q0\']
Q1 = inputs[0].PointData[\'Q1\']
Q2 = inputs[0].PointData[\'Q2\']
Q3 = inputs[0].PointData[\'Q3\']
Q4 = inputs[0].PointData[\'Q4\']

Q_mat = np.zeros((3, 3, Q0.shape[0]))

Q_mat[0, 0, :] = Q0
Q_mat[0, 1, :] = Q1
Q_mat[0, 2, :] = Q2
Q_mat[1, 1, :] = Q3
Q_mat[1, 2, :] = Q4
Q_mat[1, 0, :] = Q_mat[0, 1, :]
Q_mat[2, 0, :] = Q_mat[0, 2, :]
Q_mat[2, 1, :] = Q_mat[1, 2, :]

S = np.zeros(Q0.shape)
P = np.zeros(Q0.shape)
n = np.zeros((Q0.shape[0], 3))
m = np.zeros((Q0.shape[0], 3))

for i in range(S.shape[0]):
    w, v = np.linalg.eig(Q_mat[:, :, i])
    w_idx = np.argsort(w)
    S[i] = w[w_idx[-1]]
    P[i] = w[w_idx[-2]]
    n[i, :] = v[:, w_idx[-1]]
    m[i, :] = v[:, w_idx[-2]]

output.PointData.append(S, "S")
output.PointData.append(P, "P")
output.PointData.append(n, "n")
output.PointData.append(m, "m")
"""
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
programmableFilter1Display.OSPRayScaleArray = 'P'
programmableFilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
programmableFilter1Display.SelectOrientationVectors = 'None'
programmableFilter1Display.ScaleFactor = 26.6
programmableFilter1Display.SelectScaleArray = 'None'
programmableFilter1Display.GlyphType = 'Arrow'
programmableFilter1Display.GlyphTableIndexArray = 'None'
programmableFilter1Display.GaussianRadius = 1.33
programmableFilter1Display.SetScaleArray = ['POINTS', 'P']
programmableFilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.OpacityArray = ['POINTS', 'P']
programmableFilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.DataAxesGrid = 'GridAxesRepresentation'
programmableFilter1Display.PolarAxes = 'PolarAxesRepresentation'
programmableFilter1Display.ScalarOpacityUnitDistance = 4.9456049661243116
programmableFilter1Display.OpacityArrayName = ['POINTS', 'P']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
programmableFilter1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.06716713309288025, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
programmableFilter1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.06716713309288025, 1.0, 0.5, 0.0]

# hide data in view
Hide(q_components_two_defect_, renderView1)

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
layout1.SetSize(1513, 760)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.CameraParallelScale = 188.09040379562165

SaveScreenshot("scratch_test.png")

end = time.time()
print(end - start)

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
