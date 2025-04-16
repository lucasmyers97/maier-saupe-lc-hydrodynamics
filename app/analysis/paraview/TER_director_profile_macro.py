# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
q_components_nematic_configuration_50000pvtu = FindSource('Q_components_nematic_configuration_50000.pvtu')

# create a new 'Q tensor eigensystem'
qtensoreigensystem1 = Qtensoreigensystem(registrationName='Qtensoreigensystem1', Input=q_components_nematic_configuration_50000pvtu)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
qtensoreigensystem1Display = Show(qtensoreigensystem1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
qtensoreigensystem1Display.Representation = 'Surface'

# hide data in view
Hide(q_components_nematic_configuration_50000pvtu, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Director radial angle filter'
directorradialanglefilter1 = Directorradialanglefilter(registrationName='Directorradialanglefilter1', Input=qtensoreigensystem1)

# show data in view
directorradialanglefilter1Display = Show(directorradialanglefilter1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
directorradialanglefilter1Display.Representation = 'Surface'

# hide data in view
Hide(qtensoreigensystem1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(directorradialanglefilter1Display, ('POINTS', 'angle'))

# rescale color and/or opacity maps used to include current data range
directorradialanglefilter1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
directorradialanglefilter1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'angle'
angleLUT = GetColorTransferFunction('angle')

# get opacity transfer function/opacity map for 'angle'
anglePWF = GetOpacityTransferFunction('angle')

# get 2D transfer function for 'angle'
angleTF2D = GetTransferFunction2D('angle')

# create a new 'get director z angle'
getdirectorzangle1 = getdirectorzangle(registrationName='getdirectorzangle1', Input=directorradialanglefilter1)

# set active source
SetActiveSource(directorradialanglefilter1)

# destroy getdirectorzangle1
Delete(getdirectorzangle1)
del getdirectorzangle1

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# set active source
SetActiveSource(qtensoreigensystem1)

# create a new 'get director z angle'
getdirectorzangle1 = getdirectorzangle(registrationName='getdirectorzangle1', Input=qtensoreigensystem1)

# show data in view
getdirectorzangle1Display = Show(getdirectorzangle1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
getdirectorzangle1Display.Representation = 'Surface'

# hide data in view
Hide(qtensoreigensystem1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(getdirectorzangle1Display, ('POINTS', 'angle'))

# rescale color and/or opacity maps used to include current data range
getdirectorzangle1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
getdirectorzangle1Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(directorradialanglefilter1, renderView1)

# set active source
SetActiveSource(directorradialanglefilter1)

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=directorradialanglefilter1)

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.0, -100.0, 0.0]
plotOverLine1.Point2 = [0.0, 100.0, 0.0]

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesOpacity = ['angle', '1', 'arc_length', '1', 'm_X', '1', 'm_Y', '1', 'm_Z', '1', 'm_Magnitude', '1', 'n_X', '1', 'n_Y', '1', 'n_Z', '1', 'n_Magnitude', '1', 'P', '1', 'q1', '1', 'q2', '1', 'S', '1', 'S - P', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesPlotCorner = ['P', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'S', '0', 'S - P', '0', 'angle', '0', 'arc_length', '0', 'm_Magnitude', '0', 'm_X', '0', 'm_Y', '0', 'm_Z', '0', 'n_Magnitude', '0', 'n_X', '0', 'n_Y', '0', 'n_Z', '0', 'q1', '0', 'q2', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesLineStyle = ['P', '1', 'Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'S', '1', 'S - P', '1', 'angle', '1', 'arc_length', '1', 'm_Magnitude', '1', 'm_X', '1', 'm_Y', '1', 'm_Z', '1', 'n_Magnitude', '1', 'n_X', '1', 'n_Y', '1', 'n_Z', '1', 'q1', '1', 'q2', '1', 'vtkValidPointMask', '1']
plotOverLine1Display_1.SeriesLineThickness = ['P', '2', 'Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'S', '2', 'S - P', '2', 'angle', '2', 'arc_length', '2', 'm_Magnitude', '2', 'm_X', '2', 'm_Y', '2', 'm_Z', '2', 'n_Magnitude', '2', 'n_X', '2', 'n_Y', '2', 'n_Z', '2', 'q1', '2', 'q2', '2', 'vtkValidPointMask', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['P', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'S', '0', 'S - P', '0', 'angle', '0', 'arc_length', '0', 'm_Magnitude', '0', 'm_X', '0', 'm_Y', '0', 'm_Z', '0', 'n_Magnitude', '0', 'n_X', '0', 'n_Y', '0', 'n_Z', '0', 'q1', '0', 'q2', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesMarkerSize = ['P', '4', 'Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'S', '4', 'S - P', '4', 'angle', '4', 'arc_length', '4', 'm_Magnitude', '4', 'm_X', '4', 'm_Y', '4', 'm_Z', '4', 'n_Magnitude', '4', 'n_X', '4', 'n_Y', '4', 'n_Z', '4', 'q1', '4', 'q2', '4', 'vtkValidPointMask', '4']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['angle', 'arc_length', 'm_Magnitude', 'm_X', 'm_Y', 'm_Z', 'n_Magnitude', 'n_X', 'n_Y', 'n_Z', 'P', 'Points_Magnitude', 'Points_X', 'Points_Y', 'Points_Z', 'q1', 'q2', 'S', 'S - P', 'vtkValidPointMask']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = []

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['angle']

# set active source
SetActiveSource(getdirectorzangle1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=plotOverLine1)

# create a new 'Plot Over Line'
plotOverLine2 = PlotOverLine(registrationName='PlotOverLine2', Input=getdirectorzangle1)

# Properties modified on plotOverLine2
plotOverLine2.Point1 = [0.0, -100.0, 0.0]
plotOverLine2.Point2 = [0.0, 100.0, 0.0]

# show data in view
plotOverLine2Display = Show(plotOverLine2, lineChartView1, 'XYChartRepresentation')

# update the view to ensure updated data information
lineChartView1.Update()

# Properties modified on plotOverLine2Display
plotOverLine2Display.SeriesOpacity = ['angle', '1', 'arc_length', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine2Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'angle', '0', 'arc_length', '0', 'vtkValidPointMask', '0']
plotOverLine2Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'angle', '1', 'arc_length', '1', 'vtkValidPointMask', '1']
plotOverLine2Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'angle', '2', 'arc_length', '2', 'vtkValidPointMask', '2']
plotOverLine2Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'angle', '0', 'arc_length', '0', 'vtkValidPointMask', '0']
plotOverLine2Display.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'angle', '4', 'arc_length', '4', 'vtkValidPointMask', '4']

# Properties modified on plotOverLine2Display
plotOverLine2Display.SeriesVisibility = ['angle', 'arc_length', 'Points_Magnitude', 'Points_X', 'Points_Y', 'Points_Z', 'vtkValidPointMask']

# Properties modified on plotOverLine2Display
plotOverLine2Display.SeriesVisibility = []

# Properties modified on plotOverLine2Display
plotOverLine2Display.SeriesVisibility = ['angle']

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1412, 754)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 670.0]
renderView1.CameraParallelScale = 141.4213562373095


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------