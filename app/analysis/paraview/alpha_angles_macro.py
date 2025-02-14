# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
q_components_nematic_configuration_5000pvtu = FindSource('Q_components_nematic_configuration_5000.pvtu')

# create a new 'Q tensor eigensystem'
qtensoreigensystem1 = Qtensoreigensystem(registrationName='Qtensoreigensystem1', Input=q_components_nematic_configuration_5000pvtu)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
qtensoreigensystem1Display = Show(qtensoreigensystem1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
qtensoreigensystem1Display.Representation = 'Surface'
qtensoreigensystem1Display.ColorArrayName = [None, '']
qtensoreigensystem1Display.SelectTCoordArray = 'None'
qtensoreigensystem1Display.SelectNormalArray = 'None'
qtensoreigensystem1Display.SelectTangentArray = 'None'
qtensoreigensystem1Display.OSPRayScaleArray = 'P'
qtensoreigensystem1Display.OSPRayScaleFunction = 'PiecewiseFunction'
qtensoreigensystem1Display.SelectOrientationVectors = 'None'
qtensoreigensystem1Display.ScaleFactor = 20.0
qtensoreigensystem1Display.SelectScaleArray = 'None'
qtensoreigensystem1Display.GlyphType = 'Arrow'
qtensoreigensystem1Display.GlyphTableIndexArray = 'None'
qtensoreigensystem1Display.GaussianRadius = 1.0
qtensoreigensystem1Display.SetScaleArray = ['POINTS', 'P']
qtensoreigensystem1Display.ScaleTransferFunction = 'PiecewiseFunction'
qtensoreigensystem1Display.OpacityArray = ['POINTS', 'P']
qtensoreigensystem1Display.OpacityTransferFunction = 'PiecewiseFunction'
qtensoreigensystem1Display.DataAxesGrid = 'GridAxesRepresentation'
qtensoreigensystem1Display.PolarAxes = 'PolarAxesRepresentation'
qtensoreigensystem1Display.ScalarOpacityUnitDistance = 30.885733526356635
qtensoreigensystem1Display.OpacityArrayName = ['POINTS', 'P']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
qtensoreigensystem1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.0005216065904541933, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
qtensoreigensystem1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.0005216065904541933, 1.0, 0.5, 0.0]

# hide data in view
Hide(q_components_nematic_configuration_5000pvtu, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Director radial angle filter'
directorradialanglefilter1 = Directorradialanglefilter(registrationName='Directorradialanglefilter1', Input=qtensoreigensystem1)

# show data in view
directorradialanglefilter1Display = Show(directorradialanglefilter1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
directorradialanglefilter1Display.Representation = 'Surface'
directorradialanglefilter1Display.ColorArrayName = [None, '']
directorradialanglefilter1Display.SelectTCoordArray = 'None'
directorradialanglefilter1Display.SelectNormalArray = 'None'
directorradialanglefilter1Display.SelectTangentArray = 'None'
directorradialanglefilter1Display.OSPRayScaleArray = 'P'
directorradialanglefilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
directorradialanglefilter1Display.SelectOrientationVectors = 'None'
directorradialanglefilter1Display.ScaleFactor = 20.0
directorradialanglefilter1Display.SelectScaleArray = 'None'
directorradialanglefilter1Display.GlyphType = 'Arrow'
directorradialanglefilter1Display.GlyphTableIndexArray = 'None'
directorradialanglefilter1Display.GaussianRadius = 1.0
directorradialanglefilter1Display.SetScaleArray = ['POINTS', 'P']
directorradialanglefilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
directorradialanglefilter1Display.OpacityArray = ['POINTS', 'P']
directorradialanglefilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
directorradialanglefilter1Display.DataAxesGrid = 'GridAxesRepresentation'
directorradialanglefilter1Display.PolarAxes = 'PolarAxesRepresentation'
directorradialanglefilter1Display.ScalarOpacityUnitDistance = 30.885733526356635
directorradialanglefilter1Display.OpacityArrayName = ['POINTS', 'P']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
directorradialanglefilter1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.0005216065904541933, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
directorradialanglefilter1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.0005216065904541933, 1.0, 0.5, 0.0]

# hide data in view
Hide(qtensoreigensystem1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=directorradialanglefilter1)
plotOverLine1.Point1 = [-100.0, -100.0, 0.0]
plotOverLine1.Point2 = [100.0, 100.0, 0.0]

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [-100.0, 0.0, 0.0]
plotOverLine1.Point2 = [100.0, 0.0, 0.0]

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'
plotOverLine1Display.ColorArrayName = [None, '']
plotOverLine1Display.SelectTCoordArray = 'None'
plotOverLine1Display.SelectNormalArray = 'None'
plotOverLine1Display.SelectTangentArray = 'None'
plotOverLine1Display.OSPRayScaleArray = 'P'
plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine1Display.SelectOrientationVectors = 'None'
plotOverLine1Display.ScaleFactor = 20.0
plotOverLine1Display.SelectScaleArray = 'None'
plotOverLine1Display.GlyphType = 'Arrow'
plotOverLine1Display.GlyphTableIndexArray = 'None'
plotOverLine1Display.GaussianRadius = 1.0
plotOverLine1Display.SetScaleArray = ['POINTS', 'P']
plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.OpacityArray = ['POINTS', 'P']
plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display.ScaleTransferFunction.Points = [-1.846088583003515e-14, 0.0, 0.5, 0.0, 0.00043651920615922463, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display.OpacityTransferFunction.Points = [-1.846088583003515e-14, 0.0, 0.5, 0.0, 0.00043651920615922463, 1.0, 0.5, 0.0]

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')

# trace defaults for the display properties.
plotOverLine1Display_1.UseIndexForXAxis = 0
plotOverLine1Display_1.XArrayName = 'arc_length'
plotOverLine1Display_1.SeriesVisibility = ['angle', 'm_Magnitude', 'n_Magnitude', 'P', 'q1', 'q2', 'S', 'S - P']
plotOverLine1Display_1.SeriesLabel = ['angle', 'angle', 'arc_length', 'arc_length', 'm_X', 'm_X', 'm_Y', 'm_Y', 'm_Z', 'm_Z', 'm_Magnitude', 'm_Magnitude', 'n_X', 'n_X', 'n_Y', 'n_Y', 'n_Z', 'n_Z', 'n_Magnitude', 'n_Magnitude', 'P', 'P', 'q1', 'q1', 'q2', 'q2', 'S', 'S', 'S - P', 'S - P', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display_1.SeriesColor = ['angle', '0', '0', '0', 'arc_length', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'm_X', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'm_Y', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'm_Z', '0.6', '0.3100022888532845', '0.6399938963912413', 'm_Magnitude', '1', '0.5000076295109483', '0', 'n_X', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867', 'n_Y', '0', '0', '0', 'n_Z', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'n_Magnitude', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'P', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'q1', '0.6', '0.3100022888532845', '0.6399938963912413', 'q2', '1', '0.5000076295109483', '0', 'S', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867', 'S - P', '0', '0', '0', 'vtkValidPointMask', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'Points_X', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'Points_Y', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'Points_Z', '0.6', '0.3100022888532845', '0.6399938963912413', 'Points_Magnitude', '1', '0.5000076295109483', '0']
plotOverLine1Display_1.SeriesPlotCorner = ['angle', '0', 'arc_length', '0', 'm_X', '0', 'm_Y', '0', 'm_Z', '0', 'm_Magnitude', '0', 'n_X', '0', 'n_Y', '0', 'n_Z', '0', 'n_Magnitude', '0', 'P', '0', 'q1', '0', 'q2', '0', 'S', '0', 'S - P', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesLabelPrefix = ''
plotOverLine1Display_1.SeriesLineStyle = ['angle', '1', 'arc_length', '1', 'm_X', '1', 'm_Y', '1', 'm_Z', '1', 'm_Magnitude', '1', 'n_X', '1', 'n_Y', '1', 'n_Z', '1', 'n_Magnitude', '1', 'P', '1', 'q1', '1', 'q2', '1', 'S', '1', 'S - P', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesLineThickness = ['angle', '2', 'arc_length', '2', 'm_X', '2', 'm_Y', '2', 'm_Z', '2', 'm_Magnitude', '2', 'n_X', '2', 'n_Y', '2', 'n_Z', '2', 'n_Magnitude', '2', 'P', '2', 'q1', '2', 'q2', '2', 'S', '2', 'S - P', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['angle', '0', 'arc_length', '0', 'm_X', '0', 'm_Y', '0', 'm_Z', '0', 'm_Magnitude', '0', 'n_X', '0', 'n_Y', '0', 'n_Z', '0', 'n_Magnitude', '0', 'P', '0', 'q1', '0', 'q2', '0', 'S', '0', 'S - P', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesMarkerSize = ['angle', '4', 'arc_length', '4', 'm_X', '4', 'm_Y', '4', 'm_Z', '4', 'm_Magnitude', '4', 'n_X', '4', 'n_Y', '4', 'n_Z', '4', 'n_Magnitude', '4', 'P', '4', 'q1', '4', 'q2', '4', 'S', '4', 'S - P', '4', 'vtkValidPointMask', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'Points_Magnitude', '4']

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesPlotCorner = ['P', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'S', '0', 'S - P', '0', 'angle', '0', 'arc_length', '0', 'm_Magnitude', '0', 'm_X', '0', 'm_Y', '0', 'm_Z', '0', 'n_Magnitude', '0', 'n_X', '0', 'n_Y', '0', 'n_Z', '0', 'q1', '0', 'q2', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesLineStyle = ['P', '1', 'Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'S', '1', 'S - P', '1', 'angle', '1', 'arc_length', '1', 'm_Magnitude', '1', 'm_X', '1', 'm_Y', '1', 'm_Z', '1', 'n_Magnitude', '1', 'n_X', '1', 'n_Y', '1', 'n_Z', '1', 'q1', '1', 'q2', '1', 'vtkValidPointMask', '1']
plotOverLine1Display_1.SeriesLineThickness = ['P', '2', 'Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'S', '2', 'S - P', '2', 'angle', '2', 'arc_length', '2', 'm_Magnitude', '2', 'm_X', '2', 'm_Y', '2', 'm_Z', '2', 'n_Magnitude', '2', 'n_X', '2', 'n_Y', '2', 'n_Z', '2', 'q1', '2', 'q2', '2', 'vtkValidPointMask', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['P', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'S', '0', 'S - P', '0', 'angle', '0', 'arc_length', '0', 'm_Magnitude', '0', 'm_X', '0', 'm_Y', '0', 'm_Z', '0', 'n_Magnitude', '0', 'n_X', '0', 'n_Y', '0', 'n_Z', '0', 'q1', '0', 'q2', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesMarkerSize = ['P', '4', 'Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'S', '4', 'S - P', '4', 'angle', '4', 'arc_length', '4', 'm_Magnitude', '4', 'm_X', '4', 'm_Y', '4', 'm_Z', '4', 'n_Magnitude', '4', 'n_X', '4', 'n_Y', '4', 'n_Z', '4', 'q1', '4', 'q2', '4', 'vtkValidPointMask', '4']

# update the view to ensure updated data information
lineChartView1.Update()

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['angle', 'arc_length', 'm_Magnitude', 'm_X', 'm_Y', 'm_Z', 'n_Magnitude', 'n_X', 'n_Y', 'n_Z', 'P', 'Points_Magnitude', 'Points_X', 'Points_Y', 'Points_Z', 'q1', 'q2', 'S', 'S - P', 'vtkValidPointMask']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = []

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['angle']

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
