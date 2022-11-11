import paraview.simple as ps
from ..analysis.utilities import paraview as pu

filename = "/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/small-one-defect-anisotropic-circular/Q_components_small_single_defect_36.pvtu"
reader = ps.OpenDataFile(filename)

view = ps.Show(reader)
ps.ColorBy(view, ('POINTS', 'Q0'))
view.RescaleTransferFunctionToDataRange(True)

source = ps.GetActiveSource()
view = ps.GetActiveView()
display = ps.GetDisplayProperties(source, view)
display.SetScalarBarVisibility(view, True)

view.ViewSize = [500, 500]
render_view = ps.Render()
ps.SaveScreenshot("test_image.png", magnification=5, quality=100, view=render_view)
