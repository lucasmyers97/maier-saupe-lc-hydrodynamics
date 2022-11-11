import paraview.simple as ps

# Lets create a sphere
for i in range(100):
    sphere = ps.Sphere()
    ps.Show()
    ps.Render()
    ps.ResetSession()

# get active view
# renderView1 = GetActiveViewOrCreate('RenderView')
# renderView1.ViewSize = [1670, 1091]
# 
# # get display properties
# sphere1Display = GetDisplayProperties(sphere, view=renderView1)
# 
# # change solid color
# sphere1Display.AmbientColor = [0.0, 1.0, 0.0]
# sphere1Display.DiffuseColor = [0.0, 1.0, 0.0]
# 
# # save screenshot
# SaveScreenshot('greenSphereScreenshot.png', renderView1,
# ImageResolution=[1670, 1091])
