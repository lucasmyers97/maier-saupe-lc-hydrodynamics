import numpy as np

# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(label="Double Helix Filter")
@smproperty.input(name="Input")
class HalfVFilter(VTKPythonAlgorithmBase):
    # the rest of the code here is unchanged.
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, outputType='vtkUnstructuredGrid')
        self.distance = 1.0
        self.twist_wavenumber = 0.0

    def RequestData(self, request, inInfo, outInfo):
        # get the first input.
        input0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        d = self.distance
        alpha = self.twist_wavenumber
        
        p = input0.GetPoints()

        dist = np.minimum( np.sqrt( (p[:, 1] - d * np.cos(alpha * p[:, 0]))**2
                               + (p[:, 2] - d * np.sin(alpha * p[:, 0]))**2 ),
                       np.sqrt( (p[:, 1] + d * np.cos(alpha * p[:, 0]))**2
                               + (p[:, 2] + d * np.sin(alpha * p[:, 0]))**2 )
                         )

        # add to output
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.ShallowCopy(input0.VTKObject)
        output.PointData.append(dist, "dist");
        return 1

    @smproperty.doublevector(name="distance", default_values=1.0)
    def SetDistance(self, x):
        self.distance = x
        self.Modified()

    @smproperty.doublevector(name="twist_wavenumber", default_values=0.0)
    def SetTwistWavenumber(self, x):
        self.twist_wavenumber = x
        self.Modified()

