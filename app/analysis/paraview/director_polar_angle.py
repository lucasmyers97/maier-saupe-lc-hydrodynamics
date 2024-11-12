"""
In cylindrical coordinates, gets angle that the director makes with the
cylinder axis.
Assumes n is a unit vector.
"""
import numpy as np

# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(label="Director polar angle filter")
@smproperty.input(name="Input")
class DirectorPolarAngle(VTKPythonAlgorithmBase):
    # the rest of the code here is unchanged.
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, outputType='vtkUnstructuredGrid')
        self.input_name = 'n'
        self.axis = 'z'

    def RequestData(self, request, inInfo, outInfo):
        # get the first input.
        input0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        n = input0.PointData[self.input_name]

        axis_idx_dict = {'x': 0, 'y': 1, 'z': 2}
        i = axis_idx_dict[self.axis]

        n_dot_axis = n[:, i]

        # Make all positive, in case director is oriented incorrectly
        n_dot_axis[n_dot_axis < 0] *= -1

        # If projection is somehow greater than 1, just squish it to 1
        n_dot_axis[n_dot_axis > 1] = 1

        angle = np.arccos( n_dot_axis )

        # add to output
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.ShallowCopy(input0.VTKObject)
        output.PointData.append(angle, "angle");
        return 1

    @smproperty.stringvector(name="input name", default_values='n')
    def SetInputName(self, x):
        self.input_name = x
        self.Modified()

    @smproperty.stringvector(name="axis", default_values='z')
    def SetAxis(self, x):
        self.axis = x
        self.Modified()
