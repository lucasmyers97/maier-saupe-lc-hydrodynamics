"""
In cylindrical coordinates, gets angle that the director makes with the radial
vector. 
Note that one specifies which axis is the cylinder axis.
The radial vector is then perpendicular to the cylindrical axis.
"""
import numpy as np

# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(label="Director radial angle filter")
@smproperty.input(name="Input")
class DirectorRadialAngle(VTKPythonAlgorithmBase):
    # the rest of the code here is unchanged.
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, outputType='vtkUnstructuredGrid')
        self.input_name = 'n'
        self.axis = 'z'

    def RequestData(self, request, inInfo, outInfo):
        # get the first input.
        input0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        n = input0.PointData[self.input_name]
        p = input0.GetPoints()

        axis_list = ['x', 'y', 'z']
        axis_list.remove( self.axis )
        axis_idx_dict = {'x': 0, 'y': 1, 'z': 2}
        a = [axis_idx_dict[axis_list[0]], axis_idx_dict[axis_list[1]]]

        n_dot_p = n[:, a[0]] * p[:, a[0]] + n[:, a[1]] * p[:, a[1]]
        p_mag = np.sqrt( p[:, a[0]]**2 + p[:, a[1]]**2 )
        n_mag = np.sqrt( n[:, a[0]]**2 + n[:, a[1]]**2 )
        mag_prod = p_mag * n_mag
        mag_prod_zero = mag_prod == 0
        mag_prod[mag_prod_zero] = 1
        n_proj_p = n_dot_p / mag_prod

        # Make all positive, in case director is oriented incorrectly
        n_proj_p[n_proj_p < 0] *= -1

        # If projection is somehow greater than 1, just squish it to 1
        n_proj_p[n_proj_p > 1] = 1

        angle = np.arccos( n_proj_p )
        angle[mag_prod_zero] = np.nan

        # add to output
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.ShallowCopy(input0.VTKObject)
        output.PointData.append(angle, "angle");
        # output.PointData.append(n_dot_p, "angle");
        return 1

    @smproperty.stringvector(name="input name", default_values='n')
    def SetInputName(self, x):
        self.input_name = x
        self.Modified()

    @smproperty.stringvector(name="axis", default_values='z')
    def SetAxis(self, x):
        self.axis = x
        self.Modified()
