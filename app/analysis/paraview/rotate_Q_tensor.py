# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

import numpy as np

@smproxy.filter(name="Rotate Q Tensor")
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class RotateQTensor(VTKPythonAlgorithmBase):
    """ 
    Given a list of components Q0, ..., Q4 of the Q-tensor, return the same list
    but rotated by some amount about a given axis.
    """
    def __init__(self):
        self.rotation_axis = [1, 0, 0]
        self.rotation_angle = 0
        super().__init__(nInputPorts=1, nOutputPorts=1, outputType="vtkDataSet")

    def RequestDataObject(self, request, inInfo, outInfo):
        inData = self.GetInputData(inInfo, 0, 0)
        outData = self.GetOutputData(outInfo, 0)
        assert inData is not None

        if outData is None or (not outData.IsA(inData.GetClassName())):
            outData = inData.NewInstance()
            outInfo.GetInformationObject(0).Set(outData.DATA_OBJECT(), outData)

        return super().RequestDataObject(request, inInfo, outInfo)

    def RequestData(self, request, inInfo, outInfo):
        input0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        u = np.array(self.rotation_axis)
        u /= np.linalg.norm(u)
        theta = np.deg2rad(self.rotation_angle)

        u_tens_u = np.outer(u, u)
        I = np.eye(3)
        u_cross = np.cross(I, u)

        R = np.cos(theta) * I + np.sin(theta) * u_cross + (1 - np.cos(theta)) * u_tens_u

        Q0 = input0.PointData['Q0']
        Q1 = input0.PointData['Q1']
        Q2 = input0.PointData['Q2']
        Q3 = input0.PointData['Q3']
        Q4 = input0.PointData['Q4']
        
        Q_mat = np.zeros((Q0.shape[0], 3, 3))
        
        Q_mat[:, 0, 0] = Q0
        Q_mat[:, 0, 1] = Q1
        Q_mat[:, 0, 2] = Q2
        Q_mat[:, 1, 1] = Q3
        Q_mat[:, 1, 2] = Q4
        Q_mat[:, 1, 0] = Q_mat[:, 0, 1]
        Q_mat[:, 2, 0] = Q_mat[:, 0, 2]
        Q_mat[:, 2, 1] = Q_mat[:, 1, 2]
        Q_mat[:, 2, 2] = -(Q0 + Q3)

        rot_Q = np.zeros(Q_mat.shape)

        for i in range(Q_mat.shape[0]):
            rot_Q[i, :, :] = R @ Q_mat[i, :, :] @ R.transpose()

        Q0 = rot_Q[:, 0, 0]
        Q1 = rot_Q[:, 0, 1]
        Q2 = rot_Q[:, 0, 2]
        Q3 = rot_Q[:, 1, 1]
        Q4 = rot_Q[:, 1, 2]

        # add to output
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.ShallowCopy(input0.VTKObject)

        output.PointData.append(Q0, "Q0");
        output.PointData.append(Q1, "Q1");
        output.PointData.append(Q2, "Q2");
        output.PointData.append(Q3, "Q3");
        output.PointData.append(Q4, "Q4");
        return 1

    @smproperty.doublevector(name="rotation axis", default_values=[1, 0, 0])
    def SetRotationAxis(self, x, y, z):
        self.rotation_axis = [x, y, z]
        self.Modified()

    @smproperty.doublevector(name="rotation angle", default_values=0.0)
    def SetRotationAngle(self, x):
        self.rotation_angle = x
        self.Modified()
