"""
Calculates elastic energy densities of Q-tenor configuration from degrees of
freedom
"""
import numpy as np

# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain
import paraview.vtk.numpy_interface.algorithms as algs

@smproxy.filter(label="Elastic energy")
@smproperty.input(name="Input")
class DirectorRadialAngle(VTKPythonAlgorithmBase):
    # the rest of the code here is unchanged.
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, outputType='vtkUnstructuredGrid')
        self.L2 = 0.0
        self.L3 = 0.0

    def RequestData(self, request, inInfo, outInfo):
        # get the first input.
        input0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        Q0 = input0.PointData['Q0']
        Q1 = input0.PointData['Q1']
        Q2 = input0.PointData['Q2']
        Q3 = input0.PointData['Q3']
        Q4 = input0.PointData['Q4']
        
        dQ0 = algs.gradient(Q0)
        dQ1 = algs.gradient(Q1)
        dQ2 = algs.gradient(Q2)
        dQ3 = algs.gradient(Q3)
        dQ4 = algs.gradient(Q4)
        
        
        L1_term = (
                0.5*(-dQ0[:, 0] - dQ3[:, 0]) 
                * (-dQ0[:, 0] - dQ3[:, 0]) 
                + 0.5*(-dQ0[:, 1] - dQ3[:, 1]) 
                * (-dQ0[:, 1] - dQ3[:, 1]) 
                + 0.5*(dQ0[:, 0]) 
                * (dQ0[:, 0]) 
                + 0.5*(dQ0[:, 1]) * (dQ0[:, 1]) 
                + (dQ1[:, 0]) * (dQ1[:, 0]) 
                + (dQ1[:, 1]) * (dQ1[:, 1]) 
                + (dQ2[:, 0]) * (dQ2[:, 0]) 
                + (dQ2[:, 1]) * (dQ2[:, 1]) 
                + 0.5*(dQ3[:, 0]) * (dQ3[:, 0]) 
                + 0.5*(dQ3[:, 1]) * (dQ3[:, 1]) 
                + (dQ4[:, 0]) * (dQ4[:, 0]) 
                + (dQ4[:, 1]) * (dQ4[:, 1])
                )
        
        L2_term = (
                0.5*self.L2
                * ((dQ0[:, 0] + dQ1[:, 1]) * (dQ0[:, 0] + dQ1[:, 1]) 
                   + (dQ1[:, 0] + dQ3[:, 1]) * (dQ1[:, 0] + dQ3[:, 1]) 
                   + (dQ2[:, 0] + dQ4[:, 1]) * (dQ2[:, 0] + dQ4[:, 1]))
                )
        
        
        L3_term = (
                0.5*self.L3
                *(2*((-dQ0[:, 0] - dQ3[:, 0])*(-dQ0[:, 1] - dQ3[:, 1]) 
                     + dQ0[:, 0]*dQ0[:, 1] + 2*dQ1[:, 0]*dQ1[:, 1] 
                     + 2*dQ2[:, 0]*dQ2[:, 1] + dQ3[:, 0]*dQ3[:, 1] 
                     + 2*dQ4[:, 0]*dQ4[:, 1])*Q1 
                  + ((-dQ0[:, 0] - dQ3[:, 0]) 
                     * (-dQ0[:, 0] - dQ3[:, 0]) 
                     + (dQ0[:, 0]) * (dQ0[:, 0]) 
                     + 2*(dQ1[:, 0]) * (dQ1[:, 0]) 
                     + 2*(dQ2[:, 0]) * (dQ2[:, 0]) 
                     + (dQ3[:, 0]) * (dQ3[:, 0]) 
                     + 2*(dQ4[:, 0]) * (dQ4[:, 0]))*Q0 
                  + ((-dQ0[:, 1] - dQ3[:, 1]) 
                     * (-dQ0[:, 1] - dQ3[:, 1]) 
                     + (dQ0[:, 1]) * (dQ0[:, 1]) 
                     + 2*(dQ1[:, 1]) * (dQ1[:, 1]) 
                     + 2*(dQ2[:, 1]) * (dQ2[:, 1]) 
                     + (dQ3[:, 1]) * (dQ3[:, 1]) 
                     + 2*(dQ4[:, 1]) * (dQ4[:, 1]))*Q3)
                  )

        # add to output
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.ShallowCopy(input0.VTKObject)
        output.PointData.append(L1_term, "L1_term")
        output.PointData.append(L2_term, "L2_term")
        output.PointData.append(L3_term, "L3_term")
        output.PointData.append(L1_term + L2_term + L3_term, "total elastic energy")

        return 1

    @smproperty.doublevector(name='L2', default_values=0.0)
    def SetInputName(self, x):
        self.L2 = x
        self.Modified()

    @smproperty.doublevector(name='L3', default_values=0.0)
    def SetAxis(self, x):
        self.L3 = x
        self.Modified()
