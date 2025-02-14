# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain
# to add a source, instead of a filter, use the `smproxy.source` decorator.
@smproxy.source(label="Python-based Superquadric Source Example")
class PythonSuperquadricSource(VTKPythonAlgorithmBase):
    """This is dummy VTKPythonAlgorithmBase subclass that
    simply puts out a Superquadric poly data using a vtkSuperquadricSource
    internally"""
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
                nInputPorts=0,
                nOutputPorts=1,
                outputType='vtkPolyData')
        from vtkmodules.vtkFiltersSources import vtkSuperquadricSource
        self._realAlgorithm = vtkSuperquadricSource()

    def RequestData(self, request, inInfo, outInfo):
        from vtkmodules.vtkCommonDataModel import vtkPolyData
        self._realAlgorithm.Update()
        output = vtkPolyData.GetData(outInfo, 0)
        output.ShallowCopy(self._realAlgorithm.GetOutput())
        return 1

    # for anything too complex or not yet supported, you can explicitly
    # provide the XML for the method.
    @smproperty.xml("""
        <DoubleVectorProperty name="Center"
            number_of_elements="3"
            default_values="0 0 0"
            command="SetCenter">
            <DoubleRangeDomain name="range" />
            <Documentation>Set center of the superquadric</Documentation>
        </DoubleVectorProperty>""")
    def SetCenter(self, x, y, z):
        self._realAlgorithm.SetCenter(x,y,z)
        self.Modified()

    # In most cases, one can simply use available decorators.
    @smproperty.doublevector(name="Scale", default_values=[1, 1, 1])
    @smdomain.doublerange()
    def SetScale(self, x, y, z):
        self._realAlgorithm.SetScale(x,y,z)
        self.Modified()

    @smproperty.intvector(name="ThetaResolution", default_values=16)
    def SetThetaResolution(self, x):
        self._realAlgorithm.SetThetaResolution(x)
        self.Modified()

    @smproperty.intvector(name="PhiResolution", default_values=16)
    @smdomain.intrange(min=0, max=1000)
    def SetPhiResolution(self, x):
        self._realAlgorithm.SetPhiResolution(x)
        self.Modified()

    @smproperty.doublevector(name="Thickness", default_values=0.3333)
    @smdomain.doublerange(min=1e-24, max=1.0)
    def SetThickness(self, x):
        self._realAlgorithm.SetThickness(x)
        self.Modified()
