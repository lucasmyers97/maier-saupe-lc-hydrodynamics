"""
Module provides utility functions for working with paraview via the Python
interface.
"""
import os
import re

import paraview.vtk as vtk
import paraview.simple as ps
import paraview.servermanager as psm
from paraview.vtk.numpy_interface import dataset_adapter as dsa

import numpy as np

def get_vtu_files(folder, vtu_filename):
    """
    Takes in a folder where vtu files of the form `vtu_filename`#.pvtu live.
    Then it reads in the filenames and the #'s, and sorts the numbers and
    filenames in ascending order.

    Parameters
    ----------
    folder : string
        absolute path to folder where pvtu files live
    vtu_filename : string
        prefix for filenames of pvtu-files, e.g. for 
        nematic_configuration_600.pvtu `vtu_filename` 
        would be nematic_configuration_

    Returns
    -------
    vtu_filenames : ndarray
        numpy ndarray containing names of all #.pvtu files sorted by 
        ascending times
    times : ndarray
        numpy ndarray containing all of the times corresponding to the files
        -- sorted by ascending times
    """

    filenames = os.listdir(folder)

    pattern = vtu_filename + r'(\d*)\.pvtu'
    p = re.compile(pattern)

    vtu_filenames = []
    times = []
    for filename in filenames:
        matches = p.findall(filename)
        if matches:
            vtu_filenames.append(filename)
            times.append( int(matches[0]) )
        
    vtu_filenames = np.array(vtu_filenames)
    times = np.array(times)

    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    vtu_filenames = vtu_filenames[sorted_idx]

    return vtu_filenames, times



def make_vtk_poly(points):
    """
    Constructs vtkPolyData object out of `points` numpy array

    Parameters
    ----------
    points : ndarray
        shape (n, 3) ndarray whose rows are 3D points from which vtkPoly is 
        constructed

    Returns
    -------
    vpoly : paraview.vtk.vtkPolyData
        vtkPolyData object whose points are given by `points`
    """

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])

    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    return vpoly



def send_vtk_mesh_to_server(vtk_mesh, registrationName="tp_mesh"):

    tp_mesh = ps.TrivialProducer(registrationName=registrationName)
    myMeshClient = tp_mesh.GetClientSideObject()
    myMeshClient.SetOutput(vtk_mesh)
    tp_mesh.UpdatePipeline()

    return tp_mesh



def get_data_from_reader(reader, server_point_mesh, key_name):

    resampled_data = ps.ResampleWithDataset(registrationName='resampled_data', 
                                            SourceDataArrays=reader,
                                            DestinationMesh=server_point_mesh)

    data = psm.Fetch(resampled_data)
    data = dsa.WrapDataObject(data)

    return data.PointData[key_name]

