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

    pattern = r'^' + vtu_filename + r'(\d*)\.pvtu'
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



def get_eigenvalue_programmable_filter(Q_configuration):

    # Calculate eigenvectors and eigenvalues
    programmable_filter = ps.ProgrammableFilter(Input=Q_configuration)
    programmable_filter.Script = """
    import paraview.vtk.numpy_interface.algorithms as algs
    import numpy as np
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    Q0 = inputs[0].PointData[\'Q0\']
    Q1 = inputs[0].PointData[\'Q1\']
    Q2 = inputs[0].PointData[\'Q2\']
    Q3 = inputs[0].PointData[\'Q3\']
    Q4 = inputs[0].PointData[\'Q4\']
    
    Q_mat = np.zeros((3, 3, Q0.shape[0]))
    
    Q_mat[0, 0, :] = Q0
    Q_mat[0, 1, :] = Q1
    Q_mat[0, 2, :] = Q2
    Q_mat[1, 1, :] = Q3
    Q_mat[1, 2, :] = Q4
    Q_mat[1, 0, :] = Q_mat[0, 1, :]
    Q_mat[2, 0, :] = Q_mat[0, 2, :]
    Q_mat[2, 1, :] = Q_mat[1, 2, :]
    
    S = np.zeros(Q0.shape)
    P = np.zeros(Q0.shape)
    n = np.zeros((Q0.shape[0], 3))
    m = np.zeros((Q0.shape[0], 3))
    
    for i in range(S.shape[0]):
        w, v = np.linalg.eig(Q_mat[:, :, i])
        w_idx = np.argsort(w)
        S[i] = w[w_idx[-1]]
        P[i] = w[w_idx[-2]]
        n[i, :] = v[:, w_idx[-1]]
        m[i, :] = v[:, w_idx[-2]]
    
    output.PointData.append(S, "S")
    output.PointData.append(P, "P")
    output.PointData.append(n, "n")
    output.PointData.append(m, "m")
    """

    return programmable_filter



def generate_sample_points(r0, rmax, n, m):

    theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    r = np.linspace(r0, rmax, m)
    
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = np.zeros(X.shape)
    
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    
    poly_points = ps.PolyPointSource()
    poly_points.Points = points.transpose().flatten()

    return poly_points



def write_polydata_to_hdf5(resampled_data):

    hdf5_filter = ps.ProgrammableFilter(Input=resampled_data)
    hdf5_filter.Script = """
    import numpy as np
    
    from mpi4py import MPI
    import h5py
    
    comm = MPI.COMM_WORLD
    
    S = inputs[0].PointData["S"]
    P = inputs[0].PointData["P"]
    m = inputs[0].PointData["m"]
    n = inputs[0].PointData["n"]
    
    points = inputs[0].GetPoints()
    
    num = np.array(S.shape[0], dtype='i')
    num_g = np.zeros(comm.Get_size(), dtype='i')
    comm.Allgather([num, MPI.INT],
                    [num_g, MPI.INT])
    assert np.sum(num_g) == num_g[0]
    
    if comm.Get_rank() == 0:
        with h5py.File("single_defect_core.h5", "w") as f:
            f.create_dataset("S", data=S)
            f.create_dataset("P", data=P)
            f.create_dataset("n", data=n)
            f.create_dataset("m", data=m)
            f.create_dataset("points", data=points)
    """

    return hdf5_filter
