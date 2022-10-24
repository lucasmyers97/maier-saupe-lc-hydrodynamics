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
import h5py

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
    
    Q0 = inputs[0].PointData["Q0"]
    Q1 = inputs[0].PointData["Q1"]
    Q2 = inputs[0].PointData["Q2"]
    Q3 = inputs[0].PointData["Q3"]
    Q4 = inputs[0].PointData["Q4"]
    
    Q_mat = np.zeros((3, 3, Q0.shape[0]))
    
    Q_mat[0, 0, :] = Q0
    Q_mat[0, 1, :] = Q1
    Q_mat[0, 2, :] = Q2
    Q_mat[1, 1, :] = Q3
    Q_mat[1, 2, :] = Q4
    Q_mat[1, 0, :] = Q_mat[0, 1, :]
    Q_mat[2, 0, :] = Q_mat[0, 2, :]
    Q_mat[2, 1, :] = Q_mat[1, 2, :]
    
    q1 = np.zeros(Q0.shape)
    q2 = np.zeros(Q0.shape)
    n = np.zeros((Q0.shape[0], 3))
    m = np.zeros((Q0.shape[0], 3))
    
    for i in range(q1.shape[0]):
        w, v = np.linalg.eig(Q_mat[:, :, i])
        w_idx = np.argsort(w)
        q1[i] = w[w_idx[-1]]
        q2[i] = w[w_idx[-2]]
        n[i, :] = v[:, w_idx[-1]]
        m[i, :] = v[:, w_idx[-2]]
    
    output.PointData.append(q1, "q1")
    output.PointData.append(q2, "q2")
    output.PointData.append(n, "n")
    output.PointData.append(m, "m")
    """

    return programmable_filter



def generate_sample_points(r0, rmax, point_dims, defect_center):

    r = np.linspace(r0, rmax, num=point_dims[1])
    theta = np.linspace(0, 2*np.pi, num=point_dims[0], endpoint=False)
    
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta) + defect_center[0]
    Y = R * np.sin(Theta) + defect_center[1]
    Z = np.zeros(X.shape)
    
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    
    poly_points = ps.PolyPointSource()
    poly_points.Points = points.transpose().flatten()

    return poly_points, r, theta



def write_polydata_to_hdf5(resampled_data, hdf5_filename, r, theta, point_dims, defect_center, timestep):

    with h5py.File(hdf5_filename, "a") as f:
        grp = f.create_group( "timestep_{:d}".format(timestep) )
        grp.create_dataset("point_dims", data=point_dims)
        grp.create_dataset("r", data=r)
        grp.create_dataset("theta", data=theta)
        grp.create_dataset("defect_center", data=defect_center)

    hdf5_filter = ps.ProgrammableFilter(Input=resampled_data)
    hdf5_filter.Script = """
    import numpy as np
    
    from mpi4py import MPI
    import h5py
    
    comm = MPI.COMM_WORLD
    
    q1 = inputs[0].PointData["q1"]
    q2 = inputs[0].PointData["q2"]
    m = inputs[0].PointData["m"]
    n = inputs[0].PointData["n"]
    
    # points = inputs[0].GetPoints()
    
    num = np.array(q1.shape[0], dtype='i')
    num_g = np.zeros(comm.Get_size(), dtype='i')
    comm.Allgather([num, MPI.INT],
                    [num_g, MPI.INT])
    assert np.sum(num_g) == num_g[0]
    
    if comm.Get_rank() == 0:
        with h5py.File("{}", "a") as f:
            grp = f["timestep_{:d}"]
            grp.create_dataset("q1", data=q1)
            grp.create_dataset("q2", data=q2)
            grp.create_dataset("n", data=n)
            grp.create_dataset("m", data=m)
    #        grp.create_dataset("points", data=points)
    """.format(hdf5_filename, timestep)

    return hdf5_filter



def write_two_defect_polydata_to_hdf5(resampled_data, hdf5_filename, grp_name, r, theta, point_dims, defect_center, timestep):

    with h5py.File(hdf5_filename, "a") as f:
        grp = f[grp_name]
        new_grp = grp.create_group( "timestep_{:d}".format(timestep) )
        new_grp.create_dataset("point_dims", data=point_dims)
        new_grp.create_dataset("r", data=r)
        new_grp.create_dataset("theta", data=theta)
        new_grp.create_dataset("defect_center", data=defect_center)

    hdf5_filter = ps.ProgrammableFilter(Input=resampled_data)
    hdf5_filter.Script = """
    import numpy as np
    
    from mpi4py import MPI
    import h5py
    
    comm = MPI.COMM_WORLD
    
    q1 = inputs[0].PointData["q1"]
    q2 = inputs[0].PointData["q2"]
    m = inputs[0].PointData["m"]
    n = inputs[0].PointData["n"]
    
    # points = inputs[0].GetPoints()
    
    num = np.array(q1.shape[0], dtype='i')
    num_g = np.zeros(comm.Get_size(), dtype='i')
    comm.Allgather([num, MPI.INT],
                    [num_g, MPI.INT])
    assert np.sum(num_g) == num_g[0]
    
    if comm.Get_rank() == 0:
        with h5py.File("{}", "a") as f:
            grp = f["{}"]
            new_grp = grp["timestep_{:d}"]
            new_grp.create_dataset("q1", data=q1)
            new_grp.create_dataset("q2", data=q2)
            new_grp.create_dataset("n", data=n)
            new_grp.create_dataset("m", data=m)
    #        grp.create_dataset("points", data=points)
    """.format(hdf5_filename, grp_name, timestep)

    return hdf5_filter
