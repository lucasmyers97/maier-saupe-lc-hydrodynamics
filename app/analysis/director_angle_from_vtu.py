import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import h5py

import paraview.simple as ps
import paraview.servermanager as psm
import paraview.vtk as vtk
from paraview.vtk.numpy_interface import dataset_adapter as dsa

def get_filenames():

    description = ("Read in nematic configuration from vtu and defect position"
                   "from hdf5 to plot director angle around defect")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--configuration_filename', 
                        dest='configuration_filename',
                        help='name of vtu file holding configuration')
    parser.add_argument('--defect_filename',
                        dest='defect_filename',
                        default='defect_positions.h5',
                        help='name of h5 file holding defect positions')
    parser.add_argument('--time',
                        dest='time',
                        type=float,
                        help='timestep associated with data configuration')
    args = parser.parse_args()

    configuration_filename = os.path.join(args.data_folder, 
                                          args.configuration_filename)
    defect_filename = os.path.join(args.data_folder,
                                   args.defect_filename)

    return configuration_filename, defect_filename, args.time



def make_points(center):

    n = 1000
    r = 1.5
    theta = np.linspace(0, np.pi, num=n)
    points = np.zeros((n, 3))
    points[:, 0] = r * np.cos(theta) + center[0]
    points[:, 1] = r * np.sin(theta) + center[1]

    return theta, points


def make_vtk_poly(points):

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])

    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    return vpoly



def sanitize_director_angle(phi):

    dphi = np.diff(phi)
    jump_down_indices = np.nonzero(dphi < (-np.pi/2))
    jump_up_indices = np.nonzero(dphi > (np.pi/2))

    new_phi = np.copy(phi)
    for jump_index in jump_down_indices:
        if jump_index.shape[0] == 0:
            continue
        new_phi[(jump_index[0] + 1):] += np.pi

    for jump_index in jump_up_indices:
        if jump_index.shape[0] == 0:
            continue
        new_phi[(jump_index[0] + 1):] -= np.pi

    return new_phi

def send_vtk_mesh_to_server(vtk_mesh):

    tp_mesh = ps.TrivialProducer(registrationName="tp_mesh")
    myMeshClient = tp_mesh.GetClientSideObject()
    myMeshClient.SetOutput(vtk_mesh)
    tp_mesh.UpdatePipeline()

    return tp_mesh



def main():

    configuration_filename, defect_filename, time = get_filenames()
    
    defect_file = h5py.File(defect_filename)
    t = defect_file['t'][:]
    x = defect_file['x'][:]
    y = defect_file['y'][:]

    plt.scatter(t, y)
    plt.show()
   
    time_idx = np.argmin( np.abs(t - time) )
    center = (x[time_idx], y[time_idx])
    print(center)

    theta, points = make_points(center)
    vpoly = make_vtk_poly(points)
    server_point_mesh = send_vtk_mesh_to_server(vpoly)

    reader = ps.OpenDataFile(configuration_filename)
    resampled_data = ps.ResampleWithDataset(registrationName='resampled_data', 
                                            SourceDataArrays=reader,
                                            DestinationMesh=server_point_mesh)

    data = psm.Fetch(resampled_data)
    data = dsa.WrapDataObject(data)
    print(data.Points)
    print(data.PointData['S'].shape)
    print(data.PointData['director'].shape)

    phi = np.arctan2(data.PointData['director'][:, 1],
                     data.PointData['director'][:, 0])

    new_phi = sanitize_director_angle(phi)
    # plt.plot(theta, phi)
    plt.plot(theta, new_phi)
    plt.show()

    # plt.scatter(data.Points[:, 0], data.Points[:, 1])
    # plt.show()

if __name__ == "__main__":

    main()
