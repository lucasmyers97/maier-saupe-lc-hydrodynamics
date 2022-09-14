import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
from matplotlib.animation import FuncAnimation
from scipy import interpolate

import h5py

import paraview.simple as ps
import paraview.servermanager as psm
import paraview.vtk as vtk
from paraview.vtk.numpy_interface import dataset_adapter as dsa

def get_vtu_files(folder, vtu_filename):
    """
    Takes in a folder where vtu files of the form `vtu_filename`#.pvtu live.
    Then it reads in the filenames and the #'s, and sorts the numbers and
    filenames in ascending order.

    Returns numpy array of filenames and times
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
    parser.add_argument('--dzyaloshinskii_filename',
                        dest='dzyaloshinskii_filename',
                        help='name of h5 file holding Dzyaloshinskii solution')
    parser.add_argument('--two_defect',
                        dest='two_defect',
                        type=int,
                        help='1 if positive two-defect, -1 if negative')
    args = parser.parse_args()

    vtu_filenames, times = get_vtu_files(args.data_folder, 
                                         args.configuration_filename)
    
    vtu_full_path = []
    for vtu_filename in vtu_filenames:
        vtu_full_path.append( os.path.join(args.data_folder, vtu_filename) )

    defect_filename = os.path.join(args.data_folder,
                                   args.defect_filename)
    dzyaloshinskii_filename = os.path.join(args.data_folder,
                                           args.dzyaloshinskii_filename)

    return vtu_full_path, defect_filename, dzyaloshinskii_filename, times, args.two_defect



def make_points(center, n, r):

    # theta = np.linspace(0, np.pi, num=n)
    theta = np.linspace(np.pi, 2 * np.pi, num=n)
    points = np.zeros((n, 3))
    points[:, 0] = r * np.cos(theta) + center[0]
    points[:, 1] = r * np.sin(theta) + center[1]

    return theta - np.pi, points



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



def get_phi_from_reader(reader, server_point_mesh):

    resampled_data = ps.ResampleWithDataset(registrationName='resampled_data', 
                                            SourceDataArrays=reader,
                                            DestinationMesh=server_point_mesh)

    data = psm.Fetch(resampled_data)
    data = dsa.WrapDataObject(data)
    phi = np.arctan2(data.PointData['director'][:, 1],
                     data.PointData['director'][:, 0])
    new_phi = sanitize_director_angle(phi)

    return new_phi


def main():

    n_points = 1000
    radius = 5

    vtu_filenames, defect_filename, dzyaloshinskii_filename, times, two_defect = get_filenames()
    print(two_defect)
    # n_times = times.shape[0]
    n_times = 75
   
    # get defect locations
    defect_file = h5py.File(defect_filename)
    t = defect_file['t'][:]
    x = defect_file['x'][:]
    y = defect_file['y'][:]

    if two_defect == 1:
        pos_x_idx = np.nonzero(x > 0)[0]
        t = t[pos_x_idx]
        x = x[pos_x_idx]
        y = y[pos_x_idx]
 
    # read in phi as a function of theta for each timestep
    phi_array = np.zeros((n_times, n_points))
    for idx in range(n_times):

        time_idx = np.argmin( np.abs(t - times[idx]) )
        center = (x[time_idx], y[time_idx])

        theta, points = make_points(center, n_points, radius)
        vpoly = make_vtk_poly(points)
        server_point_mesh = send_vtk_mesh_to_server(vpoly)

        reader = ps.OpenDataFile(vtu_filenames[idx])

        phi_array[idx, :] = get_phi_from_reader(reader, server_point_mesh)

    # read dzyaloshinskii solution
    dzyaloshinskii_file = h5py.File(dzyaloshinskii_filename)
    ref_phi = np.array(dzyaloshinskii_file['phi'][:])
    ref_theta = np.array(dzyaloshinskii_file['theta'][:])

    fig, ax = plt.subplots()
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'r', label="Defect relaxation")
    ax.plot(ref_theta, ref_phi, 'b', label="Dzyaloshinskii solution")
    ax.set_xlabel("polar angle")
    ax.set_ylabel("director angle")
    ax.set_title(r"$\phi$ vs. $\theta$ for $L_3 = 0.5, R = 15$")
    fig.tight_layout()
    plt.legend()
    
    def init():
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, np.pi / 2)
        return ln,
    
    def update(frame):
        ln.set_data(theta, phi_array[frame, :])
        time_text.set_text("time = {}".format(times[frame]))
        return ln,
    
    ani = FuncAnimation(fig, update, frames=np.arange(n_times),
                        init_func=init, blit=True)
    ani.save("dzyaloshinskii_movie.mp4")

    # plot Fourier components of difference
    dzyaloshinskii_interp = interpolate.interp1d(ref_theta, ref_phi, kind='cubic')

    # read in phi as a function of theta for each timestep
    delta_phi_array = np.zeros((n_times, n_points))
    delta_phi_array_fft = np.zeros(delta_phi_array.shape)
    for idx in range(n_times):

        delta_phi_array[idx, :] = phi_array[idx, :] - dzyaloshinskii_interp(theta)
        delta_phi_array_fft[idx, :] = np.fft.rfft(delta_phi_array[idx, :])

    plt.show()
    fig, ax = plt.subplots()
    plt.plot(theta, delta_phi_array_fft[0, :])
    plt.show()

    # plt.show()

    # idx_array = [0, 1, 3, 5, 20, 80, 99]
    # for idx in idx_array:
    #     plt.plot(theta, phi_array[idx, :], 
    #              label=r'$t = {}$'.format(times[idx]))

    # plt.legend()
    # plt.show()

if __name__ == "__main__":

    main()
