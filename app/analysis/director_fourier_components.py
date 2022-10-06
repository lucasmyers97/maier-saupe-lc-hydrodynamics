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

from utilities import paraview as pvu
from utilities import nematics as nu


def get_commandline_args():

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

    vtu_filenames, times = pvu.get_vtu_files(args.data_folder, 
                                             args.configuration_filename)
    
    vtu_full_path = []
    for vtu_filename in vtu_filenames:
        vtu_full_path.append( os.path.join(args.data_folder, vtu_filename) )

    defect_filename = os.path.join(args.data_folder,
                                   args.defect_filename)
    dzyaloshinskii_filename = os.path.join(args.data_folder,
                                           args.dzyaloshinskii_filename)

    return (vtu_full_path, defect_filename, 
            dzyaloshinskii_filename, times, args.two_defect)



def make_points(center, n, r):

    theta = np.linspace(np.pi, 2 * np.pi, num=n)
    # theta = np.linspace(0, 2 * np.pi, num=n)
    points = np.zeros((n, 3))
    points[:, 0] = r * np.cos(theta) + center[0]
    points[:, 1] = r * np.sin(theta) + center[1]

    return theta - np.pi, points



def main():

    n_points = 1000
    radius = 5
    R0 = 38

    (vtu_filenames, defect_filename, 
     dzyaloshinskii_filename, times, two_defect) = get_commandline_args()
    print(vtu_filenames)
   
    # get defect locations
    defect_file = h5py.File(defect_filename)
    t = defect_file['t'][:]
    x = defect_file['x'][:]
    y = defect_file['y'][:]
    charge = defect_file['charge'][:]

    pos_idx = np.nonzero(charge > 0)[0]
    neg_idx = np.nonzero(charge < 0)[0]
    pos_t = t[pos_idx]
    neg_t = t[neg_idx]

    neg_t_idx = []
    pos_t_idx = []
    for i in range(len(neg_t)):
        match_idx = np.where(pos_t == neg_t[i])[0]
        if len(match_idx) != 0:
            neg_t_idx.append(i)
            pos_t_idx.append(match_idx[0])

    defect_dist = x[pos_idx][pos_t_idx] - x[neg_idx][neg_t_idx]
    proper_dist_idx = np.argmin(defect_dist - R0)
    plt.plot(t[pos_idx][pos_t_idx], proper_dist_idx)
    plt.show()
    t0 = pos_t[pos_t_idx][proper_dist_idx]
    x0 = x[pos_idx][pos_t_idx][proper_dist_idx]
    y0 = y[pos_idx][pos_t_idx][proper_dist_idx]

    # if two_defect == 1:
    #     pos_x_idx = np.nonzero(charge > 0)[0]
    #     t = t[pos_x_idx]
    #     x = x[pos_x_idx]
    #     y = y[pos_x_idx]
 
    # read in phi as a function of theta for each timestep
    phi_array = np.zeros((n_points))
    time_idx = np.argmin( np.abs(t0 - times) )
    print(t0)
    print(times)
    print(times.shape)
    print(time_idx)
    center = (x0, y0)

    theta, points = make_points(center, n_points, radius)
    vpoly = pvu.make_vtk_poly(points)
    server_point_mesh = pvu.send_vtk_mesh_to_server(vpoly)

    reader = ps.OpenDataFile(vtu_filenames[time_idx])

    n = pvu.get_data_from_reader(reader, server_point_mesh, 'director')
    phi = nu.director_to_angle(n)
    phi_array = nu.sanitize_director_angle(phi)

    # # read dzyaloshinskii solution
    # dzyaloshinskii_file = h5py.File(dzyaloshinskii_filename)
    # ref_phi = np.array(dzyaloshinskii_file['phi'][:])
    # ref_theta = np.array(dzyaloshinskii_file['theta'][:])

    # fig, ax = plt.subplots()
    # time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    # xdata, ydata = [], []
    # ln, = ax.plot([], [], 'r', label="Defect relaxation")
    # ax.plot(ref_theta, ref_phi, 'b', label="Dzyaloshinskii solution")
    # ax.set_xlabel("polar angle")
    # ax.set_ylabel("director angle")
    # ax.set_title(r"$\phi$ vs. $\theta$ for $+1/2$ defect, $\epsilon = -0.5$")
    # fig.tight_layout()
    # plt.legend()
    # 
    # def init():
    #     ax.set_xlim(0, np.pi)
    #     ax.set_ylim(0, np.pi / 2)
    #     return ln,
    # 
    # def update(frame):
    #     ln.set_data(theta, phi_array[frame, :])
    #     time_text.set_text("time = {}".format(times[frame]))
    #     return ln,
    # 
    # ani = FuncAnimation(fig, update, frames=np.arange(1, n_times),
    #                     init_func=init, blit=True)
    # ani.save("dzyaloshinskii_movie.mp4")
    # plt.show()

    # # plot Fourier components of difference
    # dzyaloshinskii_interp = interpolate.interp1d(ref_theta, ref_phi, kind='cubic')

    # # read in phi as a function of theta for each timestep
    # delta_phi_array = np.zeros((n_times, n_points))
    # n_fft = int(n_points / 2) + 1 if (n_points % 2 == 0) else int((n_points + 1) / 2)
    # fourier_modes = np.arange(n_fft)
    # delta_phi_array_fft = np.zeros((n_times, n_fft), dtype=np.cfloat)
    # theta_freq = np.fft.rfftfreq(theta.size)
    # for idx in range(1, n_times):

    #     delta_phi_array[idx, :] = phi_array[idx, :] - dzyaloshinskii_interp(theta)
    #     delta_phi_array_fft[idx, :] = np.fft.rfft(delta_phi_array[idx, :])

    # fig, ax = plt.subplots()
    # time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    # ln, = ax.plot([], [], 'r')
    # ax.set_xlabel("Cosine mode number")
    # ax.set_ylabel("Amplitude")
    # ax.set_title(r"FT of $\phi - \phi_\text{ref}$ for $\epsilon = -0.5$")
    # fig.tight_layout()
    # # plt.legend()

    # # min_fourier = np.min(delta_phi_array_fft.imag, (0, 1))
    # # max_fourier = np.max(delta_phi_array_fft.imag, (0, 1))
    # min_fourier = np.min(delta_phi_array_fft.real, (0, 1))
    # max_fourier = np.max(delta_phi_array_fft.real, (0, 1))
    #  
    # def init():
    #     ax.set_xlim(-2, 10)
    #     # ax.set_ylim(-5, 5)
    #     ax.set_ylim(min_fourier, max_fourier)
    #     # ax.set_ylim(-max_fourier, -min_fourier)
    #     return ln,
    # 
    # def update(frame):
    #     ln.set_data(2 * fourier_modes, delta_phi_array_fft[frame, :].real)
    #     time_text.set_text("time = {}".format(times[frame]))
    #     return ln,
    # 
    # ani = FuncAnimation(fig, update, frames=np.arange(1, n_times),
    #                     init_func=init, blit=True)
    # ani.save("dzyaloshinskii_fourier_movie.mp4")
    # plt.show()

    # plt.show()
    # fig, ax = plt.subplots()
    # # plt.plot(theta_freq, delta_phi_array_fft[0, :].imag)
    # plt.plot(delta_phi_array_fft[0, :].imag)
    # plt.show()

    # plt.show()

    # idx_array = [0, 1, 3, 5, 20, 80, 99]
    # for idx in idx_array:
    #     plt.plot(theta, phi_array[idx, :], 
    #              label=r'$t = {}$'.format(times[idx]))

    # plt.legend()
    # plt.show()

if __name__ == "__main__":

    main()
