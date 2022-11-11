"""
This script plots director angle as a function of polar angle at various
distances from the defect core.
Additionally, if Dzyaloshinskii solutions are available at the estimated
anisotropy parameter values (estimated from L3 and average S-values at the
particular radial value) then it plots the difference between the 
Dzyaloshinskii solutions and the director angles.
"""
import argparse
import os

import h5py
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

import utilities.nematics as nu

def get_commandline_args():

    description = ("Plot director field towards defect core for anisotropic defects")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--configuration_filename', 
                        dest='configuration_filename',
                        help='name of h5 file holding configuration')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='name of output png file of defect angle')
    parser.add_argument('--diff_output_filename',
                        dest='diff_output_filename',
                        help='name of output png file of difference in defect angle')
    parser.add_argument('--n_radii',
                        dest='n_radii',
                        type=int,
                        help='number of radii at which to plot director')
    parser.add_argument('--r_start',
                        dest='r_start',
                        type=float,
                        help='smallest r-value around which to plot director angle')
    parser.add_argument('--r_end',
                        dest='r_end',
                        type=float,
                        help='largest r-value around which to plot director angle')
    parser.add_argument('--dzyaloshinskii_filenames',
                        dest='dzyaloshinskii_filenames',
                        nargs='+',
                        help='filenames of h5 files holding dzyaloshinskii solutions')
    args = parser.parse_args()

    configuration_file = os.path.join(args.data_folder, args.configuration_filename)
    output_file = os.path.join(args.data_folder, args.output_filename)
    diff_output_file = os.path.join(args.data_folder, args.diff_output_filename)

    dzyaloshinskii_filenames = []
    for dzyaloshinskii_filename in args.dzyaloshinskii_filenames:
        dzyaloshinskii_filenames.append( os.path.join(args.data_folder, dzyaloshinskii_filename) )

    return (configuration_file, output_file, diff_output_file, args.n_radii, args.r_start, 
            args.r_end, dzyaloshinskii_filenames)



def get_dzyaloshinskii_interpolator(dzyaloshinskii_filename):

    file = h5py.File(dzyaloshinskii_filename)
    theta = np.array(file['theta'][:])
    phi = np.array(file['phi'][:])

    return interp1d(theta, phi)



def main():

    (configuration_file, output_file, diff_output_file, 
     n_radii, r_start, r_end, dzyaloshinskii_filenames) = get_commandline_args()

    print(dzyaloshinskii_filenames)

    file = h5py.File(configuration_file)
    grp = file['timestep_49']

    n_dim = 3
    point_dims = np.array(grp['point_dims'][:])
    point_dims = np.flip(point_dims)
    n = np.zeros((point_dims[0], point_dims[1], n_dim))
    unshaped_n = np.array(grp['n'][:])

    for i in range(n_dim):
        n[:, :, i] = unshaped_n[:, i].reshape(point_dims)

    r = grp['r'][:]
    theta = grp['theta'][:]
    S = grp['S'][:]
    S = S.reshape(point_dims)

    i_start = np.argmin( np.abs(r - r_start) )
    i_end = np.argmin( np.abs(r - r_end) )

    R, Theta = np.meshgrid(r, theta)
    step_size = int( (i_end - i_start) / n_radii)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    count = 0
    for i in range(i_start, i_end, step_size):

        current_n = n[:, i, :]
        phi = nu.director_to_angle(current_n).transpose()
        new_phi = nu.sanitize_director_angle(phi)
        theta = Theta[:, i]
        r = R[0, i]
        s = 1.5 * np.average(S[:, i])

        f = get_dzyaloshinskii_interpolator(dzyaloshinskii_filenames[count])   

        print("S = {}".format(s))
        print("r = {}".format(r))

        ax1.plot(theta, new_phi, label="r = {}, S = {}".format(r, s))
        ax2.plot(theta, new_phi - f(theta), label="r = {}, S = {}".format(r, s))

        count += 1

    ax1.legend()
    ax2.legend()

    ax1.set_xlabel(r'$\theta$ (polar angle)')
    ax1.set_ylabel(r'$\phi$ (director angle)')
    ax1.set_title(r'$\phi$ at differing radii for anisotropic defect')

    ax2.set_xlabel(r'$\theta$ (polar angle)')
    ax2.set_ylabel(r'$\phi$ (director angle)')
    ax2.set_title(r'$\phi - \phi_{Dzyaloshinskii}$ at differing radii for anisotropic defect')
    ax2.set_ylim([-.25, .25])

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(output_file)
    fig2.savefig(diff_output_file)

    plt.show()



if __name__ == "__main__":

    main()
