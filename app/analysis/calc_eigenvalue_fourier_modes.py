import argparse
import os

import h5py
import numpy as np

from .utilities import nematics as nu
from .utilities.fourier import calculate_trigonometric_fourier_coefficients_vs_r as calc_fourier_vs_r

def get_commandline_args():
    

    description = ('Get fourier modes of core structure eigenvalues from '
                   '`get_points_around_defect` program, output to h5')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--input_filename',
                        dest='input_filename',
                        help='input hdf5 filename containing cores structure')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='output hdf5 filename containing fourier modes')
    parser.add_argument('--data_key',
                        dest='data_key',
                        help='key name of timestep in hdf5 file')
    parser.add_argument('--output_data_key',
                        dest='output_data_key',
                        help='key name of timestep in output hdf5 file')
    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    output_filename = None
    if args.output_folder:
        output_filename = os.path.join(args.output_folder, args.output_filename)
    else:
        output_filename = os.path.join(args.data_folder, args.output_filename)

    return input_filename, output_filename, args.data_key, args.output_data_key

def main():

    input_filename, output_filename, data_key, output_data_key = get_commandline_args()

    file = h5py.File(input_filename)
    data = file[data_key]

    r0 = data.attrs['r0']
    rf = data.attrs['rf']
    n_r = data.attrs['n_r']
    n_theta = data.attrs['n_theta']

    Q_vec = np.array(data[:])
    Q_mat = nu.Q_vec_to_mat(Q_vec)
    q1, q2, _, _ = nu.eigensystem_from_Q(Q_mat)

    q1 = q1.reshape((n_r, n_theta))
    q2 = q2.reshape((n_r, n_theta))

    An_Gamma, _ = calc_fourier_vs_r(q1 - q2)

    output_file = h5py.File(output_filename, 'w')
    output_file[output_data_key] = An_Gamma
    output_file[output_data_key].attrs['r0'] = r0
    output_file[output_data_key].attrs['rf'] = rf
    output_file[output_data_key].attrs['n_r'] = n_r
    n_modes = (n_theta//2) + 1 if (n_theta % 2 == 0) else (n_theta + 1) // 2
    output_file[output_data_key].attrs['n_modes'] = n_modes

if __name__ == '__main__':

    main()
