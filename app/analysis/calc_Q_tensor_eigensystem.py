import argparse
import os

import h5py
import numpy as np

from .utilities import nematics as nu

def get_commandline_args():

    desc = ('Calculates eigenvalues and eigenvectors of a list of Q-tensors. '
            'This expects Q-components to be arranged in an n x 5 '
            'where n is n_r_points x n_theta_points. '
            'The dataset should contain `r0`, `rf`, `n_r`, `n_theta` '
            'attributes.')

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_folder',
                        help='folder where data lives')
    parser.add_argument('--input_filename',
                        help='h5 file where Q-tensor data lives')
    parser.add_argument('--data_key',
                        help='key in h5 file where Q-tensor data is')
    parser.add_argument('--output_filename',
                        help='h5 file where eigensystem data will be written')

    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    output_filename = os.path.join(args.data_folder, args.output_filename)

    return input_filename, args.data_key, output_filename



def main():

    input_filename, data_key, output_filename = get_commandline_args()

    file = h5py.File(input_filename, 'r')
    Q_vec = file[data_key]
    Q_vec_data = np.array(Q_vec[:])

    r0 = Q_vec.attrs['r0']
    rf = Q_vec.attrs['rf']
    n_r = Q_vec.attrs['n_r']
    n_theta = Q_vec.attrs['n_theta']
    file.close()

    Q_data = nu.Q_vec_to_mat(Q_vec_data)

    q1, q2, n, m = nu.eigensystem_from_Q(Q_data)

    output_file = h5py.File(output_filename, 'w')

    group = output_file.create_group('eigensystem')
    group.create_dataset('q1', data=q1)
    group.create_dataset('q2', data=q2)
    group.create_dataset('n', data=n)
    group.create_dataset('m', data=m)

    group.attrs['r0'] = r0
    group.attrs['rf'] = rf
    group.attrs['n_r'] = n_r
    group.attrs['n_theta'] = n_theta



if __name__ == '__main__':
    main()
