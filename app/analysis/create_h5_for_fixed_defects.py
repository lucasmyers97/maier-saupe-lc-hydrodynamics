import argparse
import os
import subprocess

import h5py
import numpy as np

from .utilities import archives as ar
from .utilities import nematics as nu

def get_commandline_args():

    description = ('Create hdf5 file and populate with defect position data '
                   'and archive times so that core structure points can be '
                   'read with executable.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        dest='data_folder',
                        help=('folder where defect location data and archive'
                              'data live'))
    parser.add_argument('--archive_prefix',
                        dest='archive_prefix',
                        help='filename prefix of archive files')
    parser.add_argument('--centers',
                        dest='centers',
                        nargs=4,
                        help='defect centers')
    parser.add_argument('--r0',
                        dest='r0',
                        type=float,
                        help='inner radius of core sample points')
    parser.add_argument('--rf',
                        dest='rf',
                        type=float,
                        help='outer radius of core sample points')
    parser.add_argument('--n_r',
                        dest='n_r',
                        type=int,
                        help='number of defect points in radial direction')
    parser.add_argument('--n_theta',
                        dest='n_theta',
                        type=int,
                        help='number of defect points in polar direction')
    parser.add_argument('--dim',
                        dest='dim',
                        type=int,
                        help='dimension of the simulation')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help=('output h5 file where Fourier mode data will be '
                              'written'))

    args = parser.parse_args()

    output_filename = os.path.join(args.data_folder,
                                   args.output_filename)

    return (args.data_folder, args.archive_prefix, 
            output_filename, args.centers,
            args.r0, args.rf, args.n_r, args.n_theta, args.dim)



def main():

    (data_folder, archive_prefix,
     output_filename, centers,
     r0, rf, n_r, n_theta, dim) = get_commandline_args()

    _, times = ar.get_archive_files(data_folder, archive_prefix)

    pos_points = np.zeros((times.shape[0], dim))
    neg_points = np.zeros((times.shape[0], dim))

    pos_points[:, 0] = centers[0]
    pos_points[:, 1] = centers[1]
    
    neg_points[:, 0] = centers[2]
    neg_points[:, 1] = centers[3]

    file = h5py.File(output_filename, 'w')
    for time in times:
        h5_groupname = 'timestep_{}'.format(time)
        timestep_grp = file.create_group(h5_groupname)
        dataset_dims = (n_r * n_theta, nu.vec_dim(dim))

        pos_Q_vec = timestep_grp.create_dataset('pos_Q_vec', dataset_dims)
        pos_Q_vec.attrs['r0'] = r0
        pos_Q_vec.attrs['rf'] = rf
        pos_Q_vec.attrs['n_r'] = n_r
        pos_Q_vec.attrs['n_theta'] = n_theta
        pos_Q_vec.attrs['dim'] = dim

        neg_Q_vec = timestep_grp.create_dataset('neg_Q_vec', dataset_dims)
        neg_Q_vec.attrs['r0'] = r0
        neg_Q_vec.attrs['rf'] = rf
        neg_Q_vec.attrs['n_r'] = n_r
        neg_Q_vec.attrs['n_theta'] = n_theta
        neg_Q_vec.attrs['dim'] = dim

    file.create_dataset('pos_centers', data=pos_points)
    file.create_dataset('neg_centers', data=neg_points)
    file.create_dataset('times', data=times)
    file.close()

if __name__ == "__main__":
    main()
