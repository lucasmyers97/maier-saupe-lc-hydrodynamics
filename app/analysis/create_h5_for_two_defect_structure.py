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
    parser.add_argument('--defect_filename',
                        dest='defect_filename',
                        help='filename of defect position data')
    parser.add_argument('--dt',
                        dest='dt',
                        type=float,
                        help='timestep length')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help=('output h5 file where Fourier mode data will be '
                              'written'))

    args = parser.parse_args()

    defect_filename = os.path.join(args.data_folder,
                                   args.defect_filename)
    output_filename = os.path.join(args.data_folder,
                                   args.output_filename)

    return (args.data_folder, args.archive_prefix, 
            defect_filename, output_filename, 
            args.r0, args.rf, args.n_r, args.n_theta, args.dim, args.dt)



def main():

    (data_folder, archive_prefix,
     defect_filename, output_filename, 
     r0, rf, n_r, n_theta, dim, dt) = get_commandline_args()

    _, times = ar.get_archive_files(data_folder, archive_prefix)

    file = h5py.File(defect_filename)
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])
    y = np.array(file['y'][:])
    charge = np.array(file['charge'][:])
    (pos_t, neg_t, 
     pos_centers, neg_centers) = nu.split_defect_centers_by_charge(charge, t, 
                                                                   x, y)

    pos_points = nu.match_times_to_points(times * dt, 
                                          pos_t, 
                                          pos_centers[:, 0], 
                                          pos_centers[:, 1])
    neg_points = nu.match_times_to_points(times * dt, 
                                          neg_t, 
                                          neg_centers[:, 0], 
                                          neg_centers[:, 1])

    file = h5py.File(output_filename, 'w')
    dataset_dims = (n_r * n_theta, nu.vec_dim(dim))
    for time in times:
        h5_groupname = 'timestep_{}'.format(time)
        timestep_grp = file.create_group(h5_groupname)

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
