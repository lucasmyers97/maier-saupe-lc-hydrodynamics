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
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help=('output h5 file where Fourier mode data will be '
                              'written'))

    parser.add_argument('--defects_midpoint',
                        dest='defects_midpoint',
                        nargs=2,
                        help=('defects midpoint in the format x y. '
                              'Use this to fix the defect midpoint at one '
                              'point. Note that this is mutually exclusive '
                              'with --dt and --defect_filename arguments '
                              'which can be used to adjust the midpoint based '
                              'on recorded defect positions'))

    parser.add_argument('--dt',
                        dest='dt',
                        type=float,
                        help='timestep length')
    parser.add_argument('--defect_filename',
                        dest='defect_filename',
                        help='filename of defect position data')

    args = parser.parse_args()

    defect_filename = None
    if (args.defect_filename):
        defect_filename = os.path.join(args.data_folder, args.defect_filename)

    output_filename = os.path.join(args.data_folder, args.output_filename)

    return (args.data_folder, args.archive_prefix, 
            args.dt, defect_filename, 
            args.defects_midpoint,
            output_filename,
            args.r0, args.rf, args.n_r, args.n_theta, args.dim)



def main():

    (data_folder, archive_prefix,
     dt, defect_filename, 
     defects_midpoint,
     output_filename,
     r0, rf, n_r, n_theta, dim) = get_commandline_args()

    _, times = ar.get_archive_files(data_folder, archive_prefix)

    points = np.zeros((times.shape[0], dim))
    if (defect_filename):
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

        for i in range(times.shape[0]):
            points[i, :] = 0.5 * (pos_points[i, :] + neg_points[i, :])
    else:
        points[:, 0] = defects_midpoint[0]
        points[:, 1] = defects_midpoint[1]


    file = h5py.File(output_filename, 'w')
    for time in times:
        h5_groupname = 'timestep_{}'.format(time)
        timestep_grp = file.create_group(h5_groupname)
        dataset_dims = (n_r * n_theta, nu.vec_dim(dim))
        Q_vec = timestep_grp.create_dataset('Q_vec', dataset_dims)
        Q_vec.attrs['r0'] = r0
        Q_vec.attrs['rf'] = rf
        Q_vec.attrs['n_r'] = n_r
        Q_vec.attrs['n_theta'] = n_theta
        Q_vec.attrs['dim'] = dim

    file.create_dataset('centers', data=points)
    file.create_dataset('times', data=times)
    file.close()

if __name__ == "__main__":
    main()
