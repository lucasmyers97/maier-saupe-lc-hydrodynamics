import argparse
import os
import subprocess

import h5py
import numpy as np

from utilities import archives as ar
from utilities import nematics as nu

def get_commandline_args():

    description = ('Get data about Fourier modes around defects from archived '
                   'nematic configuration')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        dest='data_folder',
                        help=('folder where defect location data and archive'
                              'data live'))
    parser.add_argument('--archive_prefix',
                        dest='archive_prefix',
                        help='filename prefix of archive files')
    parser.add_argument('--defect_filename',
                        dest='defect_filename',
                        help='filename of defect position data')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help=('output h5 file where Fourier mode data will be '
                              'written'))
    parser.add_argument('--r0',
                        dest='r0',
                        type=float,
                        help='inner point distance to defect core')
    parser.add_argument('--rf',
                        dest='rf',
                        type=float,
                        help='outer point distance to defect core')
    parser.add_argument('--n_r',
                        dest='n_r',
                        type=int,
                        help='number of points in the radial direction')
    parser.add_argument('--n_theta',
                        dest='n_theta',
                        type=int,
                        help='number of points in azimuthal direction')
    parser.add_argument('--n_processors',
                        dest='n_processors',
                        type=int,
                        help='number of processors to get points with')
    parser.add_argument('--mpi_program',
                        dest='mpi_program',
                        help=('mpirun executable (may be located in different '
                              'places'))
    parser.add_argument('--executable',
                        dest='executable',
                        help='location of executable to get points from archive')

    parser.add_argument('--print_output',
                        dest='print_output',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help='whether to print output to terminal')

    args = parser.parse_args()

    defect_filename = os.path.join(args.data_folder,
                                   args.defect_filename)
    output_filename = os.path.join(args.data_folder,
                                   args.output_filename)

    return (args.data_folder, args.archive_prefix, 
            defect_filename, output_filename,
            args.r0, args.rf, args.n_r, args.n_theta,
            args.n_processors, args.mpi_program, args.executable,
            args.print_output)



def main():

    (data_folder, archive_prefix,
     defect_filename, output_filename,
     r0, rf, n_r, n_theta,
     n_processors, mpi_program, executable,
     print_output) = get_commandline_args()

    archive_filenames, times = ar.get_archive_files(data_folder, 
                                                    archive_prefix)

    file = h5py.File(defect_filename)
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])
    y = np.array(file['y'][:])
    charge = np.array(file['charge'][:])
    (pos_t, neg_t, 
     pos_centers, neg_centers) = nu.split_defect_centers_by_charge(charge, t, 
                                                                   x, y)

    pos_points = nu.match_times_to_points(times, 
                                          pos_t, 
                                          pos_centers[:, 0], 
                                          pos_centers[:, 1])
    neg_points = nu.match_times_to_points(times, 
                                          neg_t, 
                                          neg_centers[:, 0], 
                                          neg_centers[:, 1])

    dim = 2
    file = h5py.File(output_filename, 'w')
    for time in times:
        h5_groupname = 'timestep_{}'.format(time)
        file.create_group(h5_groupname)
    file.close()

    for center, archive_filename, time in zip(pos_points, archive_filenames, times):

        h5_datasetname = 'pos_defect'
        h5_groupname = 'timestep_{}'.format(time)

        print('Positive defect, time = {}'.format(time))

        # actually call thing
        result = subprocess.run([mpi_program, '-np', '{}'.format(n_processors), 
                                executable,
                                '--dim', '{}'.format(dim),
                                '--r0', '{}'.format(r0),
                                '--rf', '{}'.format(rf),
                                '--center', '{}'.format(center[0]), '{}'.format(center[1]),
                                '--n_r', '{}'.format(n_r),
                                '--n_theta', '{}'.format(n_theta),
                                '--archive_filename', '{}'.format(archive_filename),
                                '--h5_filename', '{}'.format(output_filename),
                                '--h5_groupname', '{}'.format(h5_groupname),
                                '--h5_datasetname', '{}'.format(h5_datasetname)],
                                capture_output=True,
                                text=True)

        if (print_output):
            print(result.stdout)

    for center, archive_filename, time in zip(neg_points, archive_filenames, times):

        h5_groupname = 'timestep_{}'.format(time)
        h5_datasetname = 'neg_defect'
        
        print('Negative defect, time = {}'.format(time))

        # actually call thing
        subprocess.run([mpi_program, '-np', '{}'.format(n_processors), 
                        executable,
                        '--dim', '{}'.format(dim),
                        '--r0', '{}'.format(r0),
                        '--rf', '{}'.format(rf),
                        '--center', '{}'.format(center[0]), '{}'.format(center[1]),
                        '--n_r', '{}'.format(n_r),
                        '--n_theta', '{}'.format(n_theta),
                        '--archive_filename', '{}'.format(archive_filename),
                        '--h5_filename', '{}'.format(output_filename),
                        '--h5_groupname', '{}'.format(h5_groupname),
                        '--h5_datasetname', '{}'.format(h5_datasetname)],
                        capture_output=True,
                        text=True)

        if (print_output):
            print(result.stdout)



if __name__ == "__main__":
    main()
