import paraview.simple as ps
import paraview.servermanager as psm

import os
import argparse
import time

import numpy as np
from mpi4py import MPI

import utilities.paraview as pvu

def get_commandline_args():

    description = ("Get detailed core structure around defects from vtu files")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--configuration_prefix', 
                        dest='configuration_prefix',
                        help='prefix of pvtu file holding configuration')
    parser.add_argument('--defect_positions_filename',
                        dest='defect_positions_filename',
                        default='defect_positions.h5',
                        help='name of h5 file holding defect positions')
    args = parser.parse_args()

    defect_filename = os.path.join(args.data_folder,
                                   args.defect_positions_filename)

    return args.data_folder, args.configuration_prefix, defect_filename



def main():

    start = time.time()

    data_folder, configuration_prefix, defect_filename = get_commandline_args()

    vtu_filenames, times = pvu.get_vtu_files(data_folder, configuration_prefix)
    vtu_full_path = []
    for vtu_filename in vtu_filenames:
        vtu_full_path.append( os.path.join(data_folder, vtu_filename) )

    # Read in raw data
    Q_configuration = ps.XMLPartitionedUnstructuredGridReader(FileName=vtu_full_path)

    eigenvalue_filter = pvu.get_eigenvalue_programmable_filter(Q_configuration)

    # Generate sample points
    n = 1000
    m = 100
    r0 = 0.025
    rmax = 2.5

    poly_points = pvu.generate_sample_points(r0, rmax, n, m)

    # Query configuration at sample points
    resampled_data = ps.ResampleWithDataset(SourceDataArrays=eigenvalue_filter,
                                            DestinationMesh=poly_points)

    hdf5_filter = pvu.write_polydata_to_hdf5(resampled_data)

    # Show hdf5 filter so that it actually executes
    ps.Show(hdf5_filter)

    end = time.time()
    print(end - start)

if __name__ == "__main__":

    main()
