"""
Just prints to console all of the Q-tensor components at the support points
from a vtu file at a particular timestep.
"""
import paraview.simple as ps
import paraview.servermanager as psm

import os
import argparse
import time

import numpy as np
import h5py
from mpi4py import MPI

import utilities.paraview as pvu

def get_commandline_args():

    description = ("Get values of Q-tensor at points in vtu file")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where vtu data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--configuration_prefix', 
                        dest='configuration_prefix',
                        help='prefix of pvtu file holding configuration')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='output hdf5 filename containing point data')

    parser.add_argument('--timestep',
                        dest='timestep',
                        type=int,
                        default=0,
                        help='timestep at which to evaluate points')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    output_filename = os.path.join(output_folder, args.output_filename)

    return (args.data_folder, args.configuration_prefix, args.timestep, 
            output_filename)



def main():

    (data_folder, configuration_prefix, 
     timestep, hdf5_filename) = get_commandline_args()

    vtu_filenames, times = pvu.get_vtu_files(data_folder, configuration_prefix)
    vtu_full_path = []
    for vtu_filename in vtu_filenames:
        vtu_full_path.append( os.path.join(data_folder, vtu_filename) )

    times = list(times)
    time_idx = times.index(timestep)

    # Read in raw data
    Q_configuration = ps.XMLPartitionedUnstructuredGridReader(FileName=vtu_full_path)
    
    # Only here so we can control the time-step
    ps.Show()
    view = ps.GetActiveView()

    Q_config = psm.Fetch(Q_configuration)
    print(Q_config.PointData)



if __name__ == '__main__':
    main()
