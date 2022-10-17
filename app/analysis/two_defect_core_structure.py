import paraview.simple as ps
import paraview.servermanager as psm

import os
import argparse
import time

import numpy as np
import h5py
from mpi4py import MPI

import utilities.paraview as pvu
import utilities.nematics as nu

def get_commandline_args():

    description = ("Get detailed core structure around defects from vtu files")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--configuration_prefix', 
                        dest='configuration_prefix',
                        help='prefix of pvtu file holding configuration')
    parser.add_argument('--defect_positions_filename',
                        dest='defect_positions_filename',
                        default='defect_positions.h5',
                        help='name of h5 file holding defect positions')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='output hdf5 filename')

    parser.add_argument('--timestep',
                        dest='timestep',
                        type=int,
                        default=0,
                        help='final timestep at which to evaluate defect core')
    parser.add_argument('--r0',
                        dest='r0',
                        type=float,
                        default=0.25,
                        help='inner radius of sampled points at core')
    parser.add_argument('--rf',
                        dest='rf',
                        type=float,
                        default=2.5,
                        help='outer radius of sampled points at core')
    parser.add_argument('--n',
                        dest='n',
                        type=int,
                        default=1000,
                        help='number of points in the azimuthal direction')
    parser.add_argument('--m',
                        dest='m',
                        type=int,
                        default=100,
                        help='number of points in the radial direction')
    

    args = parser.parse_args()

    defect_filename = os.path.join(args.data_folder,
                                   args.defect_positions_filename)
    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    output_filename = os.path.join(output_folder, args.output_filename)

    return (args.data_folder, args.configuration_prefix, defect_filename,
            args.timestep, output_filename, args.r0, args.rf, args.m, args.n)



def main():

    start = time.time()

    (data_folder, configuration_prefix, 
     defect_filename, timestep, hdf5_filename,
     r0, rf, m, n) = get_commandline_args()
    point_dims = (m, n)

    vtu_filenames, vtu_times = pvu.get_vtu_files(data_folder, 
                                                 configuration_prefix)
    vtu_full_path = []
    for vtu_filename in vtu_filenames:
        vtu_full_path.append( os.path.join(data_folder, vtu_filename) )
    vtu_times = list(vtu_times)
    # time_idx = vtu_times.index(timestep)

    file = h5py.File(defect_filename)
    charge = np.array(file['charge'][:])
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])
    y = np.array(file['y'][:])

    (pos_t, neg_t, 
     pos_centers, neg_centers) = nu.split_defect_centers_by_charge(charge, t, 
                                                                   x, y)

    # Read in raw data
    Q_configuration = ps.XMLPartitionedUnstructuredGridReader(FileName=vtu_full_path)
    
    # Only here so we can control the time-step
    ps.Show()
    view = ps.GetActiveView()

    eigenvalue_filter = pvu.get_eigenvalue_programmable_filter(Q_configuration)

    # overwrite file if it exists
    pos_grp_name = "plus_half_defect"
    neg_grp_name = "minus_half_defect"
    f = h5py.File(hdf5_filename, "w")
    f.create_group(pos_grp_name)
    f.create_group(neg_grp_name)
    f.close()

    for i in range(timestep):

        print("Starting step: {}".format(i))

        pos_defect_time_idx = np.argmin(np.abs(pos_t - vtu_times[i]))
        pos_defect_center = (pos_centers[pos_defect_time_idx][0],
                             pos_centers[pos_defect_time_idx][1])
        neg_defect_time_idx = np.argmin(np.abs(neg_t - vtu_times[i]))
        neg_defect_center = (neg_centers[neg_defect_time_idx][0],
                             neg_centers[neg_defect_time_idx][1])
        
        (pos_poly_points, 
         pos_r, pos_theta) = pvu.generate_sample_points(r0, 
                                                        rf, 
                                                        point_dims, 
                                                        pos_defect_center)
        (neg_poly_points, 
         neg_r, neg_theta) = pvu.generate_sample_points(r0, 
                                                        rf, 
                                                        point_dims, 
                                                        neg_defect_center)

        # Query configuration at sample points
        pos_resampled_data = ps.ResampleWithDataset(SourceDataArrays=eigenvalue_filter,
                                                    DestinationMesh=pos_poly_points)
        neg_resampled_data = ps.ResampleWithDataset(SourceDataArrays=eigenvalue_filter,
                                                    DestinationMesh=neg_poly_points)

        pos_hdf5_filter = pvu.write_two_defect_polydata_to_hdf5(pos_resampled_data, 
                                                                hdf5_filename, 
                                                                pos_grp_name,
                                                                pos_r, 
                                                                pos_theta, 
                                                                point_dims, 
                                                                pos_defect_center,
                                                                vtu_times[i])
        neg_hdf5_filter = pvu.write_two_defect_polydata_to_hdf5(neg_resampled_data, 
                                                                hdf5_filename, 
                                                                neg_grp_name,
                                                                neg_r, 
                                                                neg_theta, 
                                                                point_dims, 
                                                                neg_defect_center,
                                                                vtu_times[i])

        # Show hdf5 filter so that it actually executes
        view = ps.GetActiveView()
        view.ViewTime = Q_configuration.TimestepValues[i]
        ps.Show(pos_hdf5_filter)
        ps.Show(neg_hdf5_filter)

    end = time.time()
    print(end - start)

if __name__ == "__main__":

    main()
