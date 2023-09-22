import argparse
import os

import pyvista as pv
import numpy as np

def get_commandline_args():

    desc = ('Read data from Q-tensor or perturbative director simulation vtu '
            'file. Calculate theta_c (anisotropic director correction), and '
            'then write result to hdf5 file.')
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--data_folder',
                        help='folder where the data is')
    parser.add_argument('--output_folder',
                        help='folder where the output data is written')
    parser.add_argument('--input_filename',
                        help='name of data file (vtu or pvtu file)')
    parser.add_argument('--output_filename',
                        help='name of the output file (h5)')
    parser.add_argument('--output_data_key',
                        help='name of data key in output hdf5 file')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--Q_tensor', 
                       action='store_true',
                       help='denotes the input data is for the Q-tensor')
    group.add_argument('--perturbative_director', 
                       action='store_false',
                       help=('denotes the input data is for the perturbative '
                             'director'))

    args = parser.parse_args()

    output_folder = args.data_folder if not args.output_folder else args.output_folder
    input_filename = os.path.join(args.data_folder, args.input_filename)
    output_filename = os.path.join(output_folder, args.output_filename)

    return input_filename, output_filename, args.output_data_key, args.Q_tensor



def main():
    
    args = get_commandline_args()
    print(args)



if __name__ == '__main__':
    main()
