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

    parser.add_argument('--r0',
                        type=float,
                        help='innermost radial coordinate')
    parser.add_argument('--rf',
                        type=float,
                        help='outermost radial coordinate')
    parser.add_argument('--n_r',
                        type=int,
                        help='number of points in radial direction')
    parser.add_argument('--n_phi',
                        type=int,
                        help='number of points in azimuthal direction')

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

    return (input_filename, output_filename, args.output_data_key, 
            args.r0, args.rf, args.n_r, args.n_phi, args.Q_tensor)



def get_radial_points(r0, rf, n_r, n_phi):
    """
    Makes an nx3 set of (x, y, z) points where n = n_r * n_phi.
    """

    r = np.linspace(r0, rf, n_r)
    phi = np.linspace(0, 2*np.pi, n_phi)
    R, phi = np.meshgrid(r, phi, indexing='ij')
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    Z = np.zeros(X.shape)

    return np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)



def get_theta_c(input_data, points, Q_tensor):

    mesh = pv.PolyData(points)
    if not Q_tensor:
        return mesh.sample(input_data)



def main():
    
    (input_filename, output_filename, output_data_key, 
     r0, rf, n_r, n_phi, Q_tensor) = get_commandline_args()

    input_data = pv.read(input_filename)
    points = get_radial_points(r0, rf, n_r, n_phi)
    theta_c_dataset = get_theta_c(input_data, points, Q_tensor)

    print(theta_c_dataset['theta_c'])



if __name__ == '__main__':
    main()
