import argparse

import numpy as np
import h5py

vec_dim = 5
def get_commandline_args():

    description = ('Generate Q-tensor values for uniaxial nematic with '
                   'varying S-values')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--S_lims', 
                        dest='S_lims',
                        nargs=2,
                        type=float,
                        help=('Scalar order parameter range for uniaxial '
                             'configuration'))
    parser.add_argument('--n_points',
                        dest='n_points',
                        type=int,
                        help='Number of points to sample configuration at')
    parser.add_argument('--filename',
                        dest='filename',
                        help='Name of hdf5 file which points will be written to')
    parser.add_argument('--dataset_name',
                        dest='dataset_name',
                        help='Name of dataset in hdf5 file containing Q-tensor values')

    args = parser.parse_args()

    return args.S_lims, args.n_points, args.filename, args.dataset_name



def main():

    S_lims, n_points, filename, dataset_name = get_commandline_args()

    S = np.linspace(S_lims[0], S_lims[1], n_points)
    Q = np.zeros((n_points, vec_dim))
    Q[:, 0] = S * (2 / 3)
    Q[:, 3] = -S * (1 / 3)

    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(dataset_name, data=Q)
        dset = f.create_dataset('S', data=S)


if __name__ == '__main__':
    main()
