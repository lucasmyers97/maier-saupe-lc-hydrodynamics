import argparse

import numpy as np
import h5py

vec_dim = 5
def get_commandline_args():

    description = ("Generate Q-tensor values for periodically varied director "
                   "and write to hdf5 file so that Lambda can be calculated")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--S', 
                        dest='S',
                        type=float,
                        help='Scalar order parameter of uniform configuration')
    parser.add_argument('--k',
                        dest='k',
                        type=float,
                        help='Wave number of periodic perturbation')
    parser.add_argument('--eps',
                        dest='eps',
                        type=float,
                        help='Amplitude of director perturbation')
    parser.add_argument('--limits',
                        dest='limits',
                        nargs=2,
                        type=float,
                        help='Domain limits of configuration')
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

    return (args.S, args.k, args.eps, args.limits, args.n_points, 
            args.filename, args.dataset_name)



def main():

    (S, k, eps, limits, n_points, 
     filename, dataset_name) = get_commandline_args()

    x = np.linspace(limits[0], limits[1], n_points)
    Q = np.zeros((n_points, vec_dim))
    Q[:, 0] = S * (2 / 3)
    Q[:, 1] = eps * np.sin(k * x)
    Q[:, 3] = -S * (1 / 3)

    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(dataset_name, data=Q)


if __name__ == '__main__':
    main()
