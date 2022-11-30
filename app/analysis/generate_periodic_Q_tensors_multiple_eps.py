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
    parser.add_argument('--n-eps',
                        dest='n_eps',
                        type=int,
                        help='Number of different epsilon values')
    parser.add_argument('--eps-lims',
                        dest='eps_lims',
                        nargs=2,
                        type=float,
                        help='Limits of epsilon values')
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
                        default='Q',
                        help='Name of dataset in hdf5 file containing Q-tensor values')

    args = parser.parse_args()

    return (args.S, args.k, args.n_eps, args.eps_lims, 
            args.limits, args.n_points, 
            args.filename, args.dataset_name)



def main():

    (S, k, n_eps, eps_lims, limits, n_points, 
     filename, dataset_name) = get_commandline_args()

    x = np.linspace(limits[0], limits[1], n_points)
    eps = np.linspace(eps_lims[0], eps_lims[1], n_eps)
    X, Eps = np.meshgrid(x, eps, indexing='ij')
    x = X.flatten()
    eps = Eps.flatten()

    Q = np.zeros((n_points * n_eps, vec_dim))
    Q[:, 0] = S * (2 / 3)
    Q[:, 1] = S * eps * np.sin(k * x)
    Q[:, 3] = -S * (1 / 3)

    dims = (n_points, n_eps)
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(dataset_name, data=Q)
        dset = f.create_dataset('x', data=x)
        dset = f.create_dataset('eps', data=eps)
        dset = f.create_dataset('dims', data=dims)

if __name__ == '__main__':
    main()