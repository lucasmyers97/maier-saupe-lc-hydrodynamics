import argparse
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

# plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300


def get_commandline_args():

    description = ("Plot Lambda values for periodic Q-configuration with"
                   "various epsilon values")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='Folder where Lambda data lives')
    parser.add_argument('--filename', 
                        dest='filename',
                        help='Name of hdf5 file with Lambda values')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='Name of output file with plot')
    args = parser.parse_args()

    filename = os.path.join(args.data_folder, args.filename)
    output_filename = os.path.join(args.data_folder, args.output_filename)

    return filename, output_filename



def main():

    filename, output_filename = get_commandline_args()

    file = h5py.File(filename)
    Lambda = np.array(file['Lambda'][:])
    Q = np.array(file['Q'][:])
    eps = np.array(file['eps'][:])
    x = np.array(file['x'][:])
    dims = np.array(file['dims'][:])

    Lambda2 = Lambda[:, 1].reshape(dims)
    Q2 = Q[:, 1].reshape(dims)
    Eps = eps.reshape(dims)
    X = x.reshape(dims)

    fig, ax = plt.subplots()
    for i in range(X.shape[1]):
        ax.plot(X[:, i], Lambda2[:, i] / Q2[:, i], label='eps = {}'.format(Eps[0, i]))

    fig.legend()
    fig.savefig(output_filename)



if __name__ == '__main__':
    main()
