import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    description = ('Find alpha Maier-Saupe coupling parameter which '
                   'corresponds to a set of Landau-de Gennes parameters by '
                   'performing a curve fit on the bulk free energies for a '
                   'uniaxial set of Q-tensor values')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--A',
                        dest='A',
                        type=float,
                        help='A Landau-de Gennes parameter value')
    parser.add_argument('--B',
                        dest='B',
                        type=float,
                        help='B Landau-de Gennes parameter value')
    parser.add_argument('--C',
                        dest='C',
                        type=float,
                        help='C Landau-de Gennes parameter value')

    
    parser.add_argument('--data_folder',
                        dest='data_folder',
                        help='Folder where free energy data lives')
    parser.add_argument('--data_filename',
                        dest='data_filename',
                        help='Name of hdf5 file where Q-tensor data lives')
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='Filename of output plot')

    args = parser.parse_args()

    filename = os.path.join(args.data_folder, args.data_filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return args.A, args.B, args.C, filename, plot_filename



def LdG_free_energy(S, A, B, C):

    return (1 / 27) * (9 * A + 2 * B * S + 3 * C * S**2) * S**2



def MS_free_energy(Q, Lambda, Z, alpha):

    mean_field_term = -( 0.5 * alpha
                         * 2 * (Q[:, 0]**2 
                                + Q[:, 0] * Q[:, 3] 
                                + Q[:, 1]**2 
                                + Q[:, 2]**2 
                                + Q[:, 3]**2 
                                + Q[:, 4]**2) )

    entropy_term = ( np.log(4 * np.pi)
                     - np.log(Z)
                     + (2 * Q[:, 0] * Lambda[:, 0]
                        + Q[:, 0] * Lambda[:, 3]
                        + 2 * Q[:, 1] * Lambda[:, 1]
                        + 2 * Q[:, 2] * Lambda[:, 2]
                        + Q[:, 3] * Lambda[:, 0]
                        + 2 * Q[:, 3] * Lambda[:, 3]
                        + Q[:, 4] * Lambda[:, 4]) )

    return mean_field_term + entropy_term



def main():

    A, B, C, filename, plot_filename = get_commandline_args()
    
    file = h5py.File(filename)

    S = np.array(file['S'][:])
    Q = np.array(file['Q'][:])
    Lambda = np.array(file['Lambda'][:])
    Z = np.array(file['Z'][:]) * 4 * np.pi

    alpha = 7.0
    LdG_fe = LdG_free_energy(S, A, B, C)
    MS_fe = MS_free_energy(Q, Lambda, Z, alpha)

    plt.plot(S, LdG_fe)
    # plt.plot(S, MS_fe)
    plt.show()



if __name__ == '__main__':

    main()
