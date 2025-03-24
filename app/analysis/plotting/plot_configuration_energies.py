import argparse
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ('Plot total energy as a function of'
                   ' time for nematic configuration based on hdf5 files')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where configuration energy data lives')
    parser.add_argument('--data_filename',
                        default='configuration_energy.h5',
                        help='file where configuration energy data lives')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filename',
                        default='configuration_energy.png',
                        help='filename of energy vs time plot')
    parser.add_argument('--final_timestep',
                        type=float,
                        default=None,
                        help='timestep at which to evaluate final energy')
    parser.add_argument('--title',
                        default='energy',
                        help='title of energy vs time plot')
    parser.add_argument('--show_output',
                        type=int,
                        default=True,
                        help='whether to show output in new windows')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    plot_filename = os.path.join(output_folder, args.plot_filename)

    return data_filename, plot_filename, args.title, args.final_timestep, args.show_output



def main():

    data_filename, plot_filename, title, final_timestep, show_output = get_commandline_args()

    file = h5py.File(data_filename)

    total_energy = None
    total_energy = np.array( file['mean_field_term'][:]
                             # + file['cubic_term'][:]
                             + file['entropy_term'][:]
                             + file['L1_elastic_term'][:]
                             + file['L2_elastic_term'][:]
                             + file['L3_elastic_term'][:] )

    elastic_energy = np.array( file['L1_elastic_term'][:]
                               + file['L2_elastic_term'][:]
                               + file['L3_elastic_term'][:] )

    t = np.array( file['t'][:] )

    final_idx = -1
    if final_timestep:
        final_idx = np.argmin( np.abs( t - final_timestep ) )

    print('tf = {}, Ei = {}, Ef = {}'.format(t[-1], total_energy[0], total_energy[-1]))
    print('tf = {}, Ei = {}, Ef = {}'.format(t[final_idx], total_energy[0], total_energy[final_idx]))
    print('tf = {}, Elastic Ei = {}, Elastic Ef = {}'.format(t[-1], elastic_energy[0], elastic_energy[-1]))
    print('tf = {}, Elastic Ei = {}, Elastic Ef = {}'.format(t[final_idx], elastic_energy[0], elastic_energy[final_idx]))

    mean_field_term = np.array(file['mean_field_term'][:])
    cubic_term = np.array(file['cubic_term'][:])
    entropy_term = np.array(file['entropy_term'][:])
    L1_elastic_term = np.array(file['L1_elastic_term'][:])
    L2_elastic_term = np.array(file['L2_elastic_term'][:])
    L3_elastic_term = np.array(file['L3_elastic_term'][:])
    
    print()
    print('Initial energies:')
    print('mean_field_term = {}'.format(mean_field_term[0]))
    print('cubic_term = {}'.format(cubic_term[0]))
    print('entropy_term = {}'.format(entropy_term[0]))
    print('L1_elastic_term = {}'.format(L1_elastic_term[0]))
    print('L2_elastic_term = {}'.format(L2_elastic_term[0]))
    print('L3_elastic_term = {}'.format(L3_elastic_term[0]))
    
    print()
    print('Final energies:')
    print('mean_field_term = {}'.format(mean_field_term[-1]))
    print('cubic_term = {}'.format(cubic_term[-1]))
    print('entropy_term = {}'.format(entropy_term[-1]))
    print('L1_elastic_term = {}'.format(L1_elastic_term[-1]))
    print('L2_elastic_term = {}'.format(L2_elastic_term[-1]))
    print('L3_elastic_term = {}'.format(L3_elastic_term[-1]))

    dF_dt = (total_energy[-1] - total_energy[-2]) / (t[-1] - t[-2])
    dF_dt_limit = (total_energy[final_idx] - total_energy[final_idx - 1]) / (t[final_idx] - t[final_idx - 1])
    print()
    print('Energy derivative')
    print('dF/dt = {}'.format(dF_dt))
    print('dF/dt / F = {}'.format(dF_dt / total_energy[-1]))
    print('Energy derivative at t = {}'.format(t[final_idx]))
    print('dF/dt = {}'.format(dF_dt_limit))
    print('dF/dt / F = {}'.format(dF_dt_limit / total_energy[final_idx]))

    if show_output:

        fig, ax = plt.subplots()
        ax.plot(t, total_energy)
        ax.set_title(title)
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel(r'$F / n k_B T$')
        fig.tight_layout()
        fig.savefig(plot_filename)

        fig, ax = plt.subplots()
        ax.plot(t, mean_field_term)
        ax.set_title('mean_field_term')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(t, cubic_term)
        ax.set_title('cubic_term')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(t, entropy_term)
        ax.set_title('entropy_term')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(t, L1_elastic_term)
        ax.set_title('L1_elastic_term')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(t, L2_elastic_term)
        ax.set_title('L2_elastic_term')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(t, L3_elastic_term)
        ax.set_title('L3_elastic_term')
        fig.tight_layout()

        plt.show()



if __name__ == '__main__':

    main()

