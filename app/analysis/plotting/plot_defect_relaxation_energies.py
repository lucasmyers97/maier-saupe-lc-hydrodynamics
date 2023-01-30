"""
Script reads in hdf5 files which contain the datasets `t` and 
`configuration_energy`, each being arrays which correspond to the time and
energies of the configuration at each timestep.
Parent folder can be read in, but the organization is such that the folders
which hold each of the hdf5 files are hardcoded into the script (can't think
of a nicer way to deal with this).
Then it plots energy as a function of time across the different parameters.
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_filenames():

    description = "Read in defect locations from hdf5, plot and find best fit"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--parent_folder', dest='parent_folder',
                        help='folder which houses different system subfolders')
    parser.add_argument('--energy_filename', dest='energy_filename',
                        help='filename of all energy data')
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder which will hold output plots')
    args = parser.parse_args()

    subfolder_list = [['L3_0_R_5', 'L3_0_R_10', 'L3_0_R_15'],
                      ['L3_0_5_R_5', 'L3_0_5_R_10', 'L3_0_5_R_15'],
                      ['L3_3_R_5', 'L3_3_R_10', 'L3_3_R_15']]

    energy_filenames = [[os.path.join(args.parent_folder, 
                                     subfolder,
                                     args.energy_filename)
                        for subfolder in subfolder_sublist]
                        for subfolder_sublist in subfolder_list]

    return energy_filenames, args.output_folder



def read_data(energy_filenames):

    t_data = []
    energy_data = []
    for filename_list in energy_filenames:

        current_t_data = []
        current_energy_data = []
        for filename in filename_list:
            file = h5py.File(filename)
            current_t_data.append( np.array(file['t'][:]) )
            current_energy_data.append( np.array(file['configuration_energy'][:]) )
    
        t_data.append(current_t_data)
        energy_data.append(current_energy_data)

    return t_data, energy_data



def normalize_energy_data(energy_data):

    max_energy = np.max(energy_data)
    min_energy = np.min(energy_data)
    range_energy = max_energy - min_energy

    normalized_energy_data = np.copy(energy_data)
    normalized_energy_data -= min_energy
    normalized_energy_data /= range_energy

    return normalized_energy_data
        

if __name__ == '__main__':

    energy_filenames, output_folder = get_filenames()
    L3_list = [0, 0.5, 3]
    R_list = [5, 10, 15]

    t_data, energy_data = read_data(energy_filenames)

    normalized_energy_data = []
    for energy_list in energy_data:
        normalized_sublist = []
        for energy_sublist in energy_list:
            normalized_sublist.append(normalize_energy_data(energy_sublist))

        normalized_energy_data.append(normalized_sublist)

    # Fixed L3 for different R-values, normalized
    for i in range(3):

        n = 100000
        for j in range(3):
            m = np.max(t_data[i][j].shape[0])
            n = min(m, n)
        
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(t_data[i][j][:n], normalized_energy_data[i][j][:n], label=r'$R = {}$'.format(R_list[j]))

        ax.set_title(r'Defect relaxation for $L_3 = {}$'.format(L3_list[i]))
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel('Normalized energy')
        fig.tight_layout()

        plt.legend()
        
        filename = os.path.join(output_folder, 'defect_relaxation_L3_{}.png'.format(L3_list[i]))
        fig.savefig(filename)

    # Fixed R for different L3-values, normalized
    for i in range(3):

        n = 100000
        for j in range(3):
            m = np.max(t_data[i][j].shape[0])
            n = min(m, n)
        
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(t_data[j][i][:n], normalized_energy_data[j][i][:n], label=r'$L_3 = {}$'.format(L3_list[j]))

        ax.set_title(r'Defect relaxation for $R = {}$'.format(R_list[i]))
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel('Normalized energy')
        fig.tight_layout()

        plt.legend()
        
        filename = os.path.join(output_folder, 'defect_relaxation_R_{}.png'.format(R_list[i]))
        fig.savefig(filename)

    # Fixed L3 for different R-values, energy difference over total energy
    for i in range(3):

        n = 100000
        for j in range(3):
            m = np.max(t_data[i][j].shape[0])
            n = min(m, n)
        
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(t_data[i][j][:(n - 1)], 
                    (energy_data[i][j][1:n] - energy_data[i][j][:(n - 1)]) / energy_data[i][j][:(n - 1)], 
                    label=r'$R = {}$'.format(R_list[j]))

        ax.set_title(r'Defect relaxation for $L_3 = {}$'.format(L3_list[i]))
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel(r'$\Delta f / f$')
        fig.tight_layout()

        plt.legend()
        
        filename = os.path.join(output_folder, 'defect_relaxation_delta_f_L3_{}.png'.format(L3_list[i]))
        fig.savefig(filename)

    # Fixed L3 for different R-values, log energy difference over total energy
    for i in range(3):

        n = 100000
        for j in range(3):
            m = np.max(t_data[i][j].shape[0])
            n = min(m, n)
        
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(t_data[i][j][:(n - 1)], 
                    np.log10(-(energy_data[i][j][1:n] - energy_data[i][j][:(n - 1)]) / energy_data[i][j][:(n - 1)]), 
                    label=r'$R = {}$'.format(R_list[j]))

        ax.set_title(r'Defect relaxation for $L_3 = {}$'.format(L3_list[i]))
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel(r'$\log(-\Delta f / f)$')
        fig.tight_layout()

        plt.legend()
        
        filename = os.path.join(output_folder, 'defect_relaxation_log_delta_f_L3_{}.png'.format(L3_list[i]))
        fig.savefig(filename)

    # Fixed R for different L3-values, log energy difference over total energy
    for i in range(3):

        n = 100000
        for j in range(3):
            m = np.max(t_data[j][i].shape[0])
            n = min(m, n)
        
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(t_data[j][i][:(n - 1)], 
                    (energy_data[j][i][1:n] - energy_data[j][i][:(n - 1)]) / energy_data[j][i][:(n - 1)], 
                    label=r'$L_3 = {}$'.format(L3_list[j]))

        ax.set_title(r'Defect relaxation for $R = {}$'.format(R_list[i]))
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel(r'$\Delta f / f$')
        fig.tight_layout()

        plt.legend()
        
        filename = os.path.join(output_folder, 'defect_relaxation_delta_f_R_{}.png'.format(R_list[i]))
        fig.savefig(filename)

    # Fixed R for different L3-values, log energy difference over total energy
    for i in range(3):

        n = 100000
        for j in range(3):
            m = np.max(t_data[j][i].shape[0])
            n = min(m, n)
        
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(t_data[j][i][:(n - 1)], 
                    np.log10(-(energy_data[j][i][1:n] - energy_data[j][i][:(n - 1)]) / energy_data[j][i][:(n - 1)]), 
                    label=r'$L_3 = {}$'.format(L3_list[j]))

        ax.set_title(r'Defect relaxation for $R = {}$'.format(R_list[i]))
        ax.set_xlabel(r'$t / \tau$')
        ax.set_ylabel(r'$\log(-\Delta f / f)$')
        fig.tight_layout()

        plt.legend()
        
        filename = os.path.join(output_folder, 'defect_relaxation_log_delta_f_R_{}.png'.format(R_list[i]))
        fig.savefig(filename)


    # All individualized plots, not normalized
    for i in range(3):
        for j in range(3):
            fig, ax = plt.subplots()
            ax.plot(t_data[j][i], energy_data[j][i])

            ax.set_title(r'Defect relaxation, $R = {}$, $L_3 = {}$'.format(R_list[i], L3_list[j]))
            ax.set_xlabel(r'$t / \tau$')
            ax.set_ylabel(r'$f / n k_B T$')
            fig.tight_layout()

            filename = os.path.join(output_folder, 'defect_relaxation_R_{}_L3_{}.png'.format(R_list[i], L3_list[j]))
            fig.savefig(filename)
