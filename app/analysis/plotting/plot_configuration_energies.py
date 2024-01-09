import argparse
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ('Plot total energy as a function of'
                   ' time for nematic configuration based on hdf5 files')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where configuration energy data lives')
    parser.add_argument('--data_filename',
                        help='file where configuration energy data lives')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filename',
                        help='filename of energy vs time plot')
    parser.add_argument('--title',
                        help='title of energy vs time plot')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    plot_filename = os.path.join(output_folder, args.plot_filename)

    return data_filename, plot_filename, args.title



def main():

    data_filename, plot_filename, title = get_commandline_args()

    file = h5py.File(data_filename)

    total_energy = None
    total_energy = np.array( file['mean_field_term'][:]
                             + file['entropy_term'][:]
                             + file['L1_elastic_term'][:]
                             + file['L2_elastic_term'][:]
                             + file['L3_elastic_term'][:] )

    t = np.array( file['t'][:] )

    print('tf = {}, Ef = {}'.format(t[-1], total_energy[-1]))

    fig, ax = plt.subplots()
    ax.plot(t, total_energy)
    ax.set_title(title)
    ax.set_xlabel(r'$t / \tau$')
    ax.set_ylabel(r'$F / n k_B T$')
    fig.tight_layout()
    fig.savefig(plot_filename)

    plt.show()



if __name__ == '__main__':

    main()

