import argparse
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ('Plot total energy, dE/dt, and (dE/dQ)^2 as a function of'
                   ' time for nematic configuration based on hdf5 files')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        dest='data_folder',
                        help='folder where configuration energy data lives')
    parser.add_argument('--data_filename',
                        dest='data_filename',
                        help='file where configuration energy data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--energy_filename',
                        dest='energy_filename',
                        help='filename of energy vs time plot')
    parser.add_argument('--dE_dt_filename',
                        dest='dE_dt_filename',
                        help='filename of dE/dt vs time plot')
    parser.add_argument('--dE_dQ_squared_filename',
                        dest='dE_dQ_squared_filename',
                        help='filename of (dE_dQ)^2 vs time plot')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    energy_filename = os.path.join(output_folder, args.energy_filename)
    dE_dt_filename = os.path.join(output_folder, args.dE_dt_filename)
    dE_dQ_squared_filename = os.path.join(output_folder, 
                                          args.dE_dQ_squared_filename)

    return (data_filename, 
            energy_filename, dE_dt_filename, dE_dQ_squared_filename)



def main():

    (data_filename, energy_filename, 
     dE_dt_filename, dE_dQ_squared_filename) = get_commandline_args()

    file = h5py.File(data_filename)

    total_energy = np.array( file['mean_field_term'][:]
                             + file['entropy_term'][:]
                             + file['L1_elastic_term'][:]
                             + file['L2_elastic_term'][:]
                             + file['L3_elastic_term'][:] )
    dE_dQ_squared = np.array( file['dE_dQ_squared'][:] )
    t = np.array( file['t'][:] )

    fig, ax = plt.subplots()
    ax.plot(t, total_energy)
    ax.set_title('Total Energy')
    ax.set_xlabel(r'$t$')
    fig.tight_layout()
    fig.savefig(energy_filename)

    fig, ax = plt.subplots()
    ax.plot(t[1:], np.diff(total_energy) / np.diff(t))
    ax.set_title(r'$\frac{dE}{dt}$ vs. time')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\frac{dE}{dt}$')
    fig.tight_layout()
    fig.savefig(dE_dt_filename)

    fig, ax = plt.subplots()
    ax.plot(t, dE_dQ_squared)
    ax.set_title(r'$\left( \frac{dE}{dQ} \right)^2$ vs. time')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\left( \frac{dE}{dQ} \right)^2$')
    fig.tight_layout()
    fig.savefig(dE_dQ_squared_filename)

    plt.show()



if __name__ == '__main__':

    main()

