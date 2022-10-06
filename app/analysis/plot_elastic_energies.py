import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

import argparse


plt.style.use('science')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400


def get_commandline_args():

    description = ("plot two defect configuration energies")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--energy_filename', 
                        dest='energy_filename',
                        help='name of filename holding configuration energy')
    args = parser.parse_args()

    return args.data_folder, args.energy_filename

def main():

    data_folder, energy_filename = get_commandline_args()
    energy_filename = os.path.join(data_folder, energy_filename)
    L1_output_filename = os.path.join(data_folder, "L1_energy.png")
    L3_output_filename = os.path.join(data_folder, "L3_energy.png")

    file = h5py.File(energy_filename)
    L1_elastic_energy = np.array(file['L1_elastic_term'][:])
    L3_elastic_energy = np.array(file['L3_elastic_term'][:])
    t = np.array(file['t'][:])

    fig, ax = plt.subplots()
    ax.plot(t, L1_elastic_energy)
    ax.set_xlabel("t")
    ax.set_ylabel("L1 elastic energy")
    ax.set_title("L1 elastic energy")
    fig.tight_layout()
    plt.show()
    fig.savefig(L1_output_filename)

    fig, ax = plt.subplots()
    ax.plot(t, L3_elastic_energy)
    ax.set_xlabel("t")
    ax.set_ylabel("L3 elastic energy")
    ax.set_title("L3 elastic energy")
    fig.tight_layout()
    plt.show()
    fig.savefig(L3_output_filename)

if __name__ == "__main__":
    main()
