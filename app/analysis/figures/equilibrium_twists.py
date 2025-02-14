
import argparse
import os

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    desc = 'Takes in spreadsheet of L2 values and equilibrium twist wavenumbers (omega)'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_folder', help='folder where data lives')
    parser.add_argument('--filename', help='Name of excel type file with data')
    parser.add_argument('--plot_filename', help='Filename of output plot')
    parser.add_argument('--R', type=float, help='Capillary radius')

    args = parser.parse_args()

    filename = os.path.join(args.data_folder, args.filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return filename, plot_filename, args.R



def main():

    filename, plot_filename, R = get_commandline_args()
    data = pd.read_excel(filename)

    plt.plot(data['L2'], data['omega'], marker='o')
    # plt.title('Equilibrium twists')
    plt.xlabel(r'$L_2$')
    plt.ylabel(r'$\omega_0$')
    plt.yticks([0, np.pi / (8 * R), 2 * np.pi / (8 * R)], 
               [r'$0$', r'$\pi / 8 R$', r'$\pi / 4 R$'])

    plt.tight_layout()

    plt.savefig(plot_filename)

    plt.show()

if __name__ == '__main__':

    main()
