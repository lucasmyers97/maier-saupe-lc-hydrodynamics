import argparse
import os

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    desc = 'Takes in spreadsheet of L2 values and equilibrium omega angles'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_folder', help='folder where data lives')
    parser.add_argument('--filename', help='Name of excel type file with data')
    parser.add_argument('--plot_filename', help='Filename of output plot')

    args = parser.parse_args()

    filename = os.path.join(args.data_folder, args.filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return filename, plot_filename



def main():

    filename, plot_filename = get_commandline_args()
    data = pd.read_excel(filename)

    equilibrium_data = data[data['equilibrium'] == 1]

    plt.plot(equilibrium_data['L2'], equilibrium_data['angle'], ls='', marker='o')
    # plt.title('Equilibrium twists')
    plt.xlabel(r'$L_2 / L_1$')
    plt.ylabel(r'$\beta$')
    plt.yticks([np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2], 
               [r'$\pi / 8$', r'$\pi / 4$', r'$3 \pi / 8$', r'$\pi / 2$'])

    plt.tight_layout()

    plt.savefig(plot_filename)

    plt.show()

if __name__ == '__main__':

    main()
