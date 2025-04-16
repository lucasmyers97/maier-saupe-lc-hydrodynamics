import argparse
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

linestyles = ['-', '--', '-.', ':']

def get_commandline_args():

    parser = argparse.ArgumentParser(description='Get core point data for multiple configurations and plot')
    parser.add_argument('--data_folder', help='Folder holding data')
    parser.add_argument('--filenames',
                        nargs='+',
                        help='Filenames of csv files holding point data')
    parser.add_argument('--output_filename', help='name of png plot file')
    parser.add_argument('--B_vals',
                        nargs='+',
                        type=float,
                        help='B values of configurations')
    parser.add_argument('--disclination_size',
                        type=float,
                        help='size of disclination')

    args = parser.parse_args()

    filenames = [os.path.join(args.data_folder, filename) for filename in args.filenames]
    output_filename = os.path.join(args.data_folder, args.output_filename)

    return filenames, output_filename, args.B_vals, args.disclination_size


def main():

    filenames, output_filename, B_vals, disclination_size = get_commandline_args()

    fig, ax = plt.subplots()
    for i, (filename, B) in enumerate(zip(filenames, B_vals)):
        data = pd.read_csv(filename)

        x = data['Points:0']
        S = data['S']
        P = data['P']

        x_pos = x[x > 0]
        x0 = x_pos.values[np.argmin(S[x > 0])]

        ax.plot(x - x0, S, label='B = {}'.format(B), ls=linestyles[i], color='tab:blue')
        ax.plot(x - x0, P, ls=linestyles[i], color='tab:red')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$S$, $P$')
    plt.xlim(-disclination_size, disclination_size)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()
    

if __name__ == '__main__':
    main()
