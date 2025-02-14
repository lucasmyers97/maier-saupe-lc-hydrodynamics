import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
plt.style.use('science')

colors = ['tab:blue', 'tab:orange', 'tab:red']

def get_commandline_args():

    desc = 'Plot differences between ER and TER energies as a function fo L2 for L3 = 0'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--filename', help='ods file containing energy data')
    parser.add_argument('--folder', help='folder containing data file')
    parser.add_argument('--plot_filename', help='name of plot file, must be png')
    parser.add_argument('--R', type=float, help='cylinder radius')
    parser.add_argument('--elastic', 
                        action='store_true',
                        help='whether to plot elastic or total energy')

    args = parser.parse_args()

    data_filename = os.path.join(args.folder, args.filename)
    plot_filename = os.path.join(args.folder, args.plot_filename)

    return data_filename, plot_filename, args.R


def main():

    data_filename, plot_filename, R = get_commandline_args()

    data = pd.read_excel(data_filename)
    data = data[(data['R'] == R) & (data['L3'] == 0)]

    ER_data = data[data['TER'] == 0]
    TER_data = data[data['TER'] == 1]

    merged_data = TER_data.merge(ER_data, on='L2')

    energy_TER = 'energy_x'
    energy_ER = 'energy_y'

    energy_diff = merged_data[energy_TER] - merged_data[energy_ER]
    energy_sum = merged_data[energy_TER] + merged_data[energy_ER]
    L2 = merged_data['L2']

    normalized_diff = energy_diff / np.abs(energy_sum)

    fig, ax = plt.subplots()
    ax.plot(L2, normalized_diff, marker='o', color=colors[0])
    ax.set_xlabel(r'$L_2$')
    ax.set_ylabel(r'$\frac{F_\text{TER} - F_\text{ER}}{\left|F_\text{TER} + F_\text{ER}\right|}$')
    fig.tight_layout()
    fig.savefig(plot_filename)


    plt.show()


if __name__ == '__main__':

    main()

