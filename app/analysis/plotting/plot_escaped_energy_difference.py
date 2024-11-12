import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
plt.style.use('science')

def get_commandline_args():

    desc = 'Plot differences between ER and TER energies for several R values'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--filename', help='ods file containing energy data')
    parser.add_argument('--folder', help='folder containing data file')
    parser.add_argument('--plot_filename', help='name of plot file, must be png')
    parser.add_argument('--L2', type=float, help='L2 value of configurations')
    parser.add_argument('--L3', type=float, help='L3 value of configurations')

    args = parser.parse_args()

    data_filename = os.path.join(args.folder, args.filename)
    plot_filename = os.path.join(args.folder, args.plot_filename)

    return data_filename, plot_filename, args.L2, args.L3


def main():

    data_filename, plot_filename, L2, L3 = get_commandline_args()

    data = pd.read_excel(data_filename)
    data = data[data['L2'] == L2]
    data = data[data['L3'] == L3]

    ER_data = data[data['TER'] == 0]
    TER_data = data[data['TER'] == 1]

    merged_data = TER_data.merge(ER_data, on='R')
    energy_diff = merged_data['energy_x'] - merged_data['energy_y']
    energy_sum = merged_data['energy_x'] + merged_data['energy_y']
    R = merged_data['R']

    normalized_diff = energy_diff / energy_sum

    fig, ax = plt.subplots()

    ax.plot(R, normalized_diff, linestyle='', marker='o')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$\frac{F_\text{TER} - F_\text{ER}}{F_\text{TER} + F_\text{ER}}$')
    ax.set_title(r'Escaped energy diff $L_2 = {}, L_3 = {}$'.format(L2, L3))
    
    fig.tight_layout()
    fig.savefig(plot_filename)


    plt.show()


if __name__ == '__main__':

    main()
