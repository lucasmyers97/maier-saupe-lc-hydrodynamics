import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']
markers = ['s', 'o', '^', 'X', 'P']
linestyles = ['-', '--', '-.', ':', (0, (5, 10))]

def get_commandline_args():


    description = ('Plot energies of ER and TER configurations as a function '
                   'of L3, for single L2, including decays')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where energy data lives')
    parser.add_argument('--data_filename',
                        help='excel file where energy data lives')
    parser.add_argument('--L2',
                        type=int,
                        help='L2 values to plot')
    parser.add_argument('--R',
                        type=int,
                        help='R value to plot')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filenames',
                        nargs=2,
                        help='filenames of energy vs. L3 plot and alpha vs. L3 plot')
    parser.add_argument('--title',
                        help='title of energy vs. L3 plot')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    plot_filenames = [os.path.join(output_folder, plot_filename) for plot_filename in args.plot_filenames]

    return data_filename, args.L2, args.R, plot_filenames, args.title



def main():

    data_filename, L2, R, plot_filenames, title = get_commandline_args()

    data = pd.read_excel(data_filename)
    data = data[data['R'] == R]
    TER_data = data[data['TER'] == 1]
    ER_data = data[data['TER'] == 0]

    ER_data = data[(data['TER'] == 0) & (data['metastable'] == 0)]
    metastable_data = data[(data['TER'] == 0) & (data['metastable'] == 1)]

    TER = TER_data[TER_data['L2'] == L2]
    ER = ER_data[ER_data['L2'] == L2]
    metastable_data = metastable_data[metastable_data['L2'] == L2]

    fig, ax = plt.subplots()
    ax.plot(metastable_data['L3'], metastable_data['energy'], 
            ls='',
            marker=markers[0],
            color=colors[0],
            label='ER')
    ax.plot(TER['L3'], TER['energy'], 
            ls='', 
            marker=markers[1], 
            color=colors[1],
            label='TER')
    ax.plot(ER['L3'], ER['energy'], 
            ls='', 
            marker=markers[2],
            color=colors[2],
            label=r'$\text{ER} \to \text{TER}$')

    ax.set_title(title)
    ax.set_xlabel(r'$L_3$')
    ax.set_ylabel(r'$F$')

    fig.legend(frameon=True, loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_filenames[0])

    fig, ax = plt.subplots()
    ax.plot(metastable_data['L3'], metastable_data['alpha'], 
            ls='',
            marker=markers[0],
            color=colors[0],
            label='ER')
    ax.plot(TER['L3'], TER['alpha'], 
            ls='', 
            marker=markers[1], 
            color=colors[1],
            label='TER')
    ax.plot(ER['L3'], ER['alpha'], 
            ls='', 
            marker=markers[2],
            color=colors[2],
            label=r'$\text{ER} \to \text{TER}$')

    ax.set_title(title)
    ax.set_xlabel(r'$L_3$')
    ax.set_ylabel(r'$\alpha$')

    fig.legend(frameon=True, loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_filenames[1])

    plt.show()



if __name__ == '__main__':

    main()

