import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ('Plot energies of ER and TER configurations as a function '
                   'of L2.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where energy data lives')
    parser.add_argument('--data_filename',
                        help='excel file where energy data lives')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filenames',
                        nargs=2,
                        help='filenames of energy vs. L3 plot and alpha vs. L3 plot')
    parser.add_argument('--title',
                        help='title of energy vs. L2 plot')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    plot_filenames = [os.path.join(output_folder, plot_filename) for plot_filename in args.plot_filenames]

    return data_filename, plot_filenames, args.title



def main():

    data_filename, plot_filenames, title = get_commandline_args()

    data = pd.read_excel(data_filename)
    TER_data = data[data['TER'] == 1]
    ER_data = data[(data['TER'] == 0) & (data['metastable'] == 0)]
    metastable_data = data[(data['TER'] == 0) & (data['metastable'] == 1)]

    fig, ax = plt.subplots()
    ax.plot(TER_data['L3'], TER_data['energy'], ls='', marker='o', label='TER initialized')
    ax.plot(ER_data['L3'], ER_data['energy'], ls='', marker='o', label='ER initialized')
    ax.plot(metastable_data['L3'], metastable_data['energy'], ls='', marker='o', label='ER metastable')
    ax.set_title(title)
    ax.set_xlabel(r'$L_3$')
    ax.set_ylabel(r'Energy')
    fig.legend(frameon=True, loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_filenames[0])

    fig, ax = plt.subplots()
    ax.plot(TER_data['L3'], TER_data['alpha'], ls='', marker='o', label='TER initialized')
    ax.plot(ER_data['L3'], ER_data['alpha'], ls='', marker='o', label='ER initialized')
    ax.plot(metastable_data['L3'], metastable_data['alpha'], ls='', marker='o', label='ER metastable')
    ax.set_title(title)
    ax.set_xlabel(r'$L_3$')
    ax.set_ylabel(r'$\alpha$')
    fig.legend(frameon=True, loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_filenames[1])

    plt.show()



if __name__ == '__main__':

    main()

