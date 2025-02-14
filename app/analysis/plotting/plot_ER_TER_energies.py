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
    parser.add_argument('--plot_filename',
                        help='filename of energy vs. L2 plot')
    parser.add_argument('--title',
                        help='title of energy vs. L2 plot')

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

    data = pd.read_excel(data_filename)
    TER_data = data[data['TER'] == 1]
    ER_data = data[data['TER'] == 0]

    fig, ax = plt.subplots()
    ax.plot(TER_data['L2'], TER_data['energy'])
    ax.plot(ER_data['L2'], ER_data['energy'])
    ax.set_title(title)
    ax.set_xlabel(r'$L_2$')
    ax.set_ylabel(r'Energy')
    fig.tight_layout()
    fig.savefig(plot_filename)

    plt.show()



if __name__ == '__main__':

    main()

