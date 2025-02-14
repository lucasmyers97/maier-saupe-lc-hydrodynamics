import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

def get_commandline_args():


    description = ('Plot alpha angle vs. L2 for TER configurations')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where alpha angle data lives')
    parser.add_argument('--data_filename',
                        help='excel file where alpha angle data lives')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filename',
                        help='filename of alpha vs. position plot')
    parser.add_argument('--title',
                        help='title of alpha vs. position plot')
    parser.add_argument('--R',
                        type=float,
                        help='Radius of cylinder')
    parser.add_argument('--L3',
                        type=float,
                        help='L3 value of systems')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    plot_filename = os.path.join(output_folder, args.plot_filename)

    return data_filename, plot_filename, args.title, args.R, args.L3



def main():

    data_filename, plot_filename, title, R, L3 = get_commandline_args()

    data = pd.read_excel(data_filename)
    data = data[data['R'] == R]
    data = data[data['L3'] == L3]
    TER_data = data[data['TER'] == 1]
    ER_data = data[data['TER'] == 0]

    fig, ax = plt.subplots()
    ax.plot(TER_data['L2'], TER_data['alpha'], marker='s', linestyle='--', label='TER')
    ax.plot(ER_data['L2'], ER_data['alpha'], marker='^', linestyle='-.', label='ER')
    ax.set_title(title)
    ax.set_xlabel(r'$L_2$')
    ax.set_ylabel(r'$\alpha(r = 0)$')
    fig.legend(frameon=True, loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_filename)

    plt.show()



if __name__ == '__main__':

    main()

