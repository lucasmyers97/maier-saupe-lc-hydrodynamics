import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ('For TER configurations, plot beta angle as a function '
                   'of position x and compare against analytic ER profile')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where alpha angle data lives')
    parser.add_argument('--data_filename',
                        help='file where alpha angle data lives')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filename',
                        help='filename of alpha vs. position plot')
    parser.add_argument('--title',
                        help='title of alpha vs. position plot')

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

    data = pd.read_csv(data_filename)
    x = data['Points:1']
    angle = data['angle']

    nan_mask = np.logical_not( np.isnan(angle) )

    angle = angle[nan_mask]
    x = x[nan_mask]

    R = np.max(x)

    analytic_angle = 2 * np.arctan(np.abs(x) / R)

    fig, ax = plt.subplots()
    ax.plot(x, analytic_angle, label=r'$2\text{arctan}(r/R)$')
    ax.plot(x, angle, label=r'$\beta$')
    ax.set_title(title)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\beta$')
    fig.tight_layout()
    fig.savefig(plot_filename)

    fig.legend(frameon=True)

    plt.show()



if __name__ == '__main__':

    main()

