import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ('For TER configurations, plot alpha angle as a function '
                   'of position x and fit a quadratic to interpolate value '
                   'at the center.')
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
    x = data['Points:0']
    angle = data['angle']

    nan_mask = np.logical_not( np.isnan(angle) )

    angle = angle[nan_mask]
    x = x[nan_mask]

    fit_mask = np.logical_and(x < np.max(x) / 4, x > np.min(x) / 4)
    fit = np.polynomial.polynomial.polyfit(x[fit_mask], angle[fit_mask], 2)

    print(fit)
    print(fit[0])

    fit_x = np.linspace( np.min(x) / 4, np.max(x) / 4, 1000 )

    fig, ax = plt.subplots()
    ax.plot(x, angle)
    ax.plot(fit_x, fit[0] + fit[1]*fit_x + fit[2]*fit_x**2)
    ax.plot(0, fit[0], marker='o')
    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\alpha$')
    fig.tight_layout()
    fig.savefig(plot_filename)

    plt.show()



if __name__ == '__main__':

    main()

