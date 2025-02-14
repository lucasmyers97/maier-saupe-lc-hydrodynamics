import argparse
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    desc = 'Takes in spreadsheet of twist angular velocity (alpha) vs equilibrium energy'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_folder', help='folder where data lives')
    parser.add_argument('--filenames', 
                        nargs='+',
                        help='Name of excel type file with data')
    parser.add_argument('--plot_filename', help='Filename of output plot')
    parser.add_argument('--L2',
                        type=float,
                        help='L2 value to plot')
    parser.add_argument('--R',
                        type=float,
                        help='R value to plot')
    parser.add_argument('--dataset_titles',
                        nargs='+',
                        help='titles of the datasets to plot')
    parser.add_argument('--tf',
                        type=float,
                        help='final time')
    parser.add_argument('--drop_untwisted',
                        type=bool,
                        default=False,
                        help='whether to drop untwisted configuration data')

    args = parser.parse_args()

    filenames = []
    for filename in args.filenames:
        filenames.append( os.path.join(args.data_folder, filename) )

    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return filenames, plot_filename, args.L2, args.R, args.dataset_titles, args.tf, args.drop_untwisted



def main():

    filenames, plot_filename, L2, R, dataset_titles, tf, drop_untwisted = get_commandline_args()
    
    for filename, dataset_title in zip(filenames, dataset_titles):
        data = pd.read_excel(filename)
        plot_data = data[(data['L2'] == L2) 
                         & (data['dataset title'] == dataset_title) 
                         & (data['tf'] == tf)]

        # get rid of alpha = 0 to make it more balanced
        if (plot_data['alpha'] == 0).any() and drop_untwisted:
            plot_data.drop(plot_data[plot_data['alpha'] == 0].index, inplace=True)

        energy = plot_data['energy per length'] if 'energy per length' in plot_data.keys() else plot_data['energy']
        alpha = plot_data['alpha']

        polynomial_degree = 2
        lines = plt.plot(alpha, energy, ls='', marker='o', label=dataset_title)
        linecolor = lines[0].get_color()
        if plot_data.shape[0] > polynomial_degree:
            fit = np.polynomial.polynomial.Polynomial.fit(alpha, 
                                                          energy,
                                                          polynomial_degree)

            c, b, a = fit.convert().coef
            print(-b / (2 * a))
            print('Curvature is: {}'.format(a))

            x_fit, y_fit = fit.linspace(1000)

            plt.plot(x_fit, y_fit, color=linecolor)


    plt.title(r'$L_2 = {}$'.format(L2))
    plt.xlabel('twist angular velocity')
    plt.ylabel('equilibrium energy')
    plt.legend(frameon=True)

    plt.tight_layout()

    plt.savefig(plot_filename)

    plt.show()


if __name__ == '__main__':

    main()
