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
    parser.add_argument('--filename', help='Name of excel type file with data')
    parser.add_argument('--plot_filename', help='Filename of output plot')
    parser.add_argument('--L2',
                        type=float,
                        help='L2 value to plot')

    args = parser.parse_args()

    filename = os.path.join(args.data_folder, args.filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return filename, plot_filename, args.L2



def main():

    filename, plot_filename, L2 = get_commandline_args()
    
    data = pd.read_excel(filename)
    plot_data = data[data['L2'] == L2]

    polynomial_degree = 2
    fit = np.polynomial.polynomial.Polynomial.fit(plot_data['alpha'], 
                                                  plot_data['energy'],
                                                  polynomial_degree)

    c, b, a = fit.convert().coef
    print(-b / (2 * a))
    print('Curvature is: {}'.format(a))

    x_fit, y_fit = fit.linspace(1000)

    plt.plot(x_fit, y_fit)
    plt.plot(plot_data['alpha'], plot_data['energy'], ls='', marker='o')
    plt.title(r'$L_2 = {}$'.format(L2))
    plt.xlabel('twist angular velocity')
    plt.ylabel('equilibrium energy')

    plt.tight_layout()

    plt.savefig(plot_filename)

    plt.show()


if __name__ == '__main__':

    main()
