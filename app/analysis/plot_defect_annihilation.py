"""
Script reads in a series of defect locations from an hdf5 file with the
structure /defect/<coord> where <coord> is one of x, y, t for 2D defects.
It then outputs a regular plot, a logarithmic plot, and prints out a fit
parameter for the parabolic dynamics.
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_filenames():

    description = "Read in defect locations from hdf5, plot and find best fit"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--defect_filename', dest='defect_filename',
                        help='name of defect data file')
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder which will hold output plots')
    parser.add_argument('--plot_filename', dest='plot_filename',
                        help='filename of regularly-scaled x vs. t plot')
    parser.add_argument('--log_plot_filename', dest='log_plot_filename',
                        help='filename of log-scaled x vs. t plot')
    parser.add_argument('--squared_filename', dest='squared_filename',
                        help='filename of x vs. t^2 plot')
    parser.add_argument('--L3', dest='L3',
                        help='L3 value associated with annihilation')
    args = parser.parse_args()

    defect_filename = os.path.join(args.data_folder, args.defect_filename)
    plot_filename = os.path.join(args.output_folder, args.plot_filename)
    log_plot_filename = os.path.join(args.output_folder, args.log_plot_filename)
    squared_filename = os.path.join(args.output_folder, args.squared_filename)

    return plot_filename, log_plot_filename, defect_filename, squared_filename, args.L3



def separate_defects(t, x):

    pos_idx = x > 0
    neg_idx = x <= 0

    x_pos = x[pos_idx]
    x_neg = x[neg_idx]
    t_pos = t[pos_idx]
    t_neg = t[neg_idx]

    return [t_pos, t_neg], [x_pos, x_neg]


def order_points(t, x):
    
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    return t, x



def fit_sqrt(t, x):

    if x[0] > 0:
        sign = 1
    else:
        sign = -1

    p0 = [sign * np.sqrt(np.max(np.abs(x))), np.max(t)]
    popt, _ = curve_fit(lambda t, A, B: A * np.sqrt(np.abs(B - t)), t, x, p0=p0)
    return popt


def main():

    plot_filename, log_plot_filename, defect_filename, squared_filename, L3 = get_filenames()
    
    file = h5py.File(defect_filename)
    # t = np.array(file['defect']['t'][:])
    # x = np.array(file['defect']['x'][:])
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])

    t, x = separate_defects(t, x)
    for i in range(2):
        t[i], x[i] = order_points(t[i], x[i])

    offset = 0
    t[0] = t[0][offset:]
    t[1] = t[1][offset:]
    x[0] = x[0][offset:]
    x[1] = x[1][offset:]

    # A = fit_sqrt(t[0], x[0])
    # B = fit_sqrt(t[1], x[1])

    # t_fit = np.linspace(t[0][0], t[0][-1], num=1000)
    # x_fit = [A[0] * np.sqrt(A[1] - t_fit), B[0] * np.sqrt(B[1] - t_fit)]

    # plot regular scaling
    fig, ax = plt.subplots()
    ax.plot(t[0], x[0], label="+1/2 defect")
    ax.plot(t[1], x[1], label="-1/2 defect")
    # ax.plot(t_fit, x_fit[0], 
    #         label=r'$A_0 = {:.2E}, A_1 = {:.2E}$'.format(A[0], A[1]))
    # ax.plot(t_fit, x_fit[1], 
    #         label=r'$A_0 = {:.2E}, A_1 = {:.2E}$'.format(B[0], B[1]))
    
    # ax.set_title(r"$\pm 1/2$ defect annihilation, $x = A_0 \sqrt{A_1 - t}$")
    ax.set_title(r"$\pm 1/2$ defect annihilation, $L_3 = {}$".format(L3))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_filename)

    # plot squared values
    fig, ax = plt.subplots()
    ax.plot(t[0], x[0]**2, label="+1/2 defect")
    ax.plot(t[1], x[1]**2, label="-1/2 defect")

    ax.set_title(r"$\pm 1/2$ defect annihilation, $L_3 = {}$".format(L3))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x^2$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(squared_filename)

    plt.show()
   
if __name__ == "__main__":
    main()
