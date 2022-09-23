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
    parser.add_argument('--squared_filename', dest='squared_filename',
                        help='filename of x vs. t^2 plot')
    parser.add_argument('--eps', dest='eps',
                        help='epsilon value associated with annihilation')
    parser.add_argument('--t0', dest='t0',
                        type=int,
                        default=0,
                        help='time to start annihilation plots at')
    parser.add_argument('--threshold', dest='threshold',
                        help='x-value at which to separate two defects')
    args = parser.parse_args()

    output_folder = args.output_folder
    if not output_folder:
        output_folder = args.data_folder

    defect_filename = os.path.join(args.data_folder, args.defect_filename)
    plot_filename = os.path.join(output_folder, args.plot_filename)
    squared_filename = os.path.join(output_folder, args.squared_filename)

    return plot_filename, defect_filename, squared_filename, args.eps, args.t0



def separate_defects(t, x, charge):

    pos_idx = charge > 0
    neg_idx = charge < 0

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



def get_annihilation_point(t, x):

    t_f = t[0][-1]
    x_f = (x[0][-1] + x[1][-1]) / 2

    return t_f, x_f



def fit_sqrt(t, x):

    if x[0] > 0:
        sign = 1
    else:
        sign = -1

    p0 = [sign * np.sqrt(np.max(np.abs(x))), np.max(t)]
    popt, _ = curve_fit(lambda t, A, B: A * np.sqrt(np.abs(B - t)), t, x, p0=p0)
    return popt


def main():

    plot_filename, defect_filename, squared_filename, eps, t0 = get_filenames()
    
    file = h5py.File(defect_filename)
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])
    charge = np.array(file['charge'][:])

    t, x = separate_defects(t, x, charge)
    for i in range(2):
        t[i], x[i] = order_points(t[i], x[i])

    t[0] = t[0][t0:]
    t[1] = t[1][t0:]
    x[0] = x[0][t0:]
    x[1] = x[1][t0:]

    t_f, x_f = get_annihilation_point(t, x)
    print("Annihilation point (t_f, x_f) is: ({}, {})".format(t_f, x_f))

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
    ax.set_title(r"$\pm 1/2$ defect annihilation, $\epsilon = {}$".format(eps))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_filename)

    # plot squared values
    fig, ax = plt.subplots()
    ax.plot(t[0], x[0]**2, label="+1/2 defect")
    ax.plot(t[1], x[1]**2, label="-1/2 defect")

    ax.set_title(r"$\pm 1/2$ defect annihilation, $\epsilon = {}$".format(eps))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x^2$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(squared_filename)

    plt.show()
   
if __name__ == "__main__":
    main()
