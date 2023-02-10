"""
Script reads in a series of defect locations from an hdf5 file with the
structure /defect/<coord> where <coord> is one of x, y, t for 2D defects.
It then outputs a plot of x- and y-coordinates vs time
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
    parser.add_argument('--x_plot_filename', dest='x_plot_filename',
                        help='filename of regularly-scaled x vs. t plot')
    parser.add_argument('--y_plot_filename', dest='y_plot_filename',
                        help='filename of regularly-scaled y vs. t plot')
    parser.add_argument('--eps', dest='eps',
                        help='epsilon value associated with annihilation')
    args = parser.parse_args()

    output_folder = args.output_folder
    if not output_folder:
        output_folder = args.data_folder

    defect_filename = os.path.join(args.data_folder, args.defect_filename)
    x_plot_filename = os.path.join(output_folder, args.x_plot_filename)
    y_plot_filename = os.path.join(output_folder, args.y_plot_filename)

    return x_plot_filename, y_plot_filename, defect_filename, args.eps



def order_points(t, x):
    
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    return t, x



def order_points(t, x, y):
    
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    y = y[idx]
    return t, x, y



def main():

    x_plot_filename, y_plot_filename, defect_filename, eps = get_filenames()
    
    file = h5py.File(defect_filename)
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])
    y = np.array(file['y'][:])

    t, x, y = order_points(t, x, y)

    # plot regular scaling
    fig, ax = plt.subplots()
    ax.plot(t, x, label="single defect trajectory")
    
    ax.set_title(r"single defect trajectory, $\epsilon = {}$".format(eps))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(x_plot_filename)

    fig, ax = plt.subplots()
    ax.plot(t, y, label="single defect trajectory")
    
    ax.set_title(r"single defect trajectory, $\epsilon = {}$".format(eps))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$y$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(y_plot_filename)

    plt.show()
   
if __name__ == "__main__":
    main()
