"""
Script reads in a series of defect locations from an hdf5 file with the
structure /defect/<coord> where <coord> is one of x, y, t for 2D defects.
It then outputs a regular plot, a logarithmic plot, and prints out a fit
parameter for the parabolic dynamics.

Also plots energy configurations -- I need to fix that when I have a chance
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
    # parser.add_argument('--data_folder_1', dest='data_folder_1',
    #                     help='folder where defect location data lives')
    # parser.add_argument('--data_folder_2', dest='data_folder_2',
    #                     help='folder where defect location data lives')
    parser.add_argument('--defect_filename', dest='defect_filename',
                        help='name of defect data file')
    parser.add_argument('--energy_filename', dest='energy_filename',
                        help='name of energy data file')
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder which will hold output plots')
    parser.add_argument('--plot_filename', dest='plot_filename',
                        help='filename of regularly-scaled x vs. t plot')
    parser.add_argument('--squared_filename', dest='squared_filename',
                        help='filename of x vs. t^2 plot')
    parser.add_argument('--velocity_filename', dest='velocity_filename',
                        help='filename of velocity vs 1/distance plot')
    parser.add_argument('--avg_velocity_filename', dest='avg_velocity_filename',
                        help='filename of average velocity vs 1/distance plot')
    parser.add_argument('--eps', dest='eps',
                        help='epsilon value associated with annihilation')
    parser.add_argument('--n_smooth',
                        dest='n_smooth',
                        type=int,
                        help='number of points with which to do moving average smoothing')
    parser.add_argument('--start_cutoff',
                        dest='start_cutoff',
                        type=float,
                        help='percentage of start cutoff for velocity')
    parser.add_argument('--end_cutoff',
                        dest='end_cutoff',
                        type=float,
                        help='percentage of end cutoff for velocity')
    args = parser.parse_args()

    output_folder = args.output_folder
    if not output_folder:
        output_folder = args.data_folder

    defect_filename = os.path.join(args.data_folder, args.defect_filename)
    energy_filename = os.path.join(args.data_folder, args.energy_filename)
    # defect_filename_1 = os.path.join(args.data_folder_1, args.defect_filename)
    # defect_filename_2 = os.path.join(args.data_folder_2, args.defect_filename)
    plot_filename = os.path.join(output_folder, args.plot_filename)
    squared_filename = os.path.join(output_folder, args.squared_filename)
    velocity_filename = os.path.join(output_folder, args.velocity_filename)
    avg_velocity_filename = os.path.join(output_folder, "smoothed_velocity_{}.png".format(args.n_smooth))

    return (plot_filename, defect_filename, energy_filename,
            # defect_filename_1, defect_filename_2, 
            squared_filename, 
            velocity_filename, avg_velocity_filename, args.eps, args.n_smooth,
            args.start_cutoff, args.end_cutoff)



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



def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



def fit_sqrt(t, x):

    if x[0] > 0:
        sign = 1
    else:
        sign = -1

    p0 = [sign * np.sqrt(np.max(np.abs(x))), np.max(t)]
    popt, _ = curve_fit(lambda t, A, B: A * np.sqrt(np.abs(B - t)), t, x, p0=p0)
    return popt


def main():

    (plot_filename, defect_filename, energy_filename,
     # defect_filename_1, defect_filename_2,
     squared_filename, velocity_filename, avg_velocity_filename, eps,
     n_smooth, start_cutoff, end_cutoff) = get_filenames()
    
    file = h5py.File(defect_filename)
    t = np.array(file['t'][:])
    x = np.array(file['x'][:])
    charge = np.array(file['charge'][:])

    # file_1 = h5py.File(defect_filename_1)
    # t_1 = np.array(file_1['t'][:])
    # x_1 = np.array(file_1['x'][:])
    # charge_1 = np.array(file_1['charge'][:])
    # 
    # file_2 = h5py.File(defect_filename_2)
    # t_2 = np.array(file_2['t'][:])
    # x_2 = np.array(file_2['x'][:])
    # charge_2 = np.array(file_2['charge'][:])

    t, x = separate_defects(t, x, charge)
    for i in range(2):
        t[i], x[i] = order_points(t[i], x[i])

    # t_1, x_1 = separate_defects(t_1, x_1, charge_1)
    # for i in range(2):
    #     t_1[i], x_1[i] = order_points(t_1[i], x_1[i])

    # t_2, x_2 = separate_defects(t_2, x_2, charge_2)
    # for i in range(2):
    #     t_2[i], x_2[i] = order_points(t_2[i], x_2[i])

    t[0] = t[0]
    t[1] = t[1]
    x[0] = x[0]
    x[1] = x[1]
    # t_1[0] = t_1[0]
    # t_1[1] = t_1[1]
    # x_1[0] = x_1[0]
    # x_1[1] = x_1[1]
    # t_2[0] = t_2[0]
    # t_2[1] = t_2[1]
    # x_2[0] = x_2[0]
    # x_2[1] = x_2[1]

    t_f, x_f = get_annihilation_point(t, x)
    print("Annihilation point (t_f, x_f) is: ({}, {})".format(t_f, x_f))

    x_avg = [moving_average(x[0], n_smooth), moving_average(x[1], n_smooth)]
    t_avg = [moving_average(t[0], n_smooth), moving_average(t[1], n_smooth)]

    n_avg = x_avg[0].shape[0]
    start_cutoff = int(n_avg * start_cutoff)
    end_cutoff = int(n_avg * end_cutoff)
    if end_cutoff == 0:
        end_cutoff = 1

    # plot velocity values
    v = [np.diff(x[0]) / (t[0][:-1] - t[0][1:]),
         np.diff(x[1]) / (t[1][:-1] - t[1][1:])]

    v_avg = [np.diff(x_avg[0]) / (t_avg[0][:-1] - t_avg[0][1:]), 
             np.diff(x_avg[1]) / (t_avg[1][:-1] - t_avg[1][1:]) ]

    # do linear fit for velocities
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, 
                           1 / (x_avg[0][start_cutoff:-(end_cutoff + 1)] - x_f),
                           v_avg[0][start_cutoff:-end_cutoff])
    print(popt)
    v_avg_fit = 1 / (x_avg[0][start_cutoff:-(end_cutoff + 1)] - x_f) * popt[0] + popt[1]

    # plot regular scaling
    fig, ax = plt.subplots()
    ax.plot(t[0], x[0], label="+1/2 defect")
    ax.plot(t[1], x[1], label="-1/2 defect")
    # ax.plot(t_1[0], x_1[0], label="+1/2 defect coarse")
    # ax.plot(t_1[1], x_1[1], label="-1/2 defect coarse")
    # ax.plot(t_2[0], x_2[0], label="+1/2 defect fine")
    # ax.plot(t_2[1], x_2[1], label="-1/2 defect fine")
    # ax.plot(t_avg[0], x_avg[0], label="+1/2 defect smoothed")
    # ax.plot(t_avg[1], x_avg[1], label="-1/2 defect smoothed")
    
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

    # plot regularly-calculated velocities
    fig, ax = plt.subplots()
    ax.plot(1 / (x[0][:-1] - x_f), v[0])
    # ax.plot(1 / (x[1][:-1] - x_f), v[1])

    ax.set_title(r"$\pm 1/2$ defect annihilation unsmoothed, $\epsilon = {}$".format(eps))
    ax.set_xlabel(r"$1 / x$")
    ax.set_ylabel(r"$v$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(velocity_filename)

    # plot moving average velocities
    fig, ax = plt.subplots()
    ax.plot(1 / (x_avg[0][start_cutoff:-(end_cutoff + 1)] - x_f), 
            v_avg[0][start_cutoff:-end_cutoff], label="+1/2 defect")
    ax.plot(1 / (x_avg[0][start_cutoff:-(end_cutoff + 1)] - x_f), 
            v_avg_fit, label="Curve fit")

    ax.set_title(r"defect velocity $n$-smoothed, $\epsilon = {}$, $n = {}$".format(eps, n_smooth))
    ax.set_xlabel(r"$1 / x$")
    ax.set_ylabel(r"$v$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(avg_velocity_filename)

    # plot energy squared
    file = h5py.File(energy_filename)
    t = np.array(file['t'][:])
    dE_dQ_squared = np.array(file['dE_dQ_squared'][:])

    fig, ax = plt.subplots()
    ax.plot(t[100:], dE_dQ_squared[100:])

    ax.set_title(r"$\int_\Omega \left(dE/dQ\right)^2$ vs. $t$")
    ax.set_ylabel(r"$\int_\Omega \left(dE/dQ\right)^2$")
    ax.set_xlabel(r"$t$")

    # plot dE/dt
    E = np.array(file['L1_elastic_term'][:]
                 + file['L2_elastic_term'][:]
                 + file['L3_elastic_term'][:]
                 - file['entropy_term'][:]
                 + file['mean_field_term'][:])

    fig, ax = plt.subplots()
    ax.plot(t[101:], (np.diff(E) / np.diff(t))[100:] )

    ax.set_title(r"$\int_\Omega dE/dt$ vs. $t$")
    ax.set_ylabel(r"$\int_\Omega dE/dt$")
    ax.set_xlabel(r"$t$")

    plt.show()
   
if __name__ == "__main__":
    main()
