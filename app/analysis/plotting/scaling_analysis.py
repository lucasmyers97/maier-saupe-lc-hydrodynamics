"""
This script just reads in scaling data (e.g. times vs. size of problem) from
csv files, and then plots them nicely.
Originally this was used for the XSEDE proposal.
"""

import argparse
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    description = 'Plot scaling of supercomputer for simulations'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--profiling_folder', dest='profiling_folder',
                        help='folder where profiling data is stored')
    parser.add_argument('--profiling_spreadsheet', dest='profiling_spreadsheet',
                        help='filename of spreadsheet holding profiling data')
    args = parser.parse_args()

    filename = os.path.join(args.profiling_folder, args.profiling_spreadsheet)

    return filename



def plot_wall_times(num_cores, wall_times):

    fig, ax = plt.subplots()
    ax.plot(num_cores, wall_times, marker='o', label="Actual scaling")
    ax.plot(num_cores, (wall_times[0]*num_cores[0])/num_cores, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_xticks([32, 64, 128, 256, 512, 1024],
                  labels=[32, 64, 128, 256, 512, 1024])
    ax.set_yscale("log")
    ax.set_title("Scaling for 12,684,525 DoFs in 3D")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Wall time (s)")
    ax.legend()
    plt.tight_layout()

    return fig, ax



def plot_cpu_times(num_cores, cpu_times):

    fig, ax = plt.subplots()
    ax.plot(num_cores, cpu_times, marker='o')
    ax.set_ylim(0, cpu_times[-1]*1.1)
    ax.set_title("CPU time for 12,684,525 DoFs in 3D")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Total CPU time (s)")
    plt.tight_layout()

    return fig, ax



def plot_cpu_times_per_core(num_cores, cpu_times):

    fig, ax = plt.subplots()
    ax.plot(num_cores, cpu_times / num_cores, marker='o', label="Actual scaling")

    x = np.log(num_cores)
    y = np.log(cpu_times / num_cores)
    res = stats.linregress(x, y)

    plt.plot(np.exp(x), 
             np.exp(res.intercept + res.slope*x), 
             label=r'$\text{{time}}/\text{{core}} = A/r^n,\\ A = {:.1e}, n = {:.2f}$'.format(np.exp(res.intercept), -res.slope))

    ax.set_xscale("log")
    ax.set_xticks([32, 64, 128, 256, 512, 1024],
                  labels=[32, 64, 128, 256, 512, 1024])
    ax.set_yscale("log")
    ax.set_title("CPU time / core for 12,684,525 DoFs in 3D")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("CPU time per core (s)")
    ax.legend()
    plt.tight_layout()

    return fig, ax



if __name__ == "__main__":

    filename = get_commandline_args()
    prof_data = pd.read_excel(filename)

    num_cores = prof_data['processors'].values
    wall_times = prof_data['wall time'].values
    cpu_times = prof_data['cpu time'].values

    fig, ax = plot_wall_times(num_cores, wall_times)
    fig.savefig("walltime_large_simulation_3D.png")

    fig, ax = plot_cpu_times(num_cores, cpu_times)
    fig.savefig("cputime_large_simulation_3D.png")

    fig, ax = plot_cpu_times_per_core(num_cores, cpu_times)
    fig.savefig("cputime_per_core_large_simulation_3D.png")

    plt.tight_layout()
    plt.show()
