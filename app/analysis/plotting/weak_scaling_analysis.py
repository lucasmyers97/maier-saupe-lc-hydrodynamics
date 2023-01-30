"""
This script does essentially the same thing as the `scaling_analysis` script,
but I think works on different data (or gives a different analysis or
something). 
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

if __name__ == "__main__":

    description = 'Plot weak scaling of supercomputer for simulations'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--profiling_folder', dest='profiling_folder',
                        help='folder where profiling data is stored')
    parser.add_argument('--profiling_spreadsheet', dest='profiling_spreadsheet',
                        help='filename of spreadsheet holding profiling data')
    args = parser.parse_args()

    filename = os.path.join(args.profiling_folder, args.profiling_spreadsheet)
    prof_data = pd.read_excel(filename)

    D2_sim = prof_data[prof_data['dimension'] == 2]
    num_cores = D2_sim['processors'].values
    num_dofs = D2_sim['dofs'].values
    wall_times = D2_sim['wall time'].values
    cpu_times = D2_sim['cpu time'].values
    iterations = D2_sim['iterations'].values

    fig, ax = plt.subplots()
    ax.plot(num_dofs, wall_times, marker='o', label="Actual scaling")
    ax.plot(num_dofs, (wall_times[0]/num_dofs[0])*num_dofs, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("2D wall time for 512 processors")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Wall time (s)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("walltime_weak_2D_simulation.png")

    fig, ax = plt.subplots()
    ax.plot(num_dofs, cpu_times, marker='o', label="Actual scaling")
    ax.plot(num_dofs, (cpu_times[0]/num_dofs[0])*num_dofs, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("2D CPU time for 512 cores")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Total CPU time (across all processors)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("cputime_weak_2D_simulation.png")

    fig, ax = plt.subplots()
    ax.plot(num_dofs, cpu_times/512, marker='o', label="Actual scaling")
    ax.plot(num_dofs, (cpu_times[0]/num_dofs[0])*num_dofs/512, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("2D CPU time per core for 512 cores")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("CPU time per core (s)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("cputime_per_core_weak_2D_simulation.png")

    fig, ax = plt.subplots()
    ax.plot(num_dofs, iterations, marker='o')
    ax.set_ylim([0, np.max(iterations)])
    ax.set_title("2D solver iterations for 512 processors")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Number of solver iterations")
    plt.tight_layout()
    fig.savefig("iterations_weak_2D_simulation.png")

    D3_sim = prof_data[prof_data['dimension'] == 3]
    num_cores = D3_sim['processors'].values
    wall_times = D3_sim['wall time'].values
    cpu_times = D3_sim['cpu time'].values
    iterations = D3_sim['iterations'].values
    num_dofs = D3_sim['dofs'].values

    fig, ax = plt.subplots()
    ax.plot(num_dofs, wall_times, marker='o', label="Actual scaling")
    ax.plot(num_dofs, (wall_times[0]/num_dofs[0])*num_dofs, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("3D wall time for 512 processors")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Wall time (s)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("walltime_weak_3D_simulation.png")

    fig, ax = plt.subplots()
    ax.plot(num_dofs, cpu_times, marker='o', label="Actual scaling")
    ax.plot(num_dofs, (cpu_times[0]/num_dofs[0])*num_dofs, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("3D CPU time for 512 cores")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Total CPU time (across all processors)")
    plt.tight_layout()
    fig.savefig("cputime_weak_3D_simulation.png")

    fig, ax = plt.subplots()
    ax.plot(num_dofs, cpu_times/512, marker='o', label="Actual scaling")
    ax.plot(num_dofs, (cpu_times[0]/num_dofs[0])*num_dofs/512, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("3D CPU time per core for 512 cores")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("CPU time per core (s)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("cputime_per_core_weak_3D_simulation.png")

    fig, ax = plt.subplots()
    ax.plot(num_dofs, iterations, marker='o')
    ax.set_ylim([0, np.max(iterations)])
    ax.set_title("3D solver iterations for 512 processors")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Number of solver iterations")
    plt.tight_layout()
    fig.savefig("iterations_weak_3D_simulation.png")

    plt.tight_layout()
    plt.show()
