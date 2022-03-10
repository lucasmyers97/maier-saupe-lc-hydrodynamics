import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

if __name__ == "__main__":

    description = 'Plot scaling of supercomputer for simulations'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--profiling_folder', dest='profiling_folder',
                        help='folder where profiling data is stored')
    parser.add_argument('--profiling_spreadsheet', dest='profiling_spreadsheet',
                        help='filename of spreadsheet holding profiling data')
    args = parser.parse_args()

    filename = os.path.join(args.profiling_folder, args.profiling_spreadsheet)
    prof_data = pd.read_excel(filename)

    # small_sim = prof_data[prof_data['refines'] == 5]
    # num_cores = small_sim['processors'].values
    # wall_times = small_sim['wall time'].values
    # cpu_times = small_sim['cpu time'].values
    # iterations = small_sim['iterations'].values
    # assembly_time = small_sim['assembly time'].values

    # fig, ax = plt.subplots()
    # ax.plot(num_cores, wall_times, marker='o', label="Actual scaling")
    # ax.plot(num_cores, (wall_times[0]*num_cores[0])/num_cores, label="Ideal (linear) scaling")
    # ax.set_xscale("log")
    # ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    #               labels=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    # ax.set_yscale("log")
    # ax.set_title("Wall time scaling for 148,955 DoFs")
    # ax.set_xlabel("Number of processors")
    # ax.set_ylabel("Wall time (s)")
    # ax.legend()
    # plt.tight_layout()
    # fig.savefig("walltime_small_simulation.png")


    # fig, ax = plt.subplots()
    # ax.plot(num_cores, assembly_time, marker='o', label="Actual scaling")
    # ax.plot(num_cores, (assembly_time[0]*num_cores[0])/num_cores, label="Ideal (linear) scaling")
    # ax.set_xscale("log")
    # ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    #               labels=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    # ax.set_yscale("log")
    # ax.set_title("Assembly time scaling for 148,955 DoFs")
    # ax.set_xlabel("Number of processors")
    # ax.set_ylabel("Assembly time (s)")
    # ax.legend()
    # plt.tight_layout()
    # fig.savefig("assembly_time_small_simulation.png")

    # fig, ax = plt.subplots()
    # ax.plot(num_cores, cpu_times, marker='o')
    # ax.set_ylim(0, cpu_times[-1]*1.1)
    # ax.set_title("CPU time scaling for 148,955 DoFs")
    # ax.set_xlabel("Number of processors")
    # ax.set_ylabel("Total CPU time (across all processors)")
    # plt.tight_layout()
    # fig.savefig("cputime_small_simulation.png")

    # fig, ax = plt.subplots()
    # ax.plot(num_cores, cpu_times / num_cores, marker='o', label="Actual scaling")
    # ax.plot(num_cores, (cpu_times[0])/num_cores, label="Ideal (linear) scaling")
    # ax.set_xscale("log")
    # ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    #               labels=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    # ax.set_yscale("log")
    # ax.set_title("CPU time / core scaling for 148,955 DoFs")
    # ax.set_xlabel("Number of processors")
    # ax.set_ylabel("CPU time per core")
    # ax.legend()
    # plt.tight_layout()
    # fig.savefig("cputime_per_core_small_simulation.png")

    # fig, ax = plt.subplots()
    # ax.plot(num_cores, iterations, marker='o')
    # ax.set_ylim([0, 500])
    # ax.set_title("Solver iterations for 148,955 DoFs")
    # ax.set_xlabel("Number of processors")
    # ax.set_ylabel("Number of solver iterations")
    # plt.tight_layout()
    # fig.savefig("iterations_small_simulation.png")

    large_sim = prof_data[prof_data['refines'] == 7]
    # large_sim = prof_data[prof_data['refines'] == 10]
    num_cores = large_sim['processors'].values
    wall_times = large_sim['wall time'].values
    cpu_times = large_sim['cpu time'].values
    iterations = large_sim['iterations'].values

    fig, ax = plt.subplots()
    ax.plot(num_cores, wall_times, marker='o', label="Actual scaling")
    ax.plot(num_cores, (wall_times[0]*num_cores[0])/num_cores, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_xticks([16, 32, 64, 128, 256, 512],
                  labels=[16, 32, 64, 128, 256, 512])
    ax.set_yscale("log")
    # ax.set_title("Scaling for 1,250,235 DoFs in 3D")
    ax.set_title("Scaling for 10,733,445 DoFs in 3D")
    # ax.set_title("Scaling for 5,232,645 DoFs in 2D")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Wall time (s)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("walltime_large_simulation_3D.png")
    # fig.savefig("walltime_large_simulation_2D.png")

    fig, ax = plt.subplots()
    ax.plot(num_cores, cpu_times, marker='o')
    ax.set_ylim(0, cpu_times[-1]*1.1)
    # ax.set_title("CPU time scaling for 1,250,235 DoFs 3D")
    ax.set_title("CPU time for 10,733,445 DoFs in 3D")
    # ax.set_title("CPU time scaling for 5,232,645 DoFs 2D")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Total CPU time (s)")
    plt.tight_layout()
    fig.savefig("cputime_large_simulation_3D.png")
    # fig.savefig("cputime_large_simulation_2D.png")

    fig, ax = plt.subplots()
    ax.plot(num_cores, cpu_times / num_cores, marker='o', label="Actual scaling")
    ax.plot(num_cores, (cpu_times[0])/num_cores, label="Ideal (linear) scaling")
    ax.set_xscale("log")
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024],
                  labels=[16, 32, 64, 128, 256, 512, 1024])
    ax.set_yscale("log")
    # ax.set_title("CPU time / core for 1,250,235 DoFs 3D")
    ax.set_title("CPU time / core for 10,733,445 DoFs in 3D")
    # ax.set_title("CPU time / core for 5,232,645 DoFs 2D")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("CPU time per core (s)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("cputime_per_core_large_simulation_3D.png")
    # fig.savefig("cputime_per_core_large_simulation_2D.png")

    # fig, ax = plt.subplots()
    # ax.plot(num_cores, iterations, marker='o')
    # ax.set_ylim([0, 400])
    # ax.set_title("Solver iterations for 1,250,235 DoFs")
    # ax.set_xlabel("Number of processors")
    # ax.set_ylabel("Number of solver iterations")
    # plt.tight_layout()
    # fig.savefig("iterations_large_simulation.png")

    plt.tight_layout()
    plt.show()
