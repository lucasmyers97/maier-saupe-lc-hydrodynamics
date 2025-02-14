import argparse
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

linestyles = ['-', '--', ':']
colors = ['tab:blue', 'tab:orange', 'tab:red']

def get_commandline_args():


    description = ('Plot total energy of metastable ER configurations as a '
                   'function of time for nematic configuration based on hdf5 files')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', 
                        help='folder where configuration energy data lives')
    parser.add_argument('--data_filenames',
                        nargs='+',
                        default='configuration_energy.h5',
                        help='files where configuration energy data lives')
    parser.add_argument('--output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--plot_filename',
                        default='configuration_energy.png',
                        help='filename of energy vs time plot')
    parser.add_argument('--final_timestep',
                        type=float,
                        default=None,
                        help='timestep at which to evaluate final energy')
    parser.add_argument('--title',
                        default='energy',
                        help='title of energy vs time plot')
    parser.add_argument('--L2_vals',
                        nargs='+',
                        help='Values of L2 for different configurations')

    args = parser.parse_args()

    output_folder = None
    if not args.output_folder:
        output_folder = args.data_folder
    else:
        output_folder = args.output_folder

    data_filenames = [os.path.join(args.data_folder, data_filename) 
                      for data_filename in args.data_filenames]
    plot_filename = os.path.join(output_folder, args.plot_filename)

    return data_filenames, plot_filename, args.title, args.final_timestep, args.L2_vals



def main():

    data_filenames, plot_filename, title, final_timestep, L2_vals = get_commandline_args()

    energies = []
    times = []
    for filename in data_filenames:
        file = h5py.File(filename)

        energies.append(
                np.array( file['mean_field_term'][:]
                         + file['entropy_term'][:]
                         + file['L1_elastic_term'][:]
                         + file['L2_elastic_term'][:]
                         + file['L3_elastic_term'][:] )
                )

        times.append( np.array( file['t'][:] ) )
        file.close()

    max_energy_range = max([np.max(energy) - np.min(energy) for energy in energies])

    fig, ax = plt.subplots(nrows=len(L2_vals), sharex=True, layout='constrained')
    fig.subplots_adjust(hspace=0.05)
    for energy, t, L2, i in zip(reversed(energies), 
                                reversed(times),
                                reversed(L2_vals), 
                                range(ax.shape[0])):

        final_idx = -1
        if final_timestep:
            final_idx = np.argmin( np.abs( t - final_timestep ) )

        ax[i].plot(t[:final_idx], energy[:final_idx], 
                   label=r'$L_2 = {}$'.format(L2),
                   color=colors[i],
                   linestyle=linestyles[i],
                   linewidth=1.5)

        # turn off all marks on top and bottom of subplots
        ax[i].spines.bottom.set_visible(False)
        ax[i].spines.top.set_visible(False)
        ax[i].xaxis.set_ticks_position('none')

        if i != 0:
            plt.setp(ax[i].yaxis.get_offset_text(), visible=False)


        # Make all plots use same y-scale
        actual_range = np.max(energy) - np.min(energy)
        room = max_energy_range - actual_range
        fudge = 1.0
        ax[i].set_ylim(bottom=np.min(energy) - room*0.3 - fudge,
                       top=np.max(energy) + room*0.3 + fudge)

    # Turn on top and bottom lines, turn on bottom ticks/labels
    ax[0].spines.top.set_visible(True)
    ax[-1].spines.bottom.set_visible(True)
    ax[-1].xaxis.set_ticks_position('bottom')

    # Add slashes to y-labels to indicate jumps
    d = 0.5
    l = 1.0
    kwargs = dict(marker=[(-l, -d), (l, d)], markersize=5,
                  linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)
    ax[0].plot([0, 1], [0, 0], transform=ax[0].transAxes, **kwargs)
    ax[-1].plot([0, 1], [1, 1], transform=ax[-1].transAxes, **kwargs)
    for i in range(ax.shape[0]):
        if i == 0 or i == ax.shape[0] - 1:
            continue

        ax[i].plot([0, 1], [0, 0], transform=ax[i].transAxes, **kwargs)
        ax[i].plot([0, 1], [1, 1], transform=ax[i].transAxes, **kwargs)
    
    fig.supxlabel(r'$t$')
    fig.supylabel(r'$F$')
    fig.legend(loc='upper right', frameon=True)#, bbox_to_anchor=(0.18, 0.92))

    fig.savefig(plot_filename)

    plt.show()



if __name__ == '__main__':

    main()

