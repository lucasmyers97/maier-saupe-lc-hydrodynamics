import argparse
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# plt.style.use('science')
# mpl.rcParams['figure.dpi'] = 300

linestyles = ['-', '--', '-.', ':']

def get_commandline_args():

    parser = argparse.ArgumentParser(description='Plot energies for configurations with cubic term')
    parser.add_argument('--data_folder', help='Folder holding data')
    parser.add_argument('--bulk_filename', help='Filename holding bulk energy data')
    parser.add_argument('--configuration_filename', help='Filename holding energy data for configurations')
    parser.add_argument('--output_filename', help='name of png plot file')
    parser.add_argument('--B_vals',
                        nargs='+',
                        type=float,
                        help='B values of configurations')

    args = parser.parse_args()

    bulk_filename = os.path.join(args.data_folder, args.bulk_filename)
    configuration_filename = os.path.join(args.data_folder, args.configuration_filename)
    output_filename = os.path.join(args.data_folder, args.output_filename)

    return bulk_filename, configuration_filename, output_filename, args.B_vals

def main():

    bulk_filename, configuration_filename, output_filename, B_vals = get_commandline_args()

    bulk_data = pd.read_excel(bulk_filename)
    configuration_data = pd.read_excel(configuration_filename)

    energy_labels = ('Uniform', 'PP', 'PP (bulk)', 'PP (elastic)')

    data = {}
    bulk_B_0_idx = bulk_data['B'] == 0
    config_B_0_idx = configuration_data['B'] == 0
    no_cubic_data = (bulk_data['E'][bulk_B_0_idx].values[0], 
                     configuration_data['E'][config_B_0_idx].values[0],
                     configuration_data['bulk E'][config_B_0_idx].values[0],
                     configuration_data['elastic E'][config_B_0_idx].values[0])

    for i in range(len(bulk_data)):
        key = 'B = {}'.format(bulk_data['B'][i])
        data[key] = (bulk_data['E'][i] / no_cubic_data[0], 
                     configuration_data['E'][i] / no_cubic_data[1],
                     configuration_data['bulk E'][i] / no_cubic_data[2],
                     configuration_data['elastic E'][i] / no_cubic_data[3])

    x = np.arange(len(energy_labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained')
    
    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy (scaled by B = 0)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, energy_labels)
    ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0, 250)

    plt.show()

if __name__ == '__main__':
    main()
