import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def get_commandline_args():


    description = ("Plot Q1 for a few timesteps")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--spreadsheet_names',
                        dest='spreadsheet_names',
                        nargs='*',
                        help='list of names of csv files where amplitude data is')
    parser.add_argument('--plot_filename',
                        dest='plot_filename',
                        help='filename of amplitude plot')

    parser.add_argument('--time_key',
                        dest='time_key',
                        default='Time',
                        help='key in csv file corresponding to time data')
    parser.add_argument('--amplitude_key',
                        dest='amplitude_key',
                        default='Q1',
                        help='key in csv file corresponding to amplitude data')
    parser.add_argument('--position_key',
                        default='Points:0',
                        help='key in csv file corresponding to position data')

    args = parser.parse_args()

    spreadsheet_names = []
    for spreadsheet_name in args.spreadsheet_names:
        spreadsheet_names.append( os.path.join(args.data_folder, 
                                               spreadsheet_name) )

    output_folder = args.data_folder

    output_filename = os.path.join(output_folder, args.plot_filename)

    return (spreadsheet_names, output_filename, 
            args.time_key, args.amplitude_key, args.position_key)



def main():

    (spreadsheet_names, output_filename,
     time_key, amplitude_key, position_key) = get_commandline_args()

    x_ar = []
    Q1_ar = []
    t_ar = []
    for spreadsheet_name in spreadsheet_names:
        data = pd.read_csv(spreadsheet_name)
        x_ar.append(data[position_key].values)
        Q1_ar.append(data[amplitude_key].values)
        t_ar.append(data[time_key].values[0])

    fig, ax = plt.subplots()
    for x, Q1, t in zip(x_ar, Q1_ar, t_ar):
        ax.plot(x, Q1, label='t = {}'.format(t))

    ax.set_title(r'$Q_{12}$ vs. $x$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$Q_{12}$')

    ax.legend()
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.show()

if __name__ == '__main__':
    main()
