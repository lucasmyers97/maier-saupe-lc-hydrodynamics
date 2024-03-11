import argparse
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

markers = ['o', 'v', 's']
colors = ['tab:blue', 'tab:orange', 'tab:red']

def get_commandline_args():

    desc = 'Takes in spreadsheet of twist wavenumbers (alpha) vs equilibrium energy, plots them on one graph'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_folder', help='folder where data lives')
    parser.add_argument('--filename', help='Name of excel type file with data')
    parser.add_argument('--plot_filename', help='Filename of output plot')
    parser.add_argument('--L2_values',
                        nargs='+',
                        type=float,
                        help='L2 values to plot')
    parser.add_argument('--R',
                        type=float,
                        help='Radius of capillary')

    args = parser.parse_args()

    filename = os.path.join(args.data_folder, args.filename)
    plot_filename = os.path.join(args.data_folder, args.plot_filename)

    return filename, plot_filename, args.L2_values, args.R



def main():

    fudge = 0.25

    filename, plot_filename, L2_vals, R = get_commandline_args()
    
    data = pd.read_excel(filename)

    # get offset for plots
    min_L2 = min(L2_vals)
    min_data = data[data['L2'] == min_L2]
    energy_range = min_data['energy'].max() - min_data['energy'].min()
    energy_range_exponent = np.floor(np.log10(np.abs(energy_range))).astype(int)
    offset = np.round(min_data['energy'].min() - energy_range, decimals=-energy_range_exponent)

    max_energy_range = 0
    for L2 in L2_vals:
        range_data = data[data['L2'] == L2]
        energy_range = range_data['energy'].max() - range_data['energy'].min()
        max_energy_range = max(max_energy_range, energy_range)

    max_energy_range *= (1 + fudge)
    print(max_energy_range)

    fig, ax = plt.subplots(nrows=len(L2_vals), sharex=True, layout='constrained')
    fig.subplots_adjust(hspace=0.05)
    for L2, i in zip(reversed(L2_vals), range(ax.shape[0])):
        plot_data = data[data['L2'] == L2]

        polynomial_degree = 2
        fit = np.polynomial.polynomial.Polynomial.fit(plot_data['alpha'], 
                                                      plot_data['energy'],
                                                      polynomial_degree)

        c, b, a = fit.convert().coef
        min_x = -b / (2 * a) 
        min_y = c - b**2/(4*a)
        print(min_x)
        print('Curvature is: {}'.format(a))

        x_fit, y_fit = fit.linspace(1000)

        ax[i].plot(x_fit, y_fit)
        ax[i].plot(plot_data['alpha'], plot_data['energy'], ls='', marker=markers[i], color=colors[i], label=r'$L_2 = {}$'.format(L2))
        if i == len(L2_vals) - 1:
            ax[i].plot(min_x, min_y, marker='x', ls='', label=r'$\alpha_0$')
        else:
            ax[i].plot(min_x, min_y, marker='x')
        ax[i].ticklabel_format(useMathText=True, axis='y')
        ax[i].ticklabel_format(useOffset=offset, axis='y')

        if i != 0:
            plt.setp(ax[i].yaxis.get_offset_text(), visible=False)

        ax[i].spines.bottom.set_visible(False)
        ax[i].spines.top.set_visible(False)
        ax[i].xaxis.set_ticks_position('none')

        actual_range = plot_data['energy'].max() - plot_data['energy'].min()
        room = max_energy_range - actual_range
        ax[i].set_ylim(bottom=plot_data['energy'].min() - room*0.5,
                       top=plot_data['energy'].max() + room*0.5)

    ax[0].spines.top.set_visible(True)
    # ax[0].xaxis.set_ticks_position('top')

    ax[-1].spines.bottom.set_visible(True)
    ax[-1].xaxis.set_ticks_position('bottom')

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

    ax[-1].set_xticks([np.pi / (8 * R), 2 * np.pi / (8 * R), 3 * np.pi / (8 * R)], 
                      [r'$\pi / 8 R$', r'$\pi / 4 R$', r'$3 \pi / 8 R$'])

    
    fig.supxlabel(r'$\alpha$')
    fig.supylabel(r'$F / n k_B T$')
    fig.legend(loc='upper left', frameon=True, bbox_to_anchor=(0.18, 0.92))

    plt.savefig(plot_filename)

    plt.show()


if __name__ == '__main__':

    main()
