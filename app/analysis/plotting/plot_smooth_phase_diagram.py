import argparse
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from scipy.interpolate import make_interp_spline

mpl.rcParams['figure.dpi'] = 300
plt.style.use('science')

state_colors = {'ER': 'tab:orange',
                'TER': 'tab:red',
                'PP': 'tab:green',
                'TP': 'tab:blue',
                'PR': 'black'}

def get_commandline_args():

    desc = "Given table of phases with R and L2, plot a phase diagram"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_file',
                        help='name of ods folder holding phase transition points')
    parser.add_argument('--plot_filename',
                        help='name of plot (will go in the same folder as data)')

    args = parser.parse_args()

    folder = os.path.dirname(args.data_file)
    plot_filename = os.path.join(folder, args.plot_filename)

    return args.data_file, plot_filename


def main():

    data_file, plot_filename = get_commandline_args()
    data = pd.read_excel(data_file)
    
    transitions = data['transition'].apply( lambda x: tuple(x.split('-')) )
    unique_transitions = pd.unique(transitions)

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)


    axin1 = ax.inset_axes([.1, .4, 0.3, 0.5],
                          xlim=(0, 5),
                          ylim=(0, 2.5))
    axin1.set_xticks([0, 2.5])
    axin1.set_yticks([0, 1.0])
    axin1.tick_params(which='both',
                    bottom=False, 
                    top=False,
                    left=False,
                    right=False)
    for transition in unique_transitions:
        print(transition)
        is_transition = transitions.apply(lambda x: x == transition)

        R = data[is_transition]['R'].values
        L2 = data[is_transition]['L2'].values

        idx = np.argsort(R)

        k = 1 if len(R) < 2 else len(R) - 1
        if k == 4:
            k = 3

        bspl = make_interp_spline(R[idx], L2[idx], k=k)
        R_ref = np.linspace(R[idx][0], R[idx][-1], num=1000)

        # plt.plot(R[idx], L2[idx])
        plt.plot(R_ref, bspl(R_ref), c='black')
        axin1.plot(R_ref, bspl(R_ref), c='black')


    ax.indicate_inset_zoom(axin1, edgecolor="black")
    plt.tick_params(which='both',
                    bottom=False, 
                    top=False,
                    left=False,
                    right=False)
    plt.xticks([0.0, 10, 50, 100])
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel(r'$R$ (Capillary radius)')
    plt.ylabel(r'$L_2$ (Twist elastic anisotropy)')

    plt.text(65, 6, 'TP')
    plt.text(80, 3, 'ER/TER')

    axin1.text(0.75, 1, 'PR')
    axin1.text(3, 1, 'TP')

    # plt.annotate('\}', (2.5, -2), rotation=-90)

    plt.tight_layout()

    plt.savefig(plot_filename)
    plt.show()

if __name__ == '__main__':
    main()
