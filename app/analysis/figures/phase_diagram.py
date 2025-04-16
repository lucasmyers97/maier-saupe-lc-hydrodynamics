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

state_markers = {
        ('PR', 'PP'): 'o',
        ('TP', 'ER'): '^',
        ('TP', 'TER'): '^',
        ('TP', 'PP'): 's',
        ('PP', 'PP'): '.',
        ('ER', 'ER'): '.',
        ('PR', 'PR'): '.'
        }

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
        if transition == ('TP', 'PP'):
            R_ref = np.linspace(R[idx][0], 25.5, num=1000)

        if transition == ('PR', 'PP'):
            bspl = make_interp_spline(L2[idx], R[idx], k=k)
            L2_ref = np.linspace(L2[idx][0], L2[idx][-1], num=1000)
            plt.plot(bspl(L2_ref), L2_ref, c='black')
        else:
            plt.plot(R_ref, bspl(R_ref), c='black')

        plt.plot(R, L2, ls='', marker=state_markers[transition])


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

    plt.text(2, 8, 'PR')
    plt.plot([5, 1.2], [7.7, 3.5], c='black', lw=0.5)
    plt.text(30, 4, 'TP')
    plt.text(7, 1.3, 'PP')
    plt.text(80, 3, 'ER/TER')

    plt.tight_layout()

    plt.savefig(plot_filename)
    plt.show()

if __name__ == '__main__':
    main()
