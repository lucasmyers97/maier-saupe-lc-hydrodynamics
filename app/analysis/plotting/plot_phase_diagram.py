import argparse
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

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
                        help='name of csv folder holding phase transition points')

    args = parser.parse_args()

    return args.data_file

def main():
    
    data_file = get_commandline_args()
    data = pd.read_csv(data_file)

    R = data['R']
    L2 = data['L2']

    states = data['transition'].apply(lambda x: x.split('-'))
    left_states = states.apply(lambda x: x[0])
    right_states = states.apply(lambda x: x[-1])

    left_colors = [state_colors[state] for state in left_states]
    right_colors = [state_colors[state] for state in right_states]

    print(left_colors)
    print(right_colors)

    fig, ax = plt.subplots()

    for i in range(len(left_colors)):
        ax.plot(R[i], L2[i], ls='', c=left_colors[i], 
                 marker=mpl.markers.MarkerStyle("o", fillstyle="left"), markeredgecolor='none')

    for i in range(len(right_colors)):
        ax.plot(R[i], L2[i], ls='', c=right_colors[i], marker=mpl.markers.MarkerStyle("o", fillstyle="right"), markeredgecolor='none')
    # plt.plot(R, L2)

    legend_elements = []
    for key, value in state_colors.items():
        legend_elements.append(mpl.lines.Line2D([0], [0], marker='o', color='w', label=key,
                               markerfacecolor=value))

    ax.legend(handles=legend_elements, frameon=True)
    ax.set_xlabel('R')
    ax.set_ylabel('L2')


    plt.show()

if __name__ == '__main__':
    main()
