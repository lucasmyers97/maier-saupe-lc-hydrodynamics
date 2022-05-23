import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
import h5py
import os

mpl.rcParams['figure.dpi'] = 300

plt.style.use('science')

def read_data(filename):

    f = h5py.File(filename, 'r')
    S_vals = f['S_val'][:]
    fe_3_3 = f['free_energy_3.300000'][:]
    fe_3_4049 = f['free_energy_3.404900'][:]
    fe_3_5 = f['free_energy_3.500000'][:]

    return S_vals, fe_3_3, fe_3_4049, fe_3_5

def calc_LdG_free_energy(S_val, A, B=-1, C=1):

    return ((1/3) * A * S_val**2
            + (2/27) * B * S_val**3
            + (1/9) * C * S_val**4) / ((1/27) * B**2 * C)

if __name__ == "__main__":

    description = "Plots Landau-de Gennes and Maier-Saupe bulk free energies"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--maier_saupe_folder', dest='maier_saupe_folder',
                        help='folder where maier_saupe data is stored')
    parser.add_argument('--maier_saupe_filename', dest='maier_saupe_filename',
                        help='name of data file with maier_saupe data')
    parser.add_argument('--plot_folder', dest='plot_folder',
                        help='folder where plots will be saved')
    args = parser.parse_args()

    filename = os.path.join(args.maier_saupe_folder, args.maier_saupe_filename)
    LdG_filename = os.path.join(args.plot_folder, "LdG_free_energy.png")
    MS_filename = os.path.join(args.plot_folder, "MS_free_energy.png")
    S, fe_1, fe_2, fe_3 = read_data(filename)

    fig, ax = plt.subplots()

    ax.plot(S, fe_1, label=r'$\alpha/(N k_B T) = 3.3$')
    ax.plot(S, fe_2, label=r'$\alpha/(N k_B T) = 3.4049$')
    ax.plot(S, fe_3, label=r'$\alpha/(N k_B T) = 3.5$')

    ax.set_ylim(-.02, .02)
    ax.set_xlim(-0.1, 0.7)
    ax.legend()

    ax.set_xlabel(r'$S$')
    ax.set_ylabel(r'$f_b$')

    fig.tight_layout()
    fig.savefig(MS_filename)

    S = np.linspace(-.1, .8, num=1000)

    B = -2
    C = -(B/0.95)
    A = (1/27) * B**2 / C

    f1 = calc_LdG_free_energy(S, A * 2, B, C)
    f2 = calc_LdG_free_energy(S, A * 1, B, C)
    f3 = calc_LdG_free_energy(S, A * 0.1, B, C)

    fig, ax = plt.subplots()

    ax.plot(S, f1, label=r'$A = 2A_{IN}$')
    ax.plot(S, f2, label=r'$A = A_{IN}$')
    ax.plot(S, f3, label=r'$A = A_{IN}/10$')
    ax.set_ylim(-.02, .02)
    ax.set_xlim(-0.1, 0.7)

    ax.set_xlabel(r'$S$')
    ax.set_ylabel(r'$f_b$')
    ax.legend()

    fig.tight_layout()

    fig.savefig(LdG_filename)

    plt.show()
