import argparse
import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch

plt.style.use('science')

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': 15})
mpl.rcParams.update({'lines.linewidth': 2})

def get_commandline_args():

    description = 'Plots colormap of theta_c for the perturbative director'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data_folder',
                        help='folder where h5 data lives')
    parser.add_argument('--outer_filename',
                        help='h5 file with outer structure')
    parser.add_argument('--inner_filename',
                        help='h5 file with inner structure')
    parser.add_argument('--data_key',
                        help='key in h5 files for data')

    parser.add_argument('--plot_name',
                        help='png file which plot will be written to')

    args = parser.parse_args()

    outer_filename = os.path.join(args.data_folder, args.outer_filename)
    inner_filename = os.path.join(args.data_folder, args.inner_filename)
    output_filename = os.path.join(args.data_folder, args.plot_name)

    return outer_filename, inner_filename, args.data_key, output_filename



def get_data(filename, data_key):

    file = h5py.File(filename)
    data = file[data_key]

    r0 = data.attrs['r_0']
    rf = data.attrs['r_f']
    n_r = data.attrs['n_r']
    n_theta = data.attrs['n_theta']
    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)
    R, Theta = np.meshgrid(r, theta)

    theta_c = np.array(data[:])
    theta_c = theta_c.reshape((n_r, n_theta)).transpose()

    return theta_c, R, Theta



def main():

    outer_filename, inner_filename, data_key, output_filename = get_commandline_args()

    theta_c, R, Theta = get_data(outer_filename, data_key)
    theta_c_small, R_small, Theta_small = get_data(inner_filename, data_key)

    cmap = plt.colormaps['plasma']
    cmap_small = plt.colormaps['plasma']

    # make plots
    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(5, 2), subplot_kw={'projection': 'polar'})
    pcm = ax1.pcolormesh(Theta, R, theta_c, shading='nearest', cmap=cmap, vmin=-0.01, vmax=0.01)
    fig.colorbar(pcm, ax=ax1)

    pcm_small = ax2.pcolormesh(Theta_small, R_small, theta_c_small, shading='nearest', cmap=cmap_small)
    fig.colorbar(pcm_small, ax=ax2)

    # style plots
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)

    xy1 = (np.pi / 2, np.max(R_small))
    con1 = ConnectionPatch(xyA=xy1, coordsA=ax1.transData,
                           xyB=xy1, coordsB=ax2.transData)
    xy2 = (-np.pi / 2, np.max(R_small))
    con2 = ConnectionPatch(xyA=xy2, coordsA=ax1.transData,
                           xyB=xy2, coordsB=ax2.transData)
    con1.set(lw=0.5)
    con2.set(lw=0.5)
    fig.add_artist(con1)
    fig.add_artist(con2)

    circle = plt.Circle((0, 0), np.max(R_small), color='black', fill=False, zorder=100, lw=0.5, transform=ax1.transProjectionAffine + ax1.transAxes)
    ax1.add_patch(circle)

    fig.tight_layout()
    fig.savefig(output_filename)

    plt.show()



if __name__ == '__main__':
    main()
