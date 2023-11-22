import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit

plt.style.use('science')

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'lines.linewidth': 2})

def get_commandline_args():
    

    description = ('Plot fourier modes of core structure eigenvalues from '
                   '`calc_eigenvalue_fourier_modes` script')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--input_filenames',
                        dest='input_filenames',
                        nargs=2,
                        help='input hdf5 filenames containing cores structure')
    parser.add_argument('--output_filename',
                        dest='output_filename',
                        help='prefix name of plot filename (strings will be appended)')
    parser.add_argument('--modes',
                        dest='modes',
                        nargs=2,
                        default=[0, 1],
                        type=int,
                        help='which Fourier modes to plot')
    parser.add_argument('--data_keys',
                        dest='data_keys',
                        nargs=2,
                        help='key names of timestep in hdf5 file')
    parser.add_argument('--equilibrium_S',
                        dest='equilibrium_S',
                        type=float,
                        help='equilibrium S value at far-field')
    parser.add_argument('--color',
                        help='color of higher fourier mode')
    args = parser.parse_args()

    input_filenames = [os.path.join(args.data_folder, filename) 
                       for filename in args.input_filenames]
    output_filename = None
    if args.output_folder:
        output_filename = os.path.join(args.output_folder, args.output_filename)
    else:
        output_filename = os.path.join(args.data_folder, args.output_filename)

    return (input_filenames, output_filename, 
            args.modes, args.data_keys, args.equilibrium_S, args.color)


def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)



def plot_modes_with_asymptotics(Cn1, Cn2, r, 
                                Cn1_asymp, Cn1_fit, Cn2_asymp, Cn2_fit, r_asymp,
                                ylabel1, ylabel2, equilibrium_S, color):

    color_1 = 'black'
    # color_2 = 'tab:blue'
    color_2 = color

    inset_coords = [0.48, 0.27, 0.45, 0.55]

    sparse_idx = slice(0, -1, 25)
    fig_Cn, ax_Cn = plt.subplots(figsize=(5, 3))
    ax_Cn.plot(r[sparse_idx], Cn1[sparse_idx], color=color_1, linestyle='', marker='+')

    ax_Cn.set_xlabel(r'$r / \xi$')
    ax_Cn.set_ylabel(ylabel1, color=color_1)
    ax_Cn.tick_params(axis='y', labelcolor=color_1)

    ax2_Cn = ax_Cn.twinx()
    ax2_Cn.plot(r[sparse_idx], -Cn2[sparse_idx], color=color_2, linestyle='', marker='.')

    ax2_Cn.set_ylabel(ylabel2, color=color_2)
    ax2_Cn.tick_params(axis='y', labelcolor=color_2)
    ax2_Cn.spines['right'].set_color(color_2)
    ax2_Cn.tick_params(axis='y', which='both', color=color_2, labelcolor=color_2)
    y2_lims = ax2_Cn.get_ylim()
    fig_Cn.tight_layout()

    # make zeros line up
    align_yaxis(ax_Cn, ax2_Cn)

    x_lims = ax_Cn.get_xlim()
    x_lims2 = ax2_Cn.get_xlim()
    ax_Cn.hlines(2 * equilibrium_S, x_lims[0], x_lims[1], label=r'Far-field $S - P$', linestyle='--', zorder=-1)
    ax2_Cn.hlines(0, x_lims2[0], x_lims2[1], linestyle='--', zorder=-1)

    sparse_idx_1 = slice(0, -1, 50)
    sparse_idx_2 = slice(0, -1, 100)
    ax_Cn_asymp = ax2_Cn.inset_axes(inset_coords)
    ax_Cn_asymp.plot(r_asymp, Cn1_fit, color=color_1)
    ax_Cn_asymp.plot(r_asymp[sparse_idx_1], Cn1_asymp[sparse_idx_1], color=color_1, ls='', marker='+')

    # style inset 1
    ax_Cn_asymp.spines['left'].set_color(color_1)
    ax_Cn_asymp.ticklabel_format(scilimits=(-1, 6), axis='y', useMathText=True)

    ax2_Cn_asymp = ax_Cn_asymp.twinx()
    ax2_Cn_asymp.plot(r_asymp, Cn2_fit, color=color_2)
    ax2_Cn_asymp.plot(r_asymp[sparse_idx_2], Cn2_asymp[sparse_idx_2], color=color_2, ls='', marker='.')

    # style inset 2
    ax2_Cn_asymp.ticklabel_format(scilimits=(-1, 6), axis='y', useMathText=True)
    ax2_Cn_asymp.spines['right'].set_color(color_2)
    ax2_Cn_asymp.tick_params(axis='y', which='both', color=color_2, labelcolor=color_2)
    # ax2_Cn_asymp.yaxis.label.set_color(color_2)

    # make inset axes lign up
    y2_max = np.max(Cn2_fit)
    ax2_Cn_asymp.set_ylim(top=y2_max)
    y1_lims = ax_Cn.get_ylim()
    y2_lims = ax2_Cn.get_ylim()
    small2_lims = np.array(ax2_Cn_asymp.get_ylim())
    small_lims = small2_lims * (y1_lims[1] - y1_lims[0]) / (y2_lims[1] - y2_lims[0])
    ax_Cn_asymp.set_ylim(small_lims)

    ax2_Cn.indicate_inset_zoom(ax2_Cn_asymp, edgecolor='black')

    return fig_Cn



def get_fourier_modes(filename, data_key):

    file = h5py.File(filename)
    data = file[data_key]

    r0 = data.attrs['r0']
    rf = data.attrs['rf']
    n_r = data.attrs['n_r']
    n_modes = data.attrs['n_modes']
    r = np.linspace(r0, rf, num=n_r)

    An_Gamma = np.array(data[:]).reshape((n_r, n_modes))

    file.close()

    return An_Gamma, r


def main():

    input_filenames, output_filename, modes, data_keys, equilibrium_S, color = get_commandline_args()

    An_gamma_far, r = get_fourier_modes(input_filenames[0], data_keys[0])
    An_gamma_close, r_close = get_fourier_modes(input_filenames[1], data_keys[1])

    fit_curve = lambda r, a, n, b: a * r**n + b
    A0_gamma_fit, _ = curve_fit(fit_curve, r_close, An_gamma_close[:, modes[0]], p0=(1, 1, 0))
    A1_gamma_fit, _ = curve_fit(fit_curve, r_close, -An_gamma_close[:, modes[1]], p0=(1, 2, 0))

    print(A0_gamma_fit)
    print(A1_gamma_fit)

    fig = plot_modes_with_asymptotics(An_gamma_far[:, modes[0]], 
                                      An_gamma_far[:, modes[1]], 
                                      r, 
                                      An_gamma_close[:, modes[0]], 
                                      fit_curve(r_close, *A0_gamma_fit), 
                                      -An_gamma_close[:, modes[1]], 
                                      fit_curve(r_close, *A1_gamma_fit), 
                                      r_close,
                                      r'$\Gamma_{}$'.format(modes[0]),
                                      r'$-\Gamma_{}$'.format(modes[1]),
                                      equilibrium_S,
                                      color)

    fig.savefig(output_filename)

    plt.show()


if __name__ == "__main__":

    main()
