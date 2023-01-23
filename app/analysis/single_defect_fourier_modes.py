import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit

plt.style.use('science')

mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():
    

    description = ("Get fourier modes of core structure eigenvalues")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        default=None,
                        help='folder that output file will be written to')
    parser.add_argument('--input_filename',
                        dest='input_filename',
                        help='input hdf5 filename containing cores structure')
    parser.add_argument('--plot_prefix',
                        dest='plot_prefix',
                        help='prefix name of plot filename (strings will be appended)')
    parser.add_argument('--n_modes',
                        dest='n_modes',
                        default=4,
                        type=int,
                        help='number of Fourier modes to plot')
    parser.add_argument('--timestep_key',
                        dest='timestep_key',
                        help='key name of timestep in hdf5 file')
    parser.add_argument('--r0',
                        dest='r0',
                        type=float,
                        help='cutoff for polynomial behavior of Fourier modes')
    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    plot_prefix = None
    if args.output_folder:
        plot_prefix = os.path.join(args.output_folder, args.plot_prefix)
    else:
        plot_prefix = os.path.join(args.data_folder, args.plot_prefix)

    return input_filename, plot_prefix, args.n_modes, args.timestep_key, args.r0



def get_fourier_coeffs(X, n_modes):

    An_X = np.zeros((X.shape[1], n_modes))
    Bn_X = np.zeros((X.shape[1], n_modes))

    for i in range(X.shape[1]):

        # X at each fixed R
        x = X[:, i]

        FT_x = np.fft.rfft(x)
        N = x.shape[0]

        # get sin and cos coefficients from DFT
        An_x = FT_x.real / N
        Bn_x = FT_x.imag / N

        # cos modes
        for j in range(n_modes):
            An_X[i, j] = An_x[j]

        # sin modes
        for j in range(1, n_modes):
            Bn_X[i, j] = Bn_x[j]

    return An_X, Bn_X



def plot_fourier_coeffs(Cn, r, n_modes, title, include_first=True):

    if include_first:
        start = 0
    else:
        start = 1

    # make plots
    fig_Cn, ax_Cn = plt.subplots()
    for j in range(start, n_modes):
        ax_Cn.plot(r, Cn[:, j], label="k = {}".format(j))

    ax_Cn.set_xlabel("radius")
    ax_Cn.set_ylabel("Fourier amplitude")
    ax_Cn.set_title(title)
    ax_Cn.legend()

    fig_Cn.tight_layout()
    
    return fig_Cn, ax_Cn



def plot_two_fourier_coeffs(Cn, r, ylabel1, ylabel2):

    color = 'tab:red'
    fig_Cn, ax_Cn = plt.subplots()
    ax_Cn.plot(r, Cn[:, 0], color=color)
    ax_Cn.set_xlabel(r'$r / \xi$')
    ax_Cn.set_ylabel(ylabel1, color=color)
    ax_Cn.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2_Cn = ax_Cn.twinx()
    ax2_Cn.plot(r, -Cn[:, 1], color=color)
    ax2_Cn.set_ylabel(ylabel2, color=color)
    ax2_Cn.tick_params(axis='y', labelcolor=color)
    fig_Cn.tight_layout()

    return fig_Cn, ax_Cn, ax2_Cn



def main():

    input_filename, plot_prefix, n_modes, timestep_key, r0 = get_commandline_args()

    file = h5py.File(input_filename)
    grp = file[timestep_key]

    r = np.array(grp['r'][:])
    theta = np.array(grp['theta'][:])
    R, Theta = np.meshgrid(r, theta)

    point_dims = np.array(grp['point_dims'][:])
    q1 = np.array(grp['q1'][:])
    q2 = np.array(grp['q2'][:])
    S = 1.5 * q1
    P = 0.5 * q1 + q2

    q1 = q1.reshape(point_dims)
    q2 = q2.reshape(point_dims)
    S = S.reshape(point_dims)
    P = P.reshape(point_dims)

    An_q1, Bn_q1 = get_fourier_coeffs(q1, n_modes)
    An_q2, Bn_q2 = get_fourier_coeffs(q2, n_modes)
    An_Gamma, _ = get_fourier_coeffs(S - P, n_modes)

    # plot first two modes
    (fig_An_q1, 
     ax1_An_q1, 
     ax2_An_q1) = plot_two_fourier_coeffs(An_q1, 
                                          R[0, :],
                                          ylabel1=r'$q_1^{(0)} (r)$',
                                          ylabel2=r'$-q_1^{(1)} (r)$')
    (fig_An_q2, 
     ax1_An_q2, 
     ax2_An_q2) = plot_two_fourier_coeffs(An_q2, 
                                          R[0, :],
                                          ylabel1=r'$q_2^{(0)} (r)$',
                                          ylabel2=r'$-q_2^{(1)} (r)$')
    (fig_An_Gamma, 
     ax1_An_Gamma, 
     ax2_An_Gamma) = plot_two_fourier_coeffs(An_Gamma, 
                                             R[0, :],
                                             ylabel1=r'$\Gamma_0 (r)$',
                                             ylabel2=r'$-\Gamma_1 (r)$')

    fig_An_q1.savefig(plot_prefix + "An_q1.png")
    fig_An_q2.savefig(plot_prefix + "An_q2.png")
    fig_An_Gamma.savefig(plot_prefix + "An_Gamma.png")

    plt.show()

    # Look at small r regime
    r_small_idx = R[0, :] < r0
    r_small = R[0, r_small_idx]

    A0_q1_small = An_q1[r_small_idx, 0]
    A1_q1_small = An_q1[r_small_idx, 1]
    A0_q1_small -= np.min(A0_q1_small) - 1e-8
    A1_q1_small -= np.max(A1_q1_small) + 1e-8

    # plt.figure()
    # plt.plot(r_small, A0_q1_small)
    # plt.figure()
    # plt.plot(r_small, A1_q1_small)

    A0_fig_log, A0_ax_log = plt.subplots()
    A0_ax_log.plot(np.log(r_small), np.log(A0_q1_small))
    A0_ax_log.set_xlabel(r'$\log(r)$')
    A0_ax_log.set_ylabel(r'$\log(A_{q_1}^{(0)})$')
    A0_ax_log.set_title(r'$\log(A_{q_1}^{(0)})$ for small $r$')
    A0_fig_log.tight_layout()

    A1_fig_log, A1_ax_log = plt.subplots()
    A1_ax_log.plot(np.log(r_small), np.log(-A1_q1_small))
    A1_ax_log.set_xlabel(r'$\log(r)$')
    A1_ax_log.set_ylabel(r'$\log(A_{q_1}^{(1)})$')
    A1_ax_log.set_title(r'$\log(A_{q_1}^{(1)})$ for small $r$')
    A1_fig_log.tight_layout()

    A0_fig, A0_ax = plt.subplots()
    A0_ax.plot(np.log(r_small), np.log(A0_q1_small))
    A0_ax.set_xlabel(r'$r$')
    A0_ax.set_ylabel(r'$A_{q_1}^{(0)}$')
    A0_ax.set_title(r'$A_{q_1}^{(0)}$ for small $r$')
    A0_fig.tight_layout()

    A1_fig_2, A1_ax_2 = plt.subplots()
    A1_ax_2.plot(r_small, np.sqrt(-A1_q1_small))
    A1_ax_2.set_xlabel(r'$r$')
    A1_ax_2.set_ylabel(r'$\sqrt{A_{q_1}^{(1)}}$')
    A1_ax_2.set_title(r'$\sqrt{A_{q_1}^{(1)}}$ for small $r$')
    A1_fig_2.tight_layout()

    A1_fig_3, A1_ax_3 = plt.subplots()
    A1_ax_3.plot(r_small, (-A1_q1_small)**(1 / 3))
    A1_ax_3.set_xlabel(r'$r$')
    A1_ax_3.set_ylabel(r'$(A_{q_1}^{(1)})^{1/3}$')
    A1_ax_3.set_title(r'$(A_{q_1}^{(1)})^{1/3}$ for small $r$')
    A1_fig_3.tight_layout()

    plt.show()

    # # show log of each to check power
    # An_q1[:, 0] -= np.min(An_q1[:, 0]) - 1e-8
    # An_q1[:, 1] -= np.max(An_q1[:, 1]) + 1e-8
    # An_q1[:, 0] = np.log(An_q1[:, 0])
    # An_q1[:, 1] = np.log(-An_q1[:, 1])

    # (fig_An_q1, 
    #  ax1_An_q1, 
    #  ax2_An_q1) = plot_two_fourier_coeffs(An_q1, 
    #                                       R[0, :],
    #                                       ylabel1=r'$q_1^{(0)} (r)$',
    #                                       ylabel2=r'$-q_1^{(1)} (r)$')
    # plt.show()



if __name__ == "__main__":

    main()
