import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit

from ..utilities import nematics as nu

plt.style.use('science')

mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():
    

    description = ('Get fourier modes of core structure eigenvalues from '
                   '`get_points_around_defect` program')
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
    parser.add_argument('--data_key',
                        dest='data_key',
                        help='key name of timestep in hdf5 file')
    parser.add_argument('--r_cutoff',
                        dest='r_cutoff',
                        type=float,
                        help='cutoff for polynomial behavior of Fourier modes')
    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    plot_prefix = None
    if args.output_folder:
        plot_prefix = os.path.join(args.output_folder, args.plot_prefix)
    else:
        plot_prefix = os.path.join(args.data_folder, args.plot_prefix)

    return (input_filename, plot_prefix, 
            args.n_modes, args.data_key, args.r_cutoff)



def get_fourier_coeffs(X, n_modes):

    An_X = np.zeros((X.shape[0], n_modes))
    Bn_X = np.zeros((X.shape[0], n_modes))

    for i in range(X.shape[0]):

        # X at each fixed R
        x = X[i, :]

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



def fit_curve(r, a, n):

    return a * r**n



def main():

    (input_filename, plot_prefix, n_modes, 
     data_key, r_cutoff) = get_commandline_args()

    file = h5py.File(input_filename)
    data = file[data_key]

    r0 = data.attrs['r0']
    rf = data.attrs['rf']
    n_r = data.attrs['n_r']
    n_theta = data.attrs['n_theta']
    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)
    R, Theta = np.meshgrid(r, theta)

    Q_vec = np.array(data[:])
    Q_mat = nu.Q_vec_to_mat(Q_vec)
    q1, q2, _, _ = nu.eigensystem_from_Q(Q_mat)

    S = 1.5 * q1
    P = 0.5 * q1 + q2

    q1 = q1.reshape((n_r, n_theta))
    q2 = q2.reshape((n_r, n_theta))
    S = S.reshape((n_r, n_theta))
    P = P.reshape((n_r, n_theta))

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

    # # Look at small r regime
    # r_small_idx = R[0, :] < r_cutoff
    # r_small = R[0, r_small_idx]

    # # largest eigenvalue
    # A0_q1_small = An_q1[r_small_idx, 0]
    # A1_q1_small = An_q1[r_small_idx, 1]
    # A0_q1_small -= np.min(A0_q1_small) - 1e-8
    # A1_q1_small -= np.max(A1_q1_small) + 1e-8

    # A0_q1_fit, _ = curve_fit(fit_curve, r_small, A0_q1_small, p0=(1, 1))
    # A1_q1_fit, _ = curve_fit(fit_curve, r_small, -A1_q1_small, p0=(1, 2))
    # print(A0_q1_fit)
    # print(A1_q1_fit)

    # A0_fig, A0_ax = plt.subplots()
    # A0_ax.plot(r_small, A0_q1_small, label=r'$A_{q_1}^{(0)}$')
    # A0_ax.plot(r_small, fit_curve(r_small, A0_q1_fit[0], A0_q1_fit[1]),
    #            label=r'$a_0 r^n$, $a_0 = {:.2f}$, $n = {:.2f}$'.format(A0_q1_fit[0], A0_q1_fit[1]))
    # A0_ax.set_xlabel(r'$r$')
    # A0_ax.set_ylabel(r'$A_{q_1}^{(0)}$')
    # A0_ax.set_title(r'$A_{q_1}^{(0)}$ for small $r$')
    # A0_fig.tight_layout()
    # A0_fig.legend(loc='lower right')

    # A1_fig, A1_ax = plt.subplots()
    # A1_ax.plot(r_small, -A1_q1_small, label=r'$-A_{q_1}^{(1)}$')
    # A1_ax.plot(r_small, fit_curve(r_small, A1_q1_fit[0], A1_q1_fit[1]),
    #            label=r'$a_0 r^n$, $a_0 = {:.2f}$, $n = {:.2f}$'.format(A1_q1_fit[0], A1_q1_fit[1]))
    # A1_ax.set_xlabel(r'$r$')
    # A1_ax.set_ylabel(r'$A_{q_1}^{(1)}$')
    # A1_ax.set_title(r'$A_{q_1}^{(1)}$ for small $r$')
    # A1_fig.tight_layout()
    # A1_fig.legend(loc='lower right')

    # # second eigenvalue
    # A0_q2_small = An_q2[r_small_idx, 0]
    # A1_q2_small = An_q2[r_small_idx, 1]
    # A0_q2_small -= np.min(A0_q2_small) - 1e-8
    # A1_q2_small -= np.max(A1_q2_small) + 1e-8

    # A0_q2_fit, _ = curve_fit(fit_curve, r_small, A0_q2_small, p0=(-0.18, 1))
    # A1_q2_fit, _ = curve_fit(fit_curve, r_small, -A1_q2_small, p0=(-1, 2))
    # print(A0_q2_fit)
    # print(A1_q2_fit)

    # A0_fig, A0_ax = plt.subplots()
    # A0_ax.plot(r_small, A0_q2_small, label=r'$A_{q_2}^{(0)}$')
    # A0_ax.plot(r_small, fit_curve(r_small, A0_q2_fit[0], A0_q2_fit[1]),
    #            label=r'$a_0 r^n$, $a_0 = {:.2e}$, $n = {:.2e}$'.format(A0_q2_fit[0], A0_q2_fit[1]))
    # A0_ax.set_xlabel(r'$r$')
    # A0_ax.set_ylabel(r'$A_{q_2}^{(0)}$')
    # A0_ax.set_title(r'$A_{q_2}^{(0)}$ for small $r$')
    # A0_fig.tight_layout()
    # A0_fig.legend(loc='lower right')

    # A1_fig, A1_ax = plt.subplots()
    # A1_ax.plot(r_small, -A1_q2_small, label=r'$-A_{q_2}^{(1)}$')
    # A1_ax.plot(r_small, fit_curve(r_small, A1_q2_fit[0], A1_q2_fit[1]),
    #            label=r'$a_0 r^n$, $a_0 = {:.2e}$, $n = {:.2e}$'.format(A1_q2_fit[0], A1_q2_fit[1]))
    # A1_ax.set_xlabel(r'$r$')
    # A1_ax.set_ylabel(r'$A_{q_2}^{(1)}$')
    # A1_ax.set_title(r'$A_{q_2}^{(1)}$ for small $r$')
    # A1_fig.tight_layout()
    # A1_fig.legend(loc='lower right')

    # plt.show()

    # A1_fig_log, A1_ax_log = plt.subplots()
    # A1_ax_log.plot(np.log(r_small), np.log(-A1_q1_small))
    # A1_ax_log.set_xlabel(r'$\log(r)$')
    # A1_ax_log.set_ylabel(r'$\log(A_{q_1}^{(1)})$')
    # A1_ax_log.set_title(r'$\log(A_{q_1}^{(1)})$ for small $r$')
    # A1_fig_log.tight_layout()

    # A0_fig_log, A0_ax_log = plt.subplots()
    # A0_ax_log.plot(np.log(r_small), np.log(A0_q2_small))
    # A0_ax_log.set_xlabel(r'$\log(r)$')
    # A0_ax_log.set_ylabel(r'$\log(A_{q_2}^{(0)})$')
    # A0_ax_log.set_title(r'$\log(A_{q_2}^{(0)})$ for small $r$')
    # A0_fig_log.tight_layout()

    # A1_fig_log, A1_ax_log = plt.subplots()
    # A1_ax_log.plot(np.log(r_small), np.log(-A1_q2_small))
    # A1_ax_log.set_xlabel(r'$\log(r)$')
    # A1_ax_log.set_ylabel(r'$\log(A_{q_2}^{(1)})$')
    # A1_ax_log.set_title(r'$\log(A_{q_2}^{(1)})$ for small $r$')
    # A1_fig_log.tight_layout()

    # plt.show()


if __name__ == "__main__":

    main()
