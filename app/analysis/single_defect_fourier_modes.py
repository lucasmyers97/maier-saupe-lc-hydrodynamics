import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    plot_prefix = None
    if args.output_folder:
        plot_prefix = os.path.join(args.output_folder, args.plot_prefix)
    else:
        plot_prefix = os.path.join(args.data_folder, args.plot_prefix)

    return input_filename, plot_prefix, args.n_modes, args.timestep_key



def main():

    input_filename, plot_prefix, n_modes, timestep_key = get_commandline_args()

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

    An_q1 = np.zeros((R.shape[1], n_modes))
    Bn_q1 = np.zeros((R.shape[1], n_modes))
    An_q2 = np.zeros((R.shape[1], n_modes))
    Bn_q2 = np.zeros((R.shape[1], n_modes))
    An_Gamma = np.zeros((R.shape[1], n_modes))

    for i in range(R.shape[1]):

        # values for fixed r, varied theta
        s = S[:, i]
        p = P[:, i]
        gamma = s - p
        q1_r = q1[:, i]
        q2_r = q2[:, i]

        FT_q1_r = np.fft.rfft(q1_r)
        FT_q2_r = np.fft.rfft(q2_r)
        FT_gamma = np.fft.rfft(gamma)

        # change discrete fourier into sin/cos numbered fourier modes
        N = q1_r.shape[0]
        An_q1_r = FT_q1_r.real / N
        Bn_q1_r = FT_q1_r.imag / N
        An_q2_r = FT_q2_r.real / N
        Bn_q2_r = FT_q2_r.imag / N
        An_gamma = FT_gamma.real / N

        # cos modes
        for j in range(n_modes):
            An_q1[i, j] = An_q1_r[j]
            An_q2[i, j] = An_q2_r[j]
            An_Gamma[i, j] = An_gamma[j]

        # sin modes
        for j in range(1, n_modes):
            Bn_q1[i, j] = Bn_q1_r[j]
            Bn_q2[i, j] = Bn_q2_r[j]

    # make plots
    fig_An_q1, ax_An_q1 = plt.subplots()
    fig_Bn_q1, ax_Bn_q1 = plt.subplots()
    fig_An_q2, ax_An_q2 = plt.subplots()
    fig_Bn_q2, ax_Bn_q2 = plt.subplots()
    for j in range(n_modes):
        ax_An_q1.plot(R[0, :], An_q1[:, j], label="k = {}".format(j))
        ax_An_q2.plot(R[0, :], An_q2[:, j], label="k = {}".format(j))

    for j in range(1, n_modes):
        ax_Bn_q1.plot(R[0, :], Bn_q1[:, j], label="k = {}".format(j))
        ax_Bn_q2.plot(R[0, :], Bn_q2[:, j], label="k = {}".format(j))

    ax_An_q1.set_xlabel("radius")
    ax_An_q1.set_ylabel("Fourier amplitude")
    ax_An_q1.set_title(r'$\cos$ Fourier coefficients for $q_1$')
    ax_An_q1.legend()

    ax_Bn_q1.set_xlabel("radius")
    ax_Bn_q1.set_ylabel("Fourier amplitude")
    ax_Bn_q1.set_title(r'$\sin$ Fourier coefficients for $q_1$')
    ax_Bn_q1.legend()

    ax_An_q2.set_xlabel("radius")
    ax_An_q2.set_ylabel("Fourier amplitude")
    ax_An_q2.set_title(r'$\cos$ Fourier coefficients for $q_2$')
    ax_An_q2.legend()

    ax_Bn_q2.set_xlabel("radius")
    ax_Bn_q2.set_ylabel("Fourier amplitude")
    ax_Bn_q2.set_title(r'$\sin$ Fourier coefficients for $q_2$')
    ax_Bn_q2.legend()

    fig_An_q1.tight_layout()
    fig_Bn_q1.tight_layout()
    fig_An_q2.tight_layout()
    fig_Bn_q2.tight_layout()

    fig_An_q1.savefig(plot_prefix + "An_q1.png")
    fig_Bn_q1.savefig(plot_prefix + "Bn_q1.png")
    fig_An_q2.savefig(plot_prefix + "An_q2.png")
    fig_Bn_q2.savefig(plot_prefix + "Bn_q2.png")

    color = 'tab:red'
    fig_An_Gamma, ax_An_Gamma = plt.subplots()
    ax_An_Gamma.plot(R[0, :], An_Gamma[:, 0], color=color)
    ax_An_Gamma.set_xlabel(r'$r / \xi$')
    ax_An_Gamma.set_ylabel(r'$\Gamma_0 (r)$', color=color)
    ax_An_Gamma.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2_An_Gamma = ax_An_Gamma.twinx()
    ax2_An_Gamma.plot(R[0, :], -An_Gamma[:, 1], color=color)
    ax2_An_Gamma.set_ylabel(r'$-\Gamma_1 (r)$', color=color)
    ax2_An_Gamma.tick_params(axis='y', labelcolor=color)
    ax2_An_Gamma.set_ylim([-0.001, 0.025])
    fig_An_Gamma.tight_layout()

    color = 'tab:red'
    fig_An_Gamma, ax_An_Gamma = plt.subplots()
    ax_An_Gamma.plot(R[0, :], An_Gamma[:, 0], color=color)
    ax_An_Gamma.set_xlabel(r'$r / \xi$')
    ax_An_Gamma.set_ylabel(r'$\Gamma_0 (r)$', color=color)
    ax_An_Gamma.tick_params(axis='y', labelcolor=color)
    ax_An_Gamma.set_xlim([-0.1, 0.5])
    ax_An_Gamma.set_ylim([-6e-3, 3.5e-2])

    color = 'tab:blue'
    ax2_An_Gamma = ax_An_Gamma.twinx()
    ax2_An_Gamma.plot(R[0, :], (-(An_Gamma[:, 1] - An_Gamma[0, 1] - 1e-8))**0.5, color=color)
    ax2_An_Gamma.plot(R[0, :], (-(An_Gamma[:, 1] - An_Gamma[0, 1] - 1e-8))**(1/3), color='tab:green')
    ax2_An_Gamma.set_ylabel(r'$-\Gamma_1 (r)$', color=color)
    ax2_An_Gamma.tick_params(axis='y', labelcolor=color)
    # ax2_An_Gamma.set_ylim([-4e-4, 2e-3])
    ax2_An_Gamma.set_ylim([-4e-4, 8e-3])
    ax2_An_Gamma.set_xlim([-0.1, 0.5])
    fig_An_Gamma.tight_layout()

    plt.show()



if __name__ == "__main__":

    main()
