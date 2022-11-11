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
    S = 1.5 * np.array(grp['q1'][:])
    P = 0.5 * np.array(grp['q1'][:]) + np.array(grp['q2'][:])
    S = S.reshape(point_dims)
    P = P.reshape(point_dims)

    An_S = np.zeros((R.shape[1], n_modes))
    Bn_S = np.zeros((R.shape[1], n_modes))
    An_P = np.zeros((R.shape[1], n_modes))
    Bn_P = np.zeros((R.shape[1], n_modes))
    for i in range(R.shape[1]):

        s = S[:, i]
        p = P[:, i]

        FT_s = np.fft.rfft(s)
        FT_p = np.fft.rfft(p)

        N = s.shape[0]
        An_s = FT_s.real / N
        Bn_s = FT_s.imag / N
        An_p = FT_p.real / N
        Bn_p = FT_p.imag / N

        for j in range(n_modes):
            An_S[i, j] = An_s[j]
            An_P[i, j] = An_p[j]

        for j in range(1, n_modes):
            Bn_S[i, j] = Bn_s[j]
            Bn_P[i, j] = Bn_p[j]

    fig_An_S, ax_An_S = plt.subplots()
    fig_Bn_S, ax_Bn_S = plt.subplots()
    fig_An_P, ax_An_P = plt.subplots()
    fig_Bn_P, ax_Bn_P = plt.subplots()
    for j in range(n_modes):
        ax_An_S.plot(R[0, :], An_S[:, j], label="k = {}".format(j))
        ax_An_P.plot(R[0, :], An_P[:, j], label="k = {}".format(j))

    for j in range(1, n_modes):
        ax_Bn_S.plot(R[0, :], Bn_S[:, j], label="k = {}".format(j))
        ax_Bn_P.plot(R[0, :], Bn_P[:, j], label="k = {}".format(j))

    ax_An_S.set_xlabel("radius")
    ax_An_S.set_ylabel("Fourier amplitude")
    ax_An_S.set_title(r'$\cos$ Fourier coefficients for $S$')
    ax_An_S.legend()

    ax_Bn_S.set_xlabel("radius")
    ax_Bn_S.set_ylabel("Fourier amplitude")
    ax_Bn_S.set_title(r'$\sin$ Fourier coefficients for $S$')
    ax_Bn_S.legend()

    ax_An_P.set_xlabel("radius")
    ax_An_P.set_ylabel("Fourier amplitude")
    ax_An_P.set_title(r'$\cos$ Fourier coefficients for $P$')
    ax_An_P.legend()

    ax_Bn_P.set_xlabel("radius")
    ax_Bn_P.set_ylabel("Fourier amplitude")
    ax_Bn_P.set_title(r'$\sin$ Fourier coefficients for $P$')
    ax_Bn_P.legend()

    fig_An_S.tight_layout()
    fig_Bn_S.tight_layout()
    fig_An_P.tight_layout()
    fig_Bn_P.tight_layout()

    fig_An_S.savefig(plot_prefix + "An_S.png")
    fig_Bn_S.savefig(plot_prefix + "Bn_S.png")
    fig_An_P.savefig(plot_prefix + "An_P.png")
    fig_Bn_P.savefig(plot_prefix + "Bn_P.png")

    plt.show()



if __name__ == "__main__":

    main()
