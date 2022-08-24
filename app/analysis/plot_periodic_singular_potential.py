"""
This script just reads singular potential values and the corresponding Jacobian
values from an hdf5 file as a function of x.
I think this had something to do with making sure that the singular potential
was working given a periodic Q-tensor configuration.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def print_amplitude_mean(y):

    y_max = np.max(y)
    y_min = np.min(y)
    mean = (y_max + y_min) / 2

    print("Amplitude is: {}, Mean is: {}".format(y_max - mean, mean))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="HDF5 file that holds data")
    parser.add_argument("--output_folder", help="Location to output plots")
    args = parser.parse_args()

    file = h5py.File(args.filename)
    x = file['x'][:]
    Lambda = file['Lambda'][:]
    Jac = file['Jac'][:]

    fig, ax = plt.subplots()
    ax.plot(x, Lambda[:, 0])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\Lambda_1$')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "Lambda1.png"))
    print("Lambda1")
    print_amplitude_mean(Lambda[:, 0])

    fig, ax = plt.subplots()
    ax.plot(x, Lambda[:, 1])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\Lambda_2$')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "Lambda2.png"))
    print("Lambda2")
    print_amplitude_mean(Lambda[:, 1])

    fig, ax = plt.subplots()
    ax.plot(x, Lambda[:, 3])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\Lambda_4$')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "Lambda4.png"))
    print("Lambda4")
    print_amplitude_mean(Lambda[:, 3])

    fig, ax = plt.subplots()
    ax.plot(x, Jac[:, 0, 1])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\frac{\partial \Lambda_1}{\partial Q_2}$')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "dLambda1_dQ2.png"))
    print("dLambda1_dQ2")
    print_amplitude_mean(Jac[:, 0, 1])

    fig, ax = plt.subplots()
    ax.plot(x, Jac[:, 1, 1])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\frac{\partial \Lambda_2}{\partial Q_2}$')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "dLambda2_dQ2.png"))
    print("dLambda2_dQ2")
    print_amplitude_mean(Jac[:, 1, 1])

    fig, ax = plt.subplots()
    ax.plot(x, Jac[:, 3, 1])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\frac{\partial \Lambda_4}{\partial Q_2}$')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "dLambda4_dQ2.png"))
    print("dLambda4_dQ2")
    print_amplitude_mean(Jac[:, 3, 1])


    plt.show()

if __name__ == "__main__":

    main()
