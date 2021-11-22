import sys
import argparse
import re
import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

def readCodyData(cody_data_filename):

    # read in Cody's data
    cody_data = h5py.File(cody_data_filename, "r")
    
    # parse data
    if 'Q1' in cody_data.keys():
        Q1 = cody_data['Q1'][:]
        Q2 = cody_data['Q2'][:]
        Q3 = cody_data['Q3'][:]
        Q4 = cody_data['Q4'][:]
        Q5 = cody_data['Q5'][:]
        Q = parseQVector(Q1, Q2, Q3, Q4, Q5)

        X = cody_data['X']
        Y = cody_data['Y']

    elif 'eta' in cody_data.keys():
        eta = cody_data['eta'][:]
        mu = cody_data['mu'][:]
        nu = cody_data['nu'][:]
        Q = parseAuxVariables(eta, mu, nu, True)

        x = cody_data['x'][:]
        X, Y = np.meshgrid(x, x, indexing='ij')

    return Q, X, Y

def parseQVector(Q1, Q2, Q3, Q4, Q5):

    Q = np.zeros((3, 3) + np.shape(Q1))
    Q[0, 0, :, :] = Q1
    Q[0, 1, :, :] = Q2
    Q[0, 2, :, :] = Q3
    Q[1, 0, :, :] = Q2
    Q[1, 1, :, :] = Q4
    Q[1, 2, :, :] = Q5
    Q[2, 0, :, :] = Q3
    Q[2, 1, :, :] = Q5
    Q[2, 2, :, :] = -(Q1 + Q4)
    
    return Q

def parseAuxVariables(eta, mu, nu, old_convention=False):

    if old_convention:
        eta *= 1 / np.sqrt(3)

    Q = np.zeros((3, 3) + np.shape(eta))
    Q[0, 0, :, :] = 2 / np.sqrt(3) * eta
    Q[0, 1, :, :] = nu
    Q[0, 2, :, :] = 0
    Q[1, 0, :, :] = nu
    Q[1, 1, :, :] = -1 / np.sqrt(3) * eta + mu
    Q[1, 2, :, :] = 0
    Q[2, 0, :, :] = 0
    Q[2, 1, :, :] = 0
    Q[2, 2, :, :] = -1 / np.sqrt(3) * eta - mu
        
    return Q

def calcDirectorAndS(Q):

    grid_shape = Q.shape[2:]

    S = np.zeros(grid_shape)
    n = np.zeros((3,) + grid_shape)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            w, v = np.linalg.eig(Q[:, :, i, j])

            # max eigenvalue corresponds to S & director direction
            max_idx = np.argmax(w)
            S[i, j] = (3/2)*w[max_idx]
            n[:, i, j] = v[:, max_idx]

    return n, S



def plotDirectorAndS(X, Y, n, S, title="Defect configuration"):
    
    grid_shape = S.shape

    # make mask so director isn't plotted at every gridpoint
    stride = np.array([i for i in range(0, grid_shape[0], 10)])
    sparse_idx = np.ix_(stride, stride)

    # vector components of director field
    U = n[0, :, :]
    V = n[1, :, :]

    # plot S and director
    fig, ax = plt.subplots()
    c = ax.pcolor(X, Y, S)
    q = ax.quiver(
                X[sparse_idx], Y[sparse_idx], U[sparse_idx], V[sparse_idx],
                headwidth=0, pivot='middle', headaxislength=5, scale=30, 
                width=0.002)

    # make plot look nice
    fig.colorbar(c, ax=ax, label="S Value")
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()

    return fig, ax, q



def readLucasData(lucas_data_filename):
    
    with h5py.File(lucas_data_filename, "r") as f:
        Q1_dset = f['Q1']
        Q1 = np.zeros(Q1_dset.shape)
        Q1_dset.read_direct(Q1)
        
        Q2_dset = f['Q2']
        Q2 = np.zeros(Q2_dset.shape)
        Q2_dset.read_direct(Q2)
        
        Q3_dset = f['Q3']
        Q3 = np.zeros(Q3_dset.shape)
        Q3_dset.read_direct(Q3)
        
        Q4_dset = f['Q4']
        Q4 = np.zeros(Q4_dset.shape)
        Q4_dset.read_direct(Q4)
        
        Q5_dset = f['Q5']
        Q5 = np.zeros(Q5_dset.shape)
        Q5_dset.read_direct(Q5)

        X_dset = f['X']
        X = np.zeros(X_dset.shape)
        X_dset.read_direct(X)
    
        Y_dset = f['Y']
        Y = np.zeros(Y_dset.shape)
        Y_dset.read_direct(Y)

    Q = np.zeros((3, 3) + Q1.shape)
    Q[0, 0, :, :] = Q1
    Q[0, 1, :, :] = Q2
    Q[0, 2, :, :] = Q3
    Q[1, 1, :, :] = Q4
    Q[1, 2, :, :] = Q5
    Q[1, 0, :, :] = Q2
    Q[2, 0, :, :] = Q3
    Q[2, 1, :, :] = Q5
    Q[2, 2, :, :] = -(Q1 + Q4)
    
    return Q, X, Y



def rotateQ(Q):

    # rotation matrix
    R = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])

    # rotate Q-tensor at each gridpoint
    Q_rot = np.zeros(Q.shape)
    for i in range(Q.shape[2]):
        for j in range(Q.shape[3]):
            Q_new = np.matmul(Q[:, :, i, j], R)
            Q_new = np.matmul(R.transpose(), Q_new)
            Q_rot[:, :, i, j] = Q_new
            
    # rotate Q-tensors around the grid
    for i in range(Q_rot.shape[0]):
        for j in range(Q_rot.shape[1]):
            Q_rot[i, j, :, :] = np.rot90(Q_rot[i, j, :, :], k=3)

    return Q_rot



def calcQNormedDifference(Q1, Q2):

    # only calculate degree of freedom differences
    Q_diff = np.sqrt( (Q1[0, 0, :, :] - Q2[0, 0, :, :])**2
                    + (Q1[0, 1, :, :] - Q2[0, 1, :, :])**2
                    + (Q1[0, 2, :, :] - Q2[0, 2, :, :])**2
                    + (Q1[1, 1, :, :] - Q2[1, 1, :, :])**2
                    + (Q1[1, 2, :, :] - Q2[1, 2, :, :])**2 )

    return Q_diff



def plotQNormedDifference(X, Y, Q_diff, title="Normed Q difference"):

    # plot S and director
    fig, ax = plt.subplots()
    c = ax.pcolor(X, Y, Q_diff)
        
    # make plot look nice
    fig.colorbar(c, ax=ax, label="Normed Q difference")
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()

    return fig, ax



if __name__ == "__main__":

    description = "Plots cody's data, my data, and then the norm of the difference"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cody_folder', dest='cody_folder',
                        help='folder where data from cody is stored')
    parser.add_argument('--cody_filename', dest='cody_filename',
                        help='name of data file from cody')
    parser.add_argument('--lucas_folder', dest='lucas_folder',
                        help='folder where data from lucas is stored')
    parser.add_argument('--lucas_filename', dest='lucas_filename',
                        help='name of data file from lucas')
    parser.add_argument('--cody_plot_filename', dest='cody_plot_filename',
                        help='filename of plot of cody data')
    parser.add_argument('--lucas_plot_filename', dest='lucas_plot_filename',
                        help='filename of plot of lucas data')
    parser.add_argument('--diff_plot_filename', dest='diff_plot_filename',
                        help='filename of normed difference plot')
    args = parser.parse_args() 

    cody_data_filename = os.path.join(args.cody_folder, args.cody_filename)
    if args.cody_plot_filename:
        cody_plot_filename = args.cody_plot_filename
    else:
        cody_plot_filename = re.sub(r'(.*).h5', r'\1.png', cody_data_filename)

    # read and plot cody data
    Q_cody, X, Y = readCodyData(cody_data_filename)
    n_cody, S_cody = calcDirectorAndS(Q_cody)
    fig_cody, _, _ = plotDirectorAndS(
                            X, Y, n_cody, S_cody,
                            title="-1/2 Defect from Cody, Isotropic elasticity")
    plot_filename_cody = "min_1-2_defect_iso_cody.png"
    fig_cody.savefig(cody_plot_filename)

    if args.lucas_filename:
        if args.lucas_folder:
            lucas_folder = args.lucas_folder
        else:
            lucas_folder = args.cody_folder
            
        lucas_data_filename = os.path.join(lucas_folder, args.lucas_filename)

        if args.lucas_plot_filename:
            lucas_plot_filename = os.path.join(lucas_folder, 
                                               args.lucas_plot_filename)
        else:
            lucas_plot_filename = re.sub(r'(.*).h5', r'\1.png', 
                                         lucas_data_filename)
                                         
        # read and plot lucas data
        Q_lucas, X, Y = readLucasData()
        Q_lucas = rotateQ(Q_lucas)
        n_lucas, S_lucas = calcDirectorAndS(Q_lucas)
        fig_lucas, _, _ = plotDirectorAndS(
                                X, Y, n_lucas, S_lucas,
                                title="-1/2 Defect from Lucas, Isotropic elasticity"
                                )
        plot_filename_lucas = "min_1-2_defect_iso_lucas.png"
        fig_lucas.savefig(plot_filename_lucas)

        if args.diff_plot_filename:
            plot_filename_diff = os.path.join(lucas_folder, diff_plot_filename)
        else:
            plot_filename_diff = os.path.join(lucas_folder, "diff-plot.png")

        # plot normed difference
        Q_diff = calcQNormedDifference(Q_cody, Q_lucas)
        fig_diff, _ = plotQNormedDifference(
                            X, Y, Q_diff, 
                            title="Normed difference between Cody and Lucas data")
        plot_filename_diff = "min_1-2-defect_iso_difference.png"
        fig_diff.savefig(plot_filename_diff)