import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
import h5py

dpi = 300
mpl.rcParams['figure.dpi'] = dpi

plt.style.use('science')


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
    fig.colorbar(c, ax=ax# , label="S Value"
                 )
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    # ax.set_title(title)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()

    return fig, ax, q
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

def readData(data_filename):

    # read in Cody's data
    data = h5py.File(data_filename, "r")

    # parse data
    if 'Q1' in data.keys():
        Q1 = data['Q1'][:]
        Q2 = data['Q2'][:]
        Q3 = data['Q3'][:]
        Q4 = data['Q4'][:]
        Q5 = data['Q5'][:]
        Q = parseQVector(Q1, Q2, Q3, Q4, Q5)


    elif 'eta' in data.keys():
        eta = data['eta'][:]
        mu = data['mu'][:]
        nu = data['nu'][:]
        Q = parseAuxVariables(eta, mu, nu, True)

    X = data['X'][:]
    Y = data['Y'][:]

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
    fig.colorbar(c, ax=ax# , label="S Value"
                 )
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    # ax.set_title(title)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()

    return fig, ax, q



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
    fig.colorbar(c, ax=ax# , label="Normed Q difference"
                 )
    ax.set_xlabel(r"$x/\xi$")
    ax.set_ylabel(r"$y/\xi$")
    # ax.set_title(title)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()

    return fig, ax



if __name__ == "__main__":

    description = "Find defect centers of two configurations, realign if they are not in the same place"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cody_folder', dest='cody_folder',
                        help='folder where data from cody is stored')
    parser.add_argument('--cody_filename', dest='cody_filename',
                        help='name of data file from cody')
    parser.add_argument('--lucas_folder', dest='lucas_folder',
                        help='folder where data from lucas is stored')
    parser.add_argument('--save_folder', dest='save_folder',
                        help='folder where plots are saved')
    parser.add_argument('--lucas_filename', dest='lucas_filename',
                        help='name of data file from lucas')
    parser.add_argument('--cody_plot_filename', dest='cody_plot_filename',
                        help='filename of plot of cody data')
    parser.add_argument('--lucas_plot_filename', dest='lucas_plot_filename',
                        help='filename of plot of lucas data')
    parser.add_argument('--diff_plot_filename', dest='diff_plot_filename',
                        help='filename of normed difference plot')
    parser.add_argument('--norm_diff_file', dest='norm_diff_file',
                        help='filename of normed difference plot')
    parser.add_argument('--S_crosssection_file', dest='S_crosssection_file',
                        help='Name of file for S crosssection plot')
    parser.add_argument('--configuration_file', dest='configuration_file',
                        help='Name of file for +1/2 configuration plot')
    args = parser.parse_args()

    cody_filename = os.path.join(args.cody_folder, args.cody_filename)
    lucas_filename = os.path.join(args.lucas_folder, args.lucas_filename)
    Q_cody, X_cody, Y_cody = readData(cody_filename)
    Q_lucas, X_lucas, Y_lucas = readData(lucas_filename)

    n_cody, S_cody = calcDirectorAndS(Q_cody)
    n_lucas, S_lucas = calcDirectorAndS(Q_lucas)

    cody_defect_pos = np.unravel_index(np.argmin(S_cody), S_cody.shape)
    lucas_defect_pos = np.unravel_index(np.argmin(S_lucas), S_lucas.shape)

    print(cody_defect_pos)
    print(lucas_defect_pos)

    # Q_cody = Q_cody[:, :, 2:, 1:-1]
    # Q_lucas = Q_lucas[:, :, 1:-1, 1:-1]
    X = X_lucas
    Y = Y_lucas

    n_cody, S_cody = calcDirectorAndS(Q_cody)
    n_lucas, S_lucas = calcDirectorAndS(Q_lucas)

    cody_defect_pos = np.unravel_index(np.argmin(S_cody), S_cody.shape)
    lucas_defect_pos = np.unravel_index(np.argmin(S_lucas), S_lucas.shape)

    print(cody_defect_pos)
    print(lucas_defect_pos)

    Q_diff = calcQNormedDifference(Q_cody, Q_lucas)
    fig, ax = plotQNormedDifference(X / np.sqrt(2), Y / np.sqrt(2), Q_diff, title="Normed Q difference")
    fig.savefig(os.path.join(args.save_folder, args.norm_diff_file))

    fig, ax = plt.subplots()
    ax.plot(X[:, 127] / np.sqrt(2), S_lucas[:, 127]# , label="Lucas S along X"
            )
    # ax.plot(Y[127, :], S_lucas[127, :], label="Lucas S along Y")
    # ax.plot(X[:, 127], S_cody[:, 127], label="Cody S along X")
    # ax.plot(Y[127, :], S_cody[127, :], label="Cody S along Y")
    # ax.legend()
    ax.set_xlabel(r'$x/\xi$')
    ax.set_ylabel(r'$S$')
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_folder, args.S_crosssection_file))

    fig, ax, q = plotDirectorAndS(X/np.sqrt(2), Y/np.sqrt(2), n_lucas, S_lucas, title="Defect configuration")
    fig.savefig(os.path.join(args.save_folder, args.configuration_file))

    plt.show()
