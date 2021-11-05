import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

def readCodyData():

    # get file location from environment variable + relative location
    maier_saupe_dir = os.getenv('MAIER_SAUPE_DIR')
    cody_mat_filename = os.path.join(
                                maier_saupe_dir, 
                                'examples', 
                                'cody_data', 
                                'PlusHalfQ.mat')

    # read in Cody's data
    cody_data = sio.loadmat(cody_mat_filename)
    
    # parse data
    Q = cody_data['Q']
    X = cody_data['X']
    Y = cody_data['Y']

    return Q, X, Y



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



def readLucasData():

    # get file location from environment variable + relative location
    maier_saupe_dir = os.getenv('MAIER_SAUPE_DIR')
    lucas_data_filename = os.path.join(
                                maier_saupe_dir, 'data', 'IsoSteadyState', 
                                '2021-09-16', 'minus-half-defect-me.hdf5')
    
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

    # read and plot cody data
    Q_cody, X, Y = readCodyData()
    n_cody, S_cody = calcDirectorAndS(Q_cody)
    fig_cody, _, _ = plotDirectorAndS(
                            X, Y, n_cody, S_cody,
                            title="-1/2 Defect from Cody, Isotropic elasticity")
    plot_filename_cody = "min_1-2_defect_iso_cody.png"
    fig_cody.savefig(plot_filename_cody)

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

    # plot normed difference
    Q_diff = calcQNormedDifference(Q_cody, Q_lucas)
    fig_diff, _ = plotQNormedDifference(
                        X, Y, Q_diff, 
                        title="Normed difference between Cody and Lucas data")
    plot_filename_diff = "min_1-2-defect_iso_difference.png"
    fig_diff.savefig(plot_filename_diff)