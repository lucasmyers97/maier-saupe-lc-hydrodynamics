import argparse

import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

def get_commandline_args():

    desc = ('Given an input hdf5 file with a dataset that contains the degrees '
            'of freedom of the singular potential Lambda, arranged as an '
            '`n_points`x`msc::vec_dim<dim>` array, output colormap spheres of '
            'the probability distribution function.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--filename',
                        help='h5 file containing singular potential values')
    parser.add_argument('--Lambda_key',
                        help='name of key in h5 file which holds the singular potential values')
    parser.add_argument('--Z_key',
                        help='name of key in h5 file which holds the singular potential partition function values')
    parser.add_argument('--Q_key',
                        help='name of key in h5 file which holds the Q-tensor values')
    args = parser.parse_args()

    return args.filename, args.Lambda_key, args.Z_key, args.Q_key



def main():

    filename, Lambda_key, Z_key, Q_key = get_commandline_args()
    file = h5py.File(filename)
    Lambda = np.array(file[Lambda_key][:])
    Z = np.array(file[Z_key][:])
    Q = np.array(file[Q_key][:])

    # Make data
    n = 300
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta)

    rho_list = []
    for i in range(Z.shape[0]):

        if (Z[i] == np.inf):
            continue
        rho_list.append( (1 / Z[i]) * np.exp(Lambda[i, 0] * (x*x - z*z)
                                             + 2 * Lambda[i, 1] * x*y
                                             + 2 * Lambda[i, 2] * x*z
                                             + Lambda[i, 3] * (y*y - z*z)
                                             + 2 * Lambda[i, 4] * y*z)
                        )

    total_min = 100000
    total_max = -1000000
    for rho in rho_list:
        local_min = np.amin(rho)
        local_max = np.amax(rho)
        if local_min < total_min:
            total_min = local_min
        if local_max > total_max:
            total_max = local_max

    # norm = mpl.colors.Normalize(vmin=total_min, vmax=total_max)
    
    for i, rho in enumerate(rho_list):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    
        norm = mpl.colors.Normalize(vmin=np.amin(rho), vmax=np.amax(rho))

        # Plot the surface
        ax.plot_surface(x, y, z, facecolors=cm.jet(norm(rho)), rstride=1, cstride=1)

        # Plot axes
        x0 = np.array([-1.2, -1.2, -1.2])
        y0 = np.array([-1.2, -1.2, -1.2])
        z0 = np.array([-1.2, -1.2, -1.2])
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        w = np.array([0, 0, 1])
        ax.quiver(x0, y0, z0, u, v, w, arrow_length_ratio=0.1)

        ax.text(0, -1.2, -1.2, 'x')
        ax.text(-1.2, 0, -1.2, 'y')
        ax.text(-1.2, -1.2, 0, 'z')
        
        # Set an equal aspect ratio
        ax.set_aspect('equal')

        # Get rid of axes
        ax.set_axis_off()

        ax.set_title(r'$S = {}$'.format(Q[i, 0]))

        m = cm.ScalarMappable(cmap=cm.jet, norm=norm)
        fig.colorbar(m, ax=ax)

        fig.tight_layout()
        fig.savefig('pdf_Q_{}.png'.format(Q[i, 0]))
        
        plt.show()

if __name__ == '__main__':
    main()
