import argparse
import os

import h5py
import numpy as np
import quadpy

from ..utilities import nematics as nu

def get_commandline_args():
    

    description = ('Given an h5 file with Lambda, Z, and Q_vec some (n, 5) '
                   '(n,), and (n, 5) shaped arrays respectively, find a point '
                   'closest to the input point and plot the PDF on a sphere at '
                   'that point')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where defect location data lives')
    parser.add_argument('--input_filename', dest='input_filename',
                        help='h5 file with the data')
    parser.add_argument('--output_filename', dest='output_filename',
                        help='name of output plot')
    parser.add_argument('--Q_key', 
                        dest='Q_key',
                        help='data key for Q values in h5 file')
    parser.add_argument('--Lambda_key', 
                        dest='Lambda_key',
                        default='Lambda',
                        help='data key for Lambda values in h5 file')
    parser.add_argument('--Z_key', 
                        dest='Z_key',
                        default='Z',
                        help='data key for Z values in h5 file')

    parser.add_argument('--point',
                        dest='point',
                        default=[0, 0],
                        nargs='2',
                        type=float,
                        help='point at which PDF is being evaluated')
    parser.add_argument('--order',
                        dest='order',
                        default=590,
                        type=int,
                        help='order of spherical integration in PDF evaluation')
    args = parser.parse_args()

    input_filename = os.path.join(args.data_folder, args.input_filename)
    output_filename = os.path.join(args.data_folder, args.input_filename)

    return (input_filename, output_filename, args.Q_key, args.Lambda_key, 
            args.Z_key, args.point, args.order)
    


def main():

    (input_filename, output_filename, 
     Q_key, Lambda_key, Z_key, point, order) = get_commandline_args()

    file = h5py.file(input_filename, 'r')
    data = file[Q_key]
    Q_vec = np.array(data[:])
    Lambda_vec = np.array(file[Lambda_key])
    Z_vec = np.array(file[Z_key])

    r0 = data.attrs['r0']
    rf = data.attrs['rf']
    n_r = data.attrs['n_r']
    n_theta = data.attrs['n_theta']
    r = np.linspace(r0, rf, num=n_r)
    theta = np.linspace(0, 2*np.pi, num=n_theta)

    r_point = np.sqrt(point[0]**2 + point[1]**2)
    theta_point = np.atan2(point[1], point[0])

    r_idx = np.argmin(np.abs(r - r_point))
    theta_idx = np.argmin(np.abs(theta - theta_point))

    idx = r_idx * n_theta + theta_idx

    Q = np.zeros(3, 3)
    Lambda = np.zeros(3, 3)

    Q[0, 0] = Q_vec[idx, 0]
    Q[0, 1] = Q_vec[idx, 1]
    Q[0, 2] = Q_vec[idx, 2]
    Q[1, 1] = Q_vec[idx, 3]
    Q[1, 2] = Q_vec[idx, 4]
    Q[1, 0] = Q[0, 1]
    Q[2, 0] = Q[0, 2]
    Q[2, 1] = Q[1, 2]
    Q[2, 2] = -(Q[0, 0] + Q[1, 1])

    Lambda[0, 0] = Lambda_vec[idx, 0]
    Lambda[0, 1] = Lambda_vec[idx, 1]
    Lambda[0, 2] = Lambda_vec[idx, 2]
    Lambda[1, 1] = Lambda_vec[idx, 3]
    Lambda[1, 2] = Lambda_vec[idx, 4]
    Lambda[1, 0] = Lambda[0, 1]
    Lambda[2, 0] = Lambda[0, 2]
    Lambda[2, 1] = Lambda[1, 2]
    Lambda[2, 2] = -(Lambda[0, 0] + Lambda[1, 1])

    Z = Z_vec[idx]

    theta_sphere = np.linspace(0, np.pi, num=1000)
    phi_sphere = np.linspace(0, 2*np.pi, num=1000)

    Theta_sphere, Phi_sphere = np.meshgrid(theta_sphere, phi_sphere)
    X = np.sin(Theta_sphere) * np.cos(Phi_sphere)
    Y = np.sin(Theta_sphere) * np.sin(Phi_sphere)
    Z = np.cos(Theta_sphere)

    rho = 1 / Z * np.exp(

if __name__ == '__main__':
    main()
