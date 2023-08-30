import argparse

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def get_commandline_args():

    description = ('Reads numerical dzyaloshinskii from h5 file and compares '
                   'with corresponding analytic approximation')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--solution_filename',
                        help='Name of hdf5 file containing the solution')
    parser.add_argument('--eps',
                        type=float,
                        help='Value of anisotropy parameter epsilon')
    parser.add_argument('--charge',
                        type=float,
                        help='charge of the disclination')
    args = parser.parse_args()

    return args.solution_filename, args.eps, args.charge



def perturbative_disclination_director(polar_angle, charge, eps):

    return (
            eps * charge * (2 - charge) / (4 * (1 - charge)**2)
            * np.sin(2 * (1 - charge) * polar_angle)
            +
            charge * polar_angle
            )



def main():

    solution_filename, eps, charge = get_commandline_args()
    file = h5py.File(solution_filename)

    director_angle = np.array(file['phi'][:])
    polar_angle = np.array(file['theta'][:])

    approx_director_angle = perturbative_disclination_director(polar_angle, charge, eps)

    plt.plot(polar_angle, director_angle, label='numerical')
    plt.plot(polar_angle, approx_director_angle, label='perturbative', linestyle='--')
    plt.xlabel('Polar angle')
    plt.ylabel('Director angle')
    plt.title(r'$\epsilon = {:.2f}$, $q = {:.2f}$'
              .format(eps, charge))
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
