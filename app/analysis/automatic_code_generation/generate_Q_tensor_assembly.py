import argparse
import enum

import sympy as sy

from ..utilities import tensor_calculus as tc

class Basis(enum.Enum):
    component_wise = enum.auto()
    orthogonal = enum.auto()



def get_commandline_args():

    descrption = ('Runs symbolic calculation of each of the terms in the '
                  'nematic equation of motion and generates code necessary '
                  'for matrix assembly in the finite element code')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--basis',
                        dest='basis',
                        type=str,
                        choices={'component_wise', 'orthogonal'},
                        help='name of traceless, symmetric tensor basis to use')

    args = parser.parse_args()

    return Basis[args.basis]



def main():

    vec_dim = 5
    mat_dim = 3
    
    x, y, z = sy.symbols('x y z')
    coords = (x, y)
    # coords = (x, y, z) # uncomment for 3D
    xi = tc.TensorCalculusArray([x, y, z])
    
    alpha, dt, L2, L3, Z, theta = sy.symbols(r'\alpha \delta\ t L_2 L_3 Z theta')
    A, B, C = sy.symbols(r'A B C')
    
    def make_function_array(dim, label):
        vec = sy.zeros(dim, 1)
        for i in range(dim):
            vec[i] = sy.Function(label.format(i))(*coords)
    
        return vec
    
    Q_vec = make_function_array(vec_dim, 'Q_{}')
    Q0_vec = make_function_array(vec_dim, 'Q_{{0{} }}')
    Lambda_vec = make_function_array(vec_dim, r'\Lambda_{}')
    Lambda0_vec = make_function_array(vec_dim, r'\Lambda_{{0{} }}')
    delta_Q_vec = make_function_array(vec_dim, r'\delta\ Q_{}')
    phi_vec = make_function_array(vec_dim, r'\phi_{}')
    
    phi_i = sy.Function(r'\phi_i')(*coords)
    phi_j = sy.Function(r'\phi_j')(*coords)


if __name__ == "__main__":
    main()
