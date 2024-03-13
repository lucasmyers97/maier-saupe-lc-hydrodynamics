import enum
import argparse

import sympy as sy

from ..utilities import tensor_calculus as tc
from ..utilities import dealii_code_generation as dcg
from . import Q_tensor_discretization_equations as qtde

class Basis(enum.Enum):
    component_wise = enum.auto()
    orthogonal = enum.auto()

def get_commandline_args():

    descrption = ('Runs symbolic calculation of each of the terms in the '
                  'nematic equation of motion and generates code necessary '
                  'for matrix assembly in the finite element code')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--basis',
                        choices={'component_wise', 'orthogonal'},
                        help='name of traceless, symmetric tensor basis to use')
    parser.add_argument('--space_dim',
                        choices={2, 3},
                        type=int,
                        help='space dimension of discretization')
    args = parser.parse_args()

    return Basis[args.basis], args.space_dim



def get_3x3_traceless_symmetric_tensor_basis(basis_enum):
    """
    Get a list of 3x3 traceless, symmetric tensors which form a basis for the
    space of traceless, symmetric tensors.
    Can choose which type of basis based on `basis_enum`. 
    'component_wise' enum variant just numbers components of the matrix 1-5.
    'orthogonal' gives basis elements which are orthogonal (and have the same
    norm) under the tensor inner product.

    Parameters
    ----------
    basis_enum : Basis
        Choose between component-wise and orthogonal bases for tensors

    Returns
    -------
    basis : list of tensor_calculus.TensorCalculusArray
        list of 3x3 TensorCalculusArray elements, each of which is a basis element
    """

    basis = []
    if basis_enum == Basis.component_wise:
        basis.append(tc.TensorCalculusArray([[1, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, -1]]))
        basis.append(tc.TensorCalculusArray([[0, 1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]]))
        basis.append(tc.TensorCalculusArray([[0, 0, 1],
                                             [0, 0, 0],
                                             [1, 0, 0]]))
        basis.append(tc.TensorCalculusArray([[0, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, -1]]))
        basis.append(tc.TensorCalculusArray([[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 1, 0]]))
    elif basis_enum == Basis.orthogonal:
        basis.append(tc.TensorCalculusArray([[2/sy.sqrt(3), 0, 0],
                                             [0, -1/sy.sqrt(3), 0],
                                             [0, 0, -1/sy.sqrt(3)]]))
        basis.append(tc.TensorCalculusArray([[0, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, -1]]))
        basis.append(tc.TensorCalculusArray([[0, 1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]]))
        basis.append(tc.TensorCalculusArray([[0, 0, 1],
                                             [0, 0, 0],
                                             [1, 0, 0]]))
        basis.append(tc.TensorCalculusArray([[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 1, 0]]))
    else:
        raise ValueError('Incorrect argument for generating tensor basis')
    
    return basis



def set_up_symbols(vec_dim, basis_enum, space_dim):

    x, y, z = sy.symbols('x y z')

    coords = None
    if (space_dim == 2):
        coords = (x, y)
    elif (space_dim == 3):
        coords = (x, y, z)
    else:
        raise ValueError('Space dimension must be 2 or 3')

    xi = tc.TensorCalculusArray([x, y, z])
    
    Q_vec = tc.make_function_vector(vec_dim, 'Q_{}', coords)

    basis = get_3x3_traceless_symmetric_tensor_basis(basis_enum)

    Q = tc.make_tensor_from_vector(Q_vec, basis)


    return xi, Q_vec, Q



def set_up_code_symbols(xi, Q_vec):

    Q_code = {Q_vec[i]: 'Q_vec[q][{}]'.format(i) 
              for i in range(Q_vec.shape[0])}
    dQ_code = {Q_vec[i].diff(xi[j]): 'dQ[q][{}][{}]'.format(i, j)
               for i in range(Q_vec.shape[0])
               for j in range(xi.shape[0])}

    return Q_code | dQ_code



def main():

    basis_enum, space_dim = get_commandline_args()
    vec_dim = 5

    xi, Q_vec, Q = set_up_symbols(vec_dim, basis_enum, space_dim)

    user_funcs = set_up_code_symbols(xi, Q_vec)

    D = tc.TensorCalculusArray([[ sum(sy.LeviCivita(gamma, mu, nu)
                                      * sy.LeviCivita(i, k, l)
                                      * Q[mu, alpha].diff(xi[k])
                                      * Q[nu, alpha].diff(xi[l])
                                      for mu in range(space_dim)
                                      for nu in range(space_dim)
                                      for alpha in range(space_dim)
                                      for k in range(space_dim)
                                      for l in range(space_dim))
                                  for gamma in range(space_dim)]
                                 for i in range(space_dim)])

    D = sy.simplify(D)

    # omega_squared = sum(D[gamma, i] * D[gamma, i]
    #                     for gamma in range(space_dim)
    #                     for i in range(space_dim))

    # omega_squared = sy.simplify(omega_squared)

    # print( len(omega_squared.args) )

    printer = dcg.MyPrinter(user_funcs)

    # print(printer.doprint(omega_squared))
    for i in range(3):
        for j in range(3):
            print('D[{}, {}] is: '.format(i, j))
            print(printer.doprint(D[i, j]))



if __name__ == "__main__":
    main()
