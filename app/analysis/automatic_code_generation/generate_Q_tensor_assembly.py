import argparse
import enum

import sympy as sy

from ..utilities import tensor_calculus as tc
from ..utilities import dealii_code_generation as dcg
from . import Q_tensor_discretization_equations as qtde

class Basis(enum.Enum):
    component_wise = enum.auto()
    orthogonal = enum.auto()

class FieldTheory(enum.Enum):
    singular_potential = enum.auto()
    landau_de_gennes = enum.auto()

class Discretization(enum.Enum):
    convex_splitting = enum.auto()
    semi_implicit = enum.auto()
    newton_method = enum.auto()

def get_commandline_args():

    descrption = ('Runs symbolic calculation of each of the terms in the '
                  'nematic equation of motion and generates code necessary '
                  'for matrix assembly in the finite element code')
    parser = argparse.ArgumentParser(description=descrption)
    parser.add_argument('--basis',
                        choices={'component_wise', 'orthogonal'},
                        help='name of traceless, symmetric tensor basis to use')
    parser.add_argument('--field_theory',
                        choices={'singular_potential', 'landau_de_gennes'},
                        help='name of field theory to use')
    parser.add_argument('--discretization',
                        choices={'convex_splitting',
                                 'semi_implicit',
                                 'newton_method'},
                        help='name of discretization scheme to use')
    args = parser.parse_args()

    return Basis[args.basis], FieldTheory[args.field_theory], Discretization[args.discretization]

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

def get_discretization_expressions(field_theory, discretization,
                                   Phi_i, Phi_j, xi, Q, Q0, Lambda, Lambda0, dLambda, alpha, L2, L3, dt, theta):
    residual = None
    jacobian = None
    if (field_theory == FieldTheory.singular_potential
        and discretization == Discretization.convex_splitting):
        residual = qtde.calc_singular_potential_convex_splitting_residual(Phi_i, Q, Q0, Lambda, xi, alpha, dt, L2, L3)
        jacobian = qtde.calc_singular_potential_convex_splitting_jacobian(Phi_i, Phi_j, Q, xi, dLambda, alpha, dt, L2, L3)

    elif (field_theory == FieldTheory.singular_potential
          and discretization == Discretization.semi_implicit):
        residual = qtde.calc_singular_potential_semi_implicit_residual(Phi_i, xi, Q, Q0, Lambda, Lambda0, alpha, L2, L3, dt, theta)
        jacobian = qtde.calc_singular_potential_semi_implicit_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3, dt, theta)

    elif (field_theory == FieldTheory.singular_potential
          and discretization == Discretization.newton_method):
        residual = qtde.calc_singular_potential_newton_method_residual(Phi_i, xi, Q, Lambda, alpha, L2, L3)
        jacobian = qtde.calc_singular_potential_newton_method_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3)

    else:
        raise NotImplemented('Have only implemented singular potential discretizations')

    return residual, jacobian

def main():

    basis_enum, field_theory, discretization = get_commandline_args()

    vec_dim = 5
    
    x, y, z = sy.symbols('x y z')
    coords = (x, y)
    # coords = (x, y, z) # uncomment for 3D
    xi = tc.TensorCalculusArray([x, y, z])
    
    Z = sy.symbols(r'Z')   
    A, B, C = sy.symbols(r'A B C')
    
    Q_vec = tc.make_function_vector(vec_dim, 'Q_{}', coords)
    Q0_vec = tc.make_function_vector(vec_dim, 'Q_{{0{} }}', coords)
    Lambda_vec = tc.make_function_vector(vec_dim, r'\Lambda_{}', coords)
    Lambda0_vec = tc.make_function_vector(vec_dim, r'\Lambda_{{0{} }}', coords)
    
    phi_i = sy.Function(r'\phi_i')(*coords)
    phi_j = sy.Function(r'\phi_j')(*coords)

    basis = get_3x3_traceless_symmetric_tensor_basis(basis_enum)

    Q = tc.make_tensor_from_vector(Q_vec, basis)
    Q0 = tc.make_tensor_from_vector(Q0_vec, basis)
    Lambda = tc.make_tensor_from_vector(Lambda_vec, basis)
    Lambda0 = tc.make_tensor_from_vector(Lambda0_vec, basis)
    Phi_i = tc.make_basis_functions(phi_i, basis)
    Phi_j = tc.make_basis_functions(phi_j, basis)

    dLambda_label = r'\frac{{\partial\ \Lambda_{} }}{{\partial\ Q_{} }}'
    dLambda_mat = tc.make_function_matrix(vec_dim, dLambda_label, coords)
    dLambda = tc.make_jacobian_matrix_list(dLambda_mat, Phi_j)

    alpha, L2, L3, dt, theta = sy.symbols(r'\alpha L_2 L_3 \delta\ t theta')

    residual, jacobian = get_discretization_expressions(field_theory, discretization,
                                                        Phi_i, Phi_j, xi, Q, Q0, Lambda, Lambda0, 
                                                        dLambda, alpha, L2, L3, dt, theta)

    symbols_code = {alpha: 'alpha', L2: 'L2', L3: 'L3', dt: 'dt', theta: 'theta'}
    Q_code = {Q_vec[i]: 'Q_vec[q][{}]'.format(i) 
              for i in range(Q_vec.shape[0])}
    Q0_code = {Q0_vec[i]: 'Q0_vec[q][{}]'.format(i) 
               for i in range(Q0_vec.shape[0])}
    Lambda_code = {Lambda_vec[i]: 'Lambda_vec[q][{}]'.format(i) 
                   for i in range(Lambda_vec.shape[0])}
    Lambda0_code = {Lambda0_vec[i]: 'Lambda0_vec[q][{}]'.format(i) 
                    for i in range(Lambda0_vec.shape[0])}
    dQ_code = {Q_vec[i].diff(xi[j]): 'dQ[q][{}][{}]'.format(i, j)
               for i in range(vec_dim)
               for j in range(3)}
    dLambda_code = {dLambda_mat[i, j]: 'dLambda_dQ[{}][{}]'.format(i, j)
                    for i in range(dLambda_mat.shape[0])
                    for j in range(dLambda_mat.shape[1])}
    phi_i_code = {phi_i: 'fe_values.shape_value(i, q)'}
    phi_j_code = {phi_j: 'fe_values.shape_value(j, q)'}

    user_funcs = symbols_code | Q_code | Q0_code | Lambda_code | Lambda0_code | dQ_code | phi_i_code | phi_j_code | dLambda_code

    printer = dcg.MyPrinter(user_funcs)
    for i in range(vec_dim):
        print( printer.doprint(residual[-1][i]) )

    sy.printing.preview(residual[-1])

    # # for term in residual_terms:
    # #     sy.pprint(term)

    # # for term in jacobian_terms:
    # #     sy.pprint(term)


if __name__ == "__main__":
    main()
