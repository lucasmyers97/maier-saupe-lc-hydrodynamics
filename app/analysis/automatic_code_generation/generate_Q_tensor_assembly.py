import argparse
import enum

import sympy as sy

from ..utilities import tensor_calculus as tc
from ..utilities import dealii_code_generation as dcg

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



def calc_singular_potential_convex_splitting_residual(Phi_i, Q, Q0, Lambda, xi, alpha, dt, L2, L3):

    vec_dim = len(Phi_i)
    R1 = sy.zeros(vec_dim, 1)
    R2 = sy.zeros(vec_dim, 1)
    R3 = sy.zeros(vec_dim, 1)
    E1 = sy.zeros(vec_dim, 1)
    E2 = sy.zeros(vec_dim, 1)
    E31 = sy.zeros(vec_dim, 1)
    E32 = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        R1[i, 0] = Phi_i[i].ip(Q)
        R2[i, 0] = -(1 + alpha * dt) * Phi_i[i].ip(Q0)
        R3[i, 0] = -dt * (-Phi_i[i].ip(Lambda))
        E1[i, 0] = -tc.grad(Phi_i[i], xi).ip(tc.grad(Q, xi))
        E2[i, 0] = -tc.transpose_3( tc.grad(Phi_i[i], xi) ).ip(tc.grad(Q, xi))
        E31[i, 0] = -tc.grad(Phi_i[i], xi).ip(Q * tc.grad(Q, xi))
        E32[i, 0] = -sy.Rational(1, 2) * Phi_i[i].ip(tc.grad(Q, xi) ** tc.transpose_3( tc.grad(Q, xi) ))

    R1 = sy.simplify(R1)
    R2 = sy.simplify(R2)
    R3 = sy.simplify(R3)
    E1 = sy.simplify(E1)
    E2 = sy.simplify(E2)
    E31 = sy.simplify(E31)
    E32 = sy.simplify(E32)

    return R1, R2, R3, -dt*E1, -dt*L2*E2, -dt*L3*(E31 + E32)



def calc_singular_potential_convex_splitting_jacobian(Phi_i, Phi_j, Q, dLambda, xi, alpha, dt, L2, L3):

    vec_dim = len(Phi_i)
    dR1 = sy.zeros(vec_dim, vec_dim)
    dR2 = sy.zeros(vec_dim, vec_dim)
    dE1 = sy.zeros(vec_dim, vec_dim)
    dE2 = sy.zeros(vec_dim, vec_dim)
    dE31 = sy.zeros(vec_dim, vec_dim)
    dE32 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dR1[i, j] = Phi_i[i].ip(Phi_j[j])
            dR2[i, j] = dt * Phi_i[i].ip(dLambda[j])
            dE1[i, j] = -tc.grad(Phi_i[i], xi).ip(tc.grad(Phi_j[j], xi))
            dE2[i, j] = -tc.transpose_3( tc.grad(Phi_i[i], xi) ).ip( tc.grad(Phi_j[j], xi) )
            dE31[i, j] = -tc.grad(Phi_i[i], xi).ip(Phi_j[j]*tc.grad(Q, xi) + Q*tc.grad(Phi_j[j], xi))
            dE32[i, j] = -Phi_i[i].ip(tc.grad(Phi_j[j], xi) ** tc.transpose_3( tc.grad(Q, xi) ))

    dR1 = sy.simplify(dR1) 
    dR2 = sy.simplify(dR2) 
    dE1 = sy.simplify(dE1) 
    dE2 = sy.simplify(dE2) 
    dE31 = sy.simplify(dE31)
    dE32 = sy.simplify(dE32)

    return dR1, dR2, -dt*dE1, -dt*L2*dE2, -dt*L3*(dE31 + dE32)



def main():

    basis_enum = get_commandline_args()

    vec_dim = 5
    
    x, y, z = sy.symbols('x y z')
    coords = (x, y)
    # coords = (x, y, z) # uncomment for 3D
    xi = tc.TensorCalculusArray([x, y, z])
    
    Z, theta = sy.symbols(r'Z theta')   
    A, B, C = sy.symbols(r'A B C')
    
    Q_vec = tc.make_function_vector(vec_dim, 'Q_{}', coords)
    Q0_vec = tc.make_function_vector(vec_dim, 'Q_{{0{} }}', coords)
    Lambda_vec = tc.make_function_vector(vec_dim, r'\Lambda_{}', coords)
    Lambda0_vec = tc.make_function_vector(vec_dim, r'\Lambda_{{0{} }}', coords)
    delta_Q_vec = tc.make_function_vector(vec_dim, r'\delta\ Q_{}', coords)
    
    phi_i = sy.Function(r'\phi_i')(*coords)
    phi_j = sy.Function(r'\phi_j')(*coords)

    basis = get_3x3_traceless_symmetric_tensor_basis(basis_enum)

    Q = tc.make_tensor_from_vector(Q_vec, basis)
    Q0 = tc.make_tensor_from_vector(Q0_vec, basis)
    Lambda = tc.make_tensor_from_vector(Lambda_vec, basis)
    Lambda0 = tc.make_tensor_from_vector(Lambda0_vec, basis)
    delta_Q = tc.make_tensor_from_vector(delta_Q_vec, basis)
    Phi_i = tc.make_basis_functions(phi_i, basis)
    Phi_j = tc.make_basis_functions(phi_j, basis)

    dLambda_label = r'\frac{{\partial\ \Lambda_{} }}{{\partial\ Q_{} }}'
    dLambda_mat = tc.make_function_matrix(vec_dim, dLambda_label, coords)
    dLambda = tc.make_jacobian_matrix_list(dLambda_mat, Phi_j)

    singular_potential_symbols = sy.symbols(r'\alpha \delta\ t L_2 L_3')
    residual_terms = calc_singular_potential_convex_splitting_residual(Phi_i, 
                                                                       Q, 
                                                                       Q0, 
                                                                       Lambda, 
                                                                       xi,
                                                                       *singular_potential_symbols)
    jacobian_terms = calc_singular_potential_convex_splitting_jacobian(Phi_i, 
                                                                       Phi_j, 
                                                                       Q, 
                                                                       dLambda, 
                                                                       xi,
                                                                       *singular_potential_symbols)
    symbols_code = ['alpha', 'dt', 'L2', 'L3']
    symbols_list = [list(singular_potential_symbols),
                    symbols_code]

    Q_code = 'Q_vec[q][{}]'
    Q0_code = 'Q0_vec[q][{}]'
    Lambda_code = 'Lambda_vec[{}]'
    Lambda0_code = 'Lambda0_vec[{}]'
    dLambda_code = 'dLambda_dQ[{}][{}]'
    phi_i_code = 'fe_values.shape_value(i, q)'
    phi_j_code = 'fe_values.shape_value(j, q)'
    function_list = [(Q_vec.tolist(), Q_code),
                     (Q0_vec.tolist(), Q0_code),
                     (Lambda_vec.tolist(), Lambda_code),
                     (Lambda0_vec.tolist(), Lambda0_code),
                     (dLambda_mat.tolist(), dLambda_code),
                     ([phi_i], phi_i_code),
                     ([phi_j], phi_j_code)]

    printer = dcg.MyPrinter(function_list)
        
    print(printer.doprint(residual_terms[0][0]))

    # for term in residual_terms:
    #     sy.pprint(term)

    # for term in jacobian_terms:
    #     sy.pprint(term)


if __name__ == "__main__":
    main()
