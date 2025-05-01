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
    semi_implicit_rotated = enum.auto()
    newton_method = enum.auto()

class Domain(enum.Enum):
    bulk = enum.auto()
    surface = enum.auto()

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
                                 'semi_implicit_rotated',
                                 'newton_method'},
                        help='name of discretization scheme to use')
    parser.add_argument('--domain',
                        choices={'bulk',
                                 'surface'},
                        help='whether we are calculating the bulk or surface terms')
    parser.add_argument('--space_dim',
                        choices={2, 3},
                        type=int,
                        help='space dimension of discretization')
    args = parser.parse_args()

    return (Basis[args.basis], FieldTheory[args.field_theory], 
            Discretization[args.discretization], Domain[args.domain], args.space_dim)

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
    
    Z = sy.symbols(r'Z')   
    A, B, C = sy.symbols(r'A B C')
    alpha, L2, L3, zeta, dt, theta, S0, W1, W2, omega = sy.symbols(r'\alpha L_2 L_3 \zeta \delta\ t theta S_0 W_1 W_2 \omega')
    
    Q_vec = tc.make_function_vector(vec_dim, 'Q_{}', coords)
    Q0_vec = tc.make_function_vector(vec_dim, 'Q_{{0{}}}', coords)
    Lambda_vec = tc.make_function_vector(vec_dim, r'\Lambda_{}', coords)
    Lambda0_vec = tc.make_function_vector(vec_dim, r'\Lambda_{{0{} }}', coords)
    v = tc.make_function_vector(3, r'v_{}', coords)
    if space_dim == 2:
        v[-1] = 0
    
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

    # make unit vector nu
    nu = tc.make_vector(len(coords), r'nu_{}')
    # nu[-1] = sy.sqrt(1 - sum(nu[i] * nu[i] 
    #                          for i in range(len(coords) - 1)))

    return (xi, Q_vec, Q0_vec, Lambda_vec, Lambda0_vec, v, dLambda_mat, phi_i, phi_j, 
            Q, Q0, Lambda, Lambda0, dLambda, Phi_i, Phi_j,
            Z, A, B, C, alpha, L2, L3, zeta, dt, theta, S0, W1, W2, nu, omega)

def set_up_code_symbols(xi, Q_vec, Q0_vec, Lambda_vec, Lambda0_vec, v,
                        dLambda_mat, phi_i, phi_j,
                        Z, A, B, C, alpha, L2, L3, zeta, dt, theta, S0, W1, W2, nu, omega):

    symbols_code = {alpha: 'alpha', L2: 'L2', L3: 'L3', zeta: 'zeta',
                    Z: 'Z', A: 'A', B: 'B', C: 'C',
                    dt: 'dt', theta: 'theta', 
                    S0: 'S0', W1: 'W1', W2: 'W2', omega: 'omega'}
    Q_code = {Q_vec[i]: 'Q_vec[q][{}]'.format(i) 
              for i in range(Q_vec.shape[0])}
    Q0_code = {Q0_vec[i]: 'Q0_vec[q][{}]'.format(i) 
               for i in range(Q0_vec.shape[0])}
    Lambda_code = {Lambda_vec[i]: 'Lambda_vec[{}]'.format(i) 
                   for i in range(Lambda_vec.shape[0])}
    Lambda0_code = {Lambda0_vec[i]: 'Lambda0_vec[{}]'.format(i) 
                    for i in range(Lambda0_vec.shape[0])}
    v_code = {v[i]: 'v[q][{}]'.format(i) 
              for i in range(v.shape[0])
              if v[i] != 0}
    dQ_code = {Q_vec[i].diff(xi[j]): 'dQ[q][{}][{}]'.format(i, j)
               for i in range(Q_vec.shape[0])
               for j in range(xi.shape[0])}
    dQ0_code = {Q0_vec[i].diff(xi[j]): 'dQ0[q][{}][{}]'.format(i, j)
                for i in range(Q0_vec.shape[0])
                for j in range(xi.shape[0])}
    dLambda_code = {dLambda_mat[i, j]: 'dLambda_dQ[{}][{}]'.format(i, j)
                    for i in range(dLambda_mat.shape[0])
                    for j in range(dLambda_mat.shape[1])}
    dv_code = {v[i].diff(xi[j]): 'dv[q][{}][{}]'.format(i, j)
               for i in range(v.shape[0])
               for j in range(xi.shape[0])
               if v[i] != 0}
    phi_i_code = {phi_i: 'fe_values.shape_value(i, q)'}
    phi_j_code = {phi_j: 'fe_values.shape_value(j, q)'}
    dphi_i_code = {phi_i.diff(xi[k]): 'fe_values.shape_grad(i, q)[{}]'.format(k)
                   for k in range(xi.shape[0])}
    dphi_j_code = {phi_j.diff(xi[k]): 'fe_values.shape_grad(j, q)[{}]'.format(k)
                   for k in range(xi.shape[0])}
    nu_code = {nu[k]: 'fe_values.normal_vector(q)[{}]'.format(k)
               for k in range(nu.shape[0])}

    return (symbols_code | Q_code | Q0_code | Lambda_code | Lambda0_code | v_code
            | dQ_code | dQ0_code | dv_code |phi_i_code | phi_j_code | dphi_i_code | dphi_j_code 
            | dLambda_code | nu_code)


def get_discretization_expressions(field_theory, discretization, domain,
                                   Phi_i, Phi_j, xi, Q, Q0, Lambda, Lambda0, v, dLambda, 
                                   alpha, B, L2, L3, zeta, S0, W1, W2, nu, omega, dt, theta):
    residual = None
    jacobian = None
    if (field_theory == FieldTheory.singular_potential
        and discretization == Discretization.convex_splitting
        and domain == Domain.bulk):
        residual = qtde.calc_singular_potential_convex_splitting_residual(Phi_i, Q, Q0, Lambda, xi, alpha, dt, L2, L3)
        jacobian = qtde.calc_singular_potential_convex_splitting_jacobian(Phi_i, Phi_j, Q, xi, dLambda, alpha, dt, L2, L3)

    elif (field_theory == FieldTheory.singular_potential
          and discretization == Discretization.semi_implicit
          and domain == Domain.bulk):
        residual = qtde.calc_singular_potential_semi_implicit_residual(Phi_i, xi, Q, Q0, Lambda, Lambda0, v, 
                                                                       alpha, B, L2, L3, zeta, dt, theta)
        jacobian = qtde.calc_singular_potential_semi_implicit_jacobian(Phi_i, Phi_j, xi, Q, v, dLambda, 
                                                                       alpha, B, L2, L3, zeta, dt, theta)

    elif (field_theory == FieldTheory.singular_potential
          and discretization == Discretization.semi_implicit_rotated
          and domain == Domain.bulk):
        residual = qtde.calc_singular_potential_semi_implicit_rotated_residual(Phi_i, xi, Q, Q0, Lambda, Lambda0, alpha, L2, L3, omega, dt, theta)
        jacobian = qtde.calc_singular_potential_semi_implicit_rotated_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3, omega, dt, theta)
        # jacobian = None

    elif (field_theory == FieldTheory.singular_potential
          and discretization == Discretization.newton_method
          and domain == Domain.bulk):
        residual = qtde.calc_singular_potential_newton_method_residual(Phi_i, xi, Q, Lambda, alpha, L2, L3)
        jacobian = qtde.calc_singular_potential_newton_method_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3)

    elif (field_theory == FieldTheory.singular_potential
          and discretization == Discretization.semi_implicit
          and domain == Domain.surface):
        residual = qtde.calc_singular_potential_semi_implicit_surface_residual(Phi_i, Q, Q0, nu, S0, W1, W2, dt, theta)
        jacobian = qtde.calc_singular_potential_semi_implicit_surface_jacobian(Phi_i, Phi_j, Q, nu, S0, W1, W2, dt, theta)


    else:
        raise NotImplemented('Have only implemented singular potential discretizations')

    return residual, jacobian

def print_residual_code(printer, residual, vec_dim):

    indent = ' ' * 4

    first_residual_component = 'if (component_i == 0)\n'
    residual_component = 'else if (component_i == {})\n'
    residual_lhs = indent + 'cell_rhs(i) -= (\n' # because rhs is -residual
    residual_end = 2 * indent + ') * fe_values.JxW(q);\n'

    residual_code = ''
    first_component = True
    for i in range(vec_dim):

        if first_component:
            residual_code += first_residual_component + residual_lhs
            first_component = False
        else:
            residual_code += residual_component.format(i) + residual_lhs

        first_term = True
        for term in residual:
            if term[i] == 0:
                continue

            if first_term:
                residual_code += (2 * indent 
                                  + printer.doprint(term[i]) 
                                  + '\n')
                first_term = False
            else:
                residual_code += (2 * indent 
                                  + '+\n' 
                                  + 2 * indent
                                  + printer.doprint(term[i]) 
                                  + '\n')

        residual_code += residual_end

    return residual_code

def print_jacobian_code(printer, jacobian, vec_dim):

    indent = ' ' * 4

    first_jacobian_component = 'if (component_i == 0 && component_j == 0)\n'
    jacobian_component = 'else if (component_i == {} && component_j == {})\n'
    jacobian_lhs = indent + 'cell_matrix(i, j) += (\n'
    jacobian_end = 2 * indent + ') * fe_values.JxW(q);\n'

    jacobian_code = ''
    first_component = True
    for i in range(vec_dim):
        for j in range(vec_dim):

            if first_component:
                jacobian_code += first_jacobian_component + jacobian_lhs
                first_component = False
            else:
                jacobian_code += jacobian_component.format(i, j) + jacobian_lhs

            first_term = True
            for term in jacobian:
                if term[i, j] == 0:
                    continue

                if first_term:
                    jacobian_code += (2 * indent 
                                      + printer.doprint(term[i, j]) 
                                      + '\n')
                    first_term = False
                else:
                    jacobian_code += (2 * indent 
                                      + '+\n' 
                                      + 2 * indent
                                      + printer.doprint(term[i, j]) 
                                      + '\n')

            jacobian_code += jacobian_end

    return jacobian_code


def main():

    basis_enum, field_theory, discretization, domain, space_dim = get_commandline_args()
    vec_dim = 5

    (xi, Q_vec, Q0_vec, Lambda_vec, Lambda0_vec, v, dLambda_mat, phi_i, phi_j, 
     Q, Q0, Lambda, Lambda0, dLambda, Phi_i, Phi_j,
     Z, A, B, C, alpha, L2, L3, zeta, dt, theta, S0, W1, W2, nu, omega) = set_up_symbols(vec_dim, basis_enum, space_dim)

    residual, jacobian = get_discretization_expressions(field_theory, discretization, domain,
                                                        Phi_i, Phi_j, xi, Q, Q0, Lambda, Lambda0, v, 
                                                        dLambda, alpha, B, L2, L3, zeta, S0, W1, W2, nu, omega, dt, theta)

    user_funcs = set_up_code_symbols(xi, Q_vec, Q0_vec, Lambda_vec, Lambda0_vec, v,
                                     dLambda_mat, phi_i, phi_j,
                                     Z, A, B, C, alpha, L2, L3, zeta, dt, theta, S0, W1, W2, nu, omega)

    printer = dcg.MyPrinter(user_funcs)

    for term in residual:
        print(type(term))

    if residual:
        print(print_residual_code(printer, residual, vec_dim))
    if jacobian:
        print(print_jacobian_code(printer, jacobian, vec_dim))

if __name__ == "__main__":
    main()
