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



def get_basis(basis_enum):

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



def make_function_array(dim, label, coords):
    vec = tc.TensorCalculusArray.zeros(dim)
    for i in range(dim):
        vec[i] = sy.Function(label.format(i))(*coords)

    return vec



def make_function_matrix(dim, label, coords):
    mat = tc.TensorCalculusArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            mat[i, j] = sy.Function(label.format(i, j))(*coords)

    return mat



def make_tensor_from_vector(vec, basis):

    m, n = basis[0].shape
    vec_dim = vec.shape[0]

    tensor = tc.TensorCalculusArray.zeros(m, n)
    for i in range(vec_dim):
        tensor += basis[i]*vec[i]

    return tensor



def make_basis_tensors(function, basis):

    basis_tensors = []
    for i in range(len(basis)):
        basis_tensors.append( tc.TensorCalculusArray(basis[i]*function) )

    return basis_tensors



def make_tensor_from_matrix(mat, basis):

    vec_dim = mat.shape[0]
    mat_dim = basis[0].shape[0]

    tensor = tc.TensorCalculusArray.zeros(vec_dim, mat_dim, mat_dim)
    for i in range(vec_dim):
        for j in range(vec_dim):
            tensor[j, :, :] += mat[i, j] * basis[i]

    return tensor



def calc_singular_potential_convex_splitting_residual(Phi_i, Q, Q0, Lambda, alpha, dt, L2, L3):

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
        E1[i, 0] = -tc.grad(Phi_i[i]).ip(tc.grad(Q))
        E2[i, 0] = -tc.transpose_3( tc.grad(Phi_i[i]) ).ip(tc.grad(Q))
        E31[i, 0] = -tc.grad(Phi_i[i]).ip(Q * tc.grad(Q))
        E32[i, 0] = -sy.Rational(1, 2) * Phi_i[i].ip(tc.grad(Q) ** tc.transpose_3( tc.grad(Q) ))

    R1 = sy.simplify(R1)
    R2 = sy.simplify(R2)
    R3 = sy.simplify(R3)
    E1 = sy.simplify(E1)
    E2 = sy.simplify(E2)
    E31 = sy.simplify(E31)
    E32 = sy.simplify(E32)

    return R1, R2, R3, -dt*E1, -dt*L2*E2, -dt*L3*(E31 + E32)



def calc_singular_potential_convex_splitting_jacobian(Phi_i, Phi_j, phi_j, Q, dLambda, alpha, dt, L2, L3):

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
            dR2[i, j] = dt * Phi_i[i].ip(dLambda[j, :, :] * phi_j)
            dE1[i, j] = -tc.grad(Phi_i[i]).ip(tc.grad(Phi_j[j]))
            dE2[i, j] = -tc.transpose_3( tc.grad(Phi_i[i]) ).ip( tc.grad(Phi_j[j]) )
            dE31[i, j] = -tc.grad(Phi_i[i]).ip(Phi_j[j]*tc.grad(Q) + Q*tc.grad(Phi_j[j]))
            dE32[i, j] = -Phi_i[i].ip(tc.grad(Phi_j[j]) ** tc.transpose_3( tc.grad(Q) ))

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
    mat_dim = 3
    
    x, y, z = sy.symbols('x y z')
    coords = (x, y)
    # coords = (x, y, z) # uncomment for 3D
    xi = tc.TensorCalculusArray([x, y, z])
    
    Z, theta = sy.symbols(r'Z theta')   
    A, B, C = sy.symbols(r'A B C')
    
    Q_vec = make_function_array(vec_dim, 'Q_{}', coords)
    Q0_vec = make_function_array(vec_dim, 'Q_{{0{} }}', coords)
    Lambda_vec = make_function_array(vec_dim, r'\Lambda_{}', coords)
    Lambda0_vec = make_function_array(vec_dim, r'\Lambda_{{0{} }}', coords)
    delta_Q_vec = make_function_array(vec_dim, r'\delta\ Q_{}', coords)
    
    phi_i = sy.Function(r'\phi_i')(*coords)
    phi_j = sy.Function(r'\phi_j')(*coords)

    basis = get_basis(basis_enum)

    Q = make_tensor_from_vector(Q_vec, basis)
    Q0 = make_tensor_from_vector(Q0_vec, basis)
    Lambda = make_tensor_from_vector(Lambda_vec, basis)
    Lambda0 = make_tensor_from_vector(Lambda0_vec, basis)
    delta_Q = make_tensor_from_vector(delta_Q_vec, basis)
    Phi_i = make_basis_tensors(phi_i, basis)
    Phi_j = make_basis_tensors(phi_j, basis)

    dLambda_label = r'\frac{{\partial\ \Lambda_{} }}{{\partial\ Q_{} }}'
    dLambda_mat = make_function_matrix(vec_dim, dLambda_label, coords)
    dLambda = make_tensor_from_matrix(dLambda_mat, basis)

    singular_potential_symbols = sy.symbols(r'\alpha \delta\ t L_2 L_3')
    residual_terms = calc_singular_potential_convex_splitting_residual(Phi_i, 
                                                                       Q, 
                                                                       Q0, 
                                                                       Lambda, 
                                                                       *singular_potential_symbols)
    jacobian_terms = calc_singular_potential_convex_splitting_jacobian(Phi_i, 
                                                                       Phi_j, 
                                                                       phi_j, 
                                                                       Q, 
                                                                       dLambda, 
                                                                       *singular_potential_symbols)

    for term in residual_terms:
        sy.pprint(term)

    for term in jacobian_terms:
        sy.pprint(term)


if __name__ == "__main__":
    main()
