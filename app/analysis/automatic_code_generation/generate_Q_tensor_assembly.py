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
    vec = sy.zeros(dim, 1)
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


def main():

    basis_enum = get_commandline_args()

    vec_dim = 5
    mat_dim = 3
    
    x, y, z = sy.symbols('x y z')
    coords = (x, y)
    # coords = (x, y, z) # uncomment for 3D
    xi = tc.TensorCalculusArray([x, y, z])
    
    alpha, dt, L2, L3, Z, theta = sy.symbols(r'\alpha \delta\ t L_2 L_3 Z theta')
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

    sy.pprint(Q)
    sy.pprint(Q0)
    sy.pprint(Lambda)
    sy.pprint(Lambda0)
    sy.pprint(delta_Q)
    for i in range(vec_dim):
        sy.pprint(Phi_i[i])
    for i in range(vec_dim):
        sy.pprint(Phi_j[i])

    sy.pprint(dLambda)


if __name__ == "__main__":
    main()
