import enum

import sympy as sy

from . import tensor_calculus as tc


class QTensorMath:


    def __init__(self, space_dim, basis_enum):

        vec_dim = 5
        self.domain = self.Domain(space_dim, vec_dim)
        self.basis = self.get_3x3_traceless_symmetric_tensor_basis(basis_enum)
        self.fe_fields = self.FEFields(self.domain)
        self.tensors = self.Tensors(self.fe_fields, self.basis)
        self.scalars = self.Scalars()
        self.parameters = self.Parameters()
        self.code_symbols = self.set_up_code_symbols(self.domain, 
                                                     self.fe_fields, 
                                                     self.scalars, 
                                                     self.parameters)

    class Basis(enum.Enum):

        component_wise = enum.auto()
        orthogonal = enum.auto()

    class Domain():

        def __init__(self, space_dim, vec_dim):

            if not (space_dim == 2 or space_dim == 3):
                raise ValueError('Space dimension must be 2 or 3')

            self.vec_dim = vec_dim
            self.space_dim = space_dim

            x, y, z = sy.symbols('x y z')
            self.coords = (x, y) if space_dim == 2 else (x, y, z)
            x, y, z = (x, y, z)
            self.xi = tc.TensorCalculusArray([x, y, z])
            self.nu = tc.make_vector(space_dim, r'nu_{}')

    class FEFields:

        def __init__(self, domain):

            self.Q_vec = tc.make_function_vector(domain.vec_dim, 'Q_{}', domain.coords)
            self.Q0_vec = tc.make_function_vector(domain.vec_dim, 'Q_{{0{}}}', domain.coords)
            self.Lambda_vec = tc.make_function_vector(domain.vec_dim, r'\Lambda_{}', domain.coords)
            self.Lambda0_vec = tc.make_function_vector(domain.vec_dim, r'\Lambda_{{0{} }}', domain.coords)

            dLambda_label = r'\frac{{\partial\ \Lambda_{} }}{{\partial\ Q_{} }}'
            self.dLambda_mat = tc.make_function_matrix(domain.vec_dim, dLambda_label, domain.coords)

            self.phi_i = sy.Function(r'\phi_i')(*domain.coords)
            self.phi_j = sy.Function(r'\phi_j')(*domain.coords)

    class Tensors:

        def __init__(self, fe_fields, basis):

            self.Q = tc.make_tensor_from_vector(fe_fields.Q_vec, basis)
            self.Q0 = tc.make_tensor_from_vector(fe_fields.Q0_vec, basis)
            self.Lambda = tc.make_tensor_from_vector(fe_fields.Lambda_vec, basis)
            self.Lambda0 = tc.make_tensor_from_vector(fe_fields.Lambda0_vec, basis)
            self.Phi_i = tc.make_basis_functions(fe_fields.phi_i, basis)
            self.Phi_j = tc.make_basis_functions(fe_fields.phi_j, basis)
    
            self.dLambda = tc.make_jacobian_matrix_list(fe_fields.dLambda_mat, self.Phi_j)

            self.A = tc.TensorCalculusArray([[0, -1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]])
            self.P = tc.TensorCalculusArray([[1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 0]])
            self.I = tc.TensorCalculusArray(sy.eye(3))

    class Scalars:

        def __init__(self):

            self.Z = sy.symbols(r'Z')   

    class Parameters:

        def __init__(self):

            symbols_string = r'A B C \alpha L_2 L_3 \delta\ t theta S_0 W_1 W_2 \omega'

            (self.A, self.B, self.C, 
             self.alpha, self.L2, self.L3, 
             self.dt, self.theta, 
             self.S0, self.W1, self.W2, 
             self.omega) = sy.symbols(symbols_string, real=True)

    def get_3x3_traceless_symmetric_tensor_basis(self, basis_enum):
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
        if basis_enum == self.Basis.component_wise:
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
        elif basis_enum == self.Basis.orthogonal:
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



    def set_up_code_symbols(self, domain, fe_fields, scalars, parameters):
    
        symbols_code = {scalars.Z: 'Z',
                        parameters.alpha: 'alpha', 
                        parameters.L2: 'L2', 
                        parameters.L3: 'L3', 
                        parameters.A: 'A', 
                        parameters.B: 'B',
                        parameters.C: 'C',
                        parameters.dt: 'dt',
                        parameters.theta: 'theta',
                        parameters.S0: 'S0',
                        parameters.W1: 'W1',
                        parameters.W2: 'W2',
                        parameters.omega: 'omega'}

        Q_code = {fe_fields.Q_vec[i]: 'Q_vec[q][{}]'.format(i) 
                  for i in range(fe_fields.Q_vec.shape[0])}
        Q0_code = {fe_fields.Q0_vec[i]: 'Q0_vec[q][{}]'.format(i) 
                   for i in range(fe_fields.Q0_vec.shape[0])}
        Lambda_code = {fe_fields.Lambda_vec[i]: 'Lambda_vec[{}]'.format(i) 
                       for i in range(fe_fields.Lambda_vec.shape[0])}
        Lambda0_code = {fe_fields.Lambda0_vec[i]: 'Lambda0_vec[{}]'.format(i) 
                        for i in range(fe_fields.Lambda0_vec.shape[0])}

        dQ_code = {fe_fields.Q_vec[i].diff(domain.xi[j]): 'dQ[q][{}][{}]'.format(i, j)
                   for i in range(fe_fields.Q_vec.shape[0])
                   for j in range(domain.xi.shape[0])}
        dQ0_code = {fe_fields.Q0_vec[i].diff(domain.xi[j]): 'dQ0[q][{}][{}]'.format(i, j)
                    for i in range(fe_fields.Q0_vec.shape[0])
                    for j in range(domain.xi.shape[0])}
        dLambda_code = {fe_fields.dLambda_mat[i, j]: 'dLambda_dQ[{}][{}]'.format(i, j)
                        for i in range(fe_fields.dLambda_mat.shape[0])
                        for j in range(fe_fields.dLambda_mat.shape[1])}

        phi_i_code = {fe_fields.phi_i: 'fe_values.shape_value(i, q)'}
        phi_j_code = {fe_fields.phi_j: 'fe_values.shape_value(j, q)'}
        dphi_i_code = {fe_fields.phi_i.diff(domain.xi[k]): 'fe_values.shape_grad(i, q)[{}]'.format(k)
                       for k in range(domain.xi.shape[0])}
        dphi_j_code = {fe_fields.phi_j.diff(domain.xi[k]): 'fe_values.shape_grad(j, q)[{}]'.format(k)
                       for k in range(domain.xi.shape[0])}
        nu_code = {domain.nu[k]: 'fe_values.normal_vector(q)[{}]'.format(k)
                   for k in range(domain.nu.shape[0])}
    
        return (symbols_code | Q_code | Q0_code | Lambda_code | Lambda0_code 
                | dQ_code | dQ0_code |phi_i_code | phi_j_code | dphi_i_code | dphi_j_code 
                | dLambda_code | nu_code)

    def ST(self, M):
    
        return sy.Rational(1, 2) * (M + M.transpose()) - sy.Rational(1, 3) * M.trace() * self.tensors.I
