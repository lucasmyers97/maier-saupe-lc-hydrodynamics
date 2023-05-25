import sympy as sy

class TensorCalculusArray(sy.MutableDenseNDimArray):
    
    def __mul__(self, other):
        try:
            return super().__mul__(other)
        except ValueError:
            if not isinstance(other, TensorCalculusArray):
                raise ValueError("Need TensorCalculusArray for contraction")
            
            rank_self = self.rank()
            prod = sy.tensorproduct(self, other)
            contraction = sy.tensorcontraction(prod, (rank_self - 1, rank_self))
            
            if type(contraction) == sy.Array:
                return TensorCalculusArray(contraction)
            else:
                return contraction
    
    def __matmul__(self, other):
        try:
            return super().__mul__(other)
        except ValueError:
            if not isinstance(other, TensorCalculusArray):
                raise ValueError("Need TensorCalculusArray for tensor product")

            return TensorCalculusArray( sy.tensorproduct(self, other) )
    
    def __pow__(self, other):
        try:
            return super().__mul__(other)
        except ValueError:
            if not isinstance(other, TensorCalculusArray):
                raise ValueError("Need TensorCalculusArray for double contraction")

            rank_self = self.rank()
            prod = sy.tensorproduct(self, other)
            contract = sy.tensorcontraction(prod, (rank_self - 1, rank_self))
            contraction = sy.tensorcontraction(contract, (rank_self - 2, rank_self - 1))
            
            if type(contraction) == sy.Array:
                return TensorCalculusArray(contraction)
            else:
                return contraction
        
    def ip(self, other):
        if not isinstance(other, TensorCalculusArray):
            raise ValueError("Need TensorCalculusArray for inner product")
        
        rank_self = self.rank()
        prod = sy.tensorproduct(self, other)
        while rank_self > 0:
            prod = sy.tensorcontraction(prod, (0, rank_self))
            rank_self -= 1
            
        return prod



def grad(M, xi):
    return TensorCalculusArray(sy.derive_by_array(M, xi))



def div(M, xi):
    return sy.tensorcontraction(grad(M, xi), (0, 1))



def transpose_3(M):
    return sy.permutedims(M, (1, 2, 0))



def make_function_vector(dim, label, coords):
    """
    Makes tensor_calculus.TensorCalculusArray of dimension dim.
    Each of the elements are a function of coords and the symbol name for the
    function is `label.format(i)` where `i` is the element of the vector.

    Parameters
    ----------
    dim : int
        dimension of the vector
    label : formattable string
        Symbol label for each of the vector entries is produced by calling
        `label.format(i)` where `i` is the vector index.
    coords : tuple
        Symbols which the vector entries are a function of (e.g. (x, y))

    Returns
    -------
    vec : tensor_calculus.TensorCalculusArray
        Vector whose elements are functions of coords.
    """

    vec = TensorCalculusArray.zeros(dim)
    for i in range(dim):
        vec[i] = sy.Function(label.format(i))(*coords)

    return vec



def make_function_matrix(dim, label, coords):
    """
    Makes tensor_calculus.TensorCalculusArray of dimension dim x dim.
    Each of the elements are a function of coords and the symbol name for the
    function is `label.format(i, j)` where `i` and `j` are the element of the 
    matrix.

    Parameters
    ----------
    dim : int
        dimension of the matrix
    label : formattable string
        Symbol label for each of the vector entries is produced by calling
        `label.format(i, j)` where `i` and `j` are the matrix indices.
    coords : tuple
        Symbols which the vector entries are a function of (e.g. (x, y))

    Returns
    -------
    mat : tensor_calculus.TensorCalculusArray
        Matrix whose elements are functions of coords.
    """

    mat = TensorCalculusArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            mat[i, j] = sy.Function(label.format(i, j))(*coords)

    return mat



def make_tensor_from_vector(vec, basis):
    """
    Given a vector whose elements represent degrees of freedom in some space
    and a list of basis elements, generate a tensor as: 
    tensor = sum_i vec_i * basis_i

    Parameters
    ----------
    vec : array-like
        List of symbols which are degrees of freedom in some space
    basis : array-like of tensor_calculus.TensorCalculusArray
        List of basis elements of some space. Should be same length as `vec`.

    Returns
    -------
    tensor : tensor_calculus.TensorCalculusArray
        Element in space which is represented by the elements of `vec` given
        the basis `basis`.
    """

    m, n = basis[0].shape
    vec_dim = vec.shape[0]

    tensor = TensorCalculusArray.zeros(m, n)
    for i in range(vec_dim):
        tensor += basis[i]*vec[i]

    return tensor



def make_basis_functions(function, basis):
    """
    Given a function and a list of basis objects (tensors, vectors, e)
    create a list which is the function multiplied by the basis elements.

    Parameters
    ----------
    function : sympy.Function
        Function of some coordinates
    basis : array-like of tensor_calculus.TensorCalculusArray
        List of elements representing basis of some space.

    Returns
    -------
    basis_functions : list of tensor_calculus.TensorCalculusArray
        List of basis elements multiplied by `function`
    """

    basis_functions = []
    for i in range(len(basis)):
        basis_functions.append( TensorCalculusArray(basis[i]*function) )

    return basis_functions



def make_jacobian_matrix_list(jacobian, Phi_j):
    """
    Given a `vec_dim`x`vec_dim` Jacobian matrix whose elements are the 
    derivatives of the degrees of freedom of some function with respect to the
    degrees of freedom of some underlying object as well as a set of basis 
    elements of the corresponding space which are also functions of the spatial
    coordinates, generate a Jacobian in the space with respect to the
    underlying degrees of freedom.

    Parameters
    ----------
    jacobian : tensor_calculus.TensorCalculusArray
        `vec_dim`x`vec_dim` TensorCalculusArray whose elements correspond to
        the derivatives of the degrees of freedom of some function with respect
        to the degrees of freedom of the field that it's a function of.
    Phi_j : array-like of tensor_calculus.TensorCalculusArray
        `vec_dim` length list of `mat_dim`x`mat_dim` TensorCalculusArray
        objects.
        Elements in TensorCalculusArrays should be scalar test functions of 
        the spatial coordinates.

    Returns
    -------
    jacobian_matrix_list : array-like of tensor_calculus.TensorCalculusArray
        `vec_dim` length list of `mat_dim`x`mat_dim` TensorCalculusArray
        objects.
        Will be sum_k jacobian[k, j] * Phi_j[k].
    """

    vec_dim = jacobian.shape[0]
    mat_dim = Phi_j[0].shape[0]

    jacobian_matrix_list = []
    for j in range(vec_dim):
        
        jacobian_matrix = TensorCalculusArray.zeros(mat_dim, mat_dim)
        for k in range(vec_dim):
            jacobian_matrix += jacobian[k, j] * Phi_j[k]

        jacobian_matrix_list.append(jacobian_matrix)

    return jacobian_matrix_list