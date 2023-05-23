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

x, y, z = sy.symbols('x y z')
xi = TensorCalculusArray([x, y, z])
    
def grad(M):
    return TensorCalculusArray(sy.derive_by_array(M, xi))

def div(M):
    return sy.tensorcontraction(grad(M), (0, 1))

def transpose_3(M):
    return sy.permutedims(M, (1, 2, 0))
