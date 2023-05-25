import sympy as sy
from sympy.printing.cxx import CXX11CodePrinter


def flatten(l):
    """
    Given a nested list of lists, gives back an iterator which gives tuples
    whose first argument is the entry, and whose second element is a 
    variable-length tuple that indexes the nested list for the element.
    See source: https://stackoverflow.com/a/49012189/7736506
    """
    stack = [enumerate(l)]
    path = [None]
    while stack:
        for path[-1], x in stack[-1]:
            if isinstance(x, list):
                stack.append(enumerate(x))
                path.append(None)
            else:
                yield x, tuple(path)
            break
        else:
            stack.pop()
            path.pop()

class MyPrinter(CXX11CodePrinter):
    """
    This printer prints expressions as C++ code, and appropriately formats
    symbols that are specific to the nematic Q-tensor finite element simulation
    """

    def __init__(self, 
                 symbol_list=[],
                 function_list=[], 
                 derivative_list=[]): 
        """
        Parameters
        ----------
        basis_function_list : list of 2-component tuples
            Each tuple corresponds to a basis function with the first component
            being the corresponding sympy Function, and the second being a
            string which is the C++ code for the basis function.
        basis_derivative_list : list of 2-component tuples
            Each tuple corresponds to the gradient of a basis function with
            the first component a list of derivatives of the basis function
            with respect to each component, and the second component a string
            that can be formatted using the list index.
        symbol_list : list of 2-component tuples
            Each tuple corresponds to a symbol, with the first component being
            the sympy symbol and the second being the string which in C++
            corresponds to the symbol.
        function_list : list of 2-component tuples
            Each tuple corresponds to a function, with the first component an
            arbitrarily nested list of sympy Functions, and the second component
            a string that is formattable by their indices.
        """
        super().__init__()

        self.symbol_list = symbol_list
        self.function_list = function_list
        self.derivative_list = derivative_list



    def _print_Symbol(self, symbol):

        if symbol in self.symbol_list[0]:
            idx = self.symbol_list.index(symbol)
            return self._print(self.symbol_list[1][idx])

        return super()._print_Symbol(symbol)
    


    def _print_Function(self, function):

        for func_group, code_string in self.function_list:
            func, coords = zip(*flatten(func_group))
            if function in func:
                idx = func.index(function)
                return self._print(code_string.format(*coords[idx]))
        
        return super()._print_Function(function)



    def _print_Derivative(self, derivative):

        for deriv_group, code_string in self.derivative_list:
            deriv, coords = zip(*flatten(deriv_group))
            if derivative in deriv:
                idx = deriv.index(derivative)
                return self._print(code_string.format(*coords[idx]))

        return super()._print_Derivative(derivative)
        


    def _print_Pow(self, Pow):
        
        if Pow.args[1] == 2:
            return "({}) * ({})".format(self.doprint(Pow.args[0]), self.doprint(Pow.args[0]))
        return super()._print_Pow(Pow)
