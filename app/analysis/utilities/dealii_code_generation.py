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

    def __init__(self, user_symbols={}): 
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
        self.user_symbols = user_symbols



    def _print_Symbol(self, symbol):

        if symbol in self.user_symbols:
            return self._print(self.user_symbols[symbol])

        return super()._print_Symbol(symbol)
    


    def _print_Function(self, function):

        if function in self.user_symbols:
            return self._print(self.user_symbols[function])
        
        return super()._print_Function(function)



    def _print_Derivative(self, derivative):

        if derivative in self.user_symbols:
            return self._print(self.user_symbols[derivative])

        return super()._print_Derivative(derivative)
        


    def _print_Pow(self, Pow):
        
        if Pow.args[1] == 2:
            return "({}) * ({})".format(self.doprint(Pow.args[0]), self.doprint(Pow.args[0]))
        elif Pow.args[1] == 3:
            return "({}) * ({}) * ({})".format(self.doprint(Pow.args[0]), 
                                               self.doprint(Pow.args[0]), 
                                               self.doprint(Pow.args[0]))
        elif Pow.args[1] == 4:
            return "({}) * ({}) * ({}) * ({})".format(self.doprint(Pow.args[0]), 
                                                      self.doprint(Pow.args[0]), 
                                                      self.doprint(Pow.args[0]),
                                                      self.doprint(Pow.args[0]))
        return super()._print_Pow(Pow)
