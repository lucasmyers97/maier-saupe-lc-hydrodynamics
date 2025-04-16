"""
Module provides the ability to calculate the singular potential in limited
cases.
"""

from scipy.integrate import lebedev_rule

class SingularPotential:
    def __init__(self, Q, L, order, tol, max_iters):
        """
        Q: 2 x n numpy array holding Q0 and Q1 for n Q-values
        L: 2 x n numpy array holding Lambda0 and Lambda1 for n Q-values
        order: Lebedev quadrature order
        tol: Newton-Raphson tolerance
        max_iters: maximum Newton-Raphson iterations
        """
        self.Q = Q
        self.L = L
        self.order = order
        self.tol = tol
        self.max_iters = max_iters

    
    def calc_residual():

        exp = 
