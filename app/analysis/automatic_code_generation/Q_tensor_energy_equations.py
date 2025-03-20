"""
Holds functions which symbolically produce the energy terms for different
field theories
"""

import sympy as sy
from ..utilities import tensor_calculus as tc

def calc_singular_potential_energy(Q, Lambda, xi, alpha, B, L2, L3, Z, omega):

    delta = tc.TensorCalculusArray( sy.eye(3) )

    E = -alpha / 2 * Q**Q
    cubic = B * Q**(Q*Q)
    TdS = sy.log(4 * sy.pi) - sy.log(Z) + Lambda ** (Q + sy.Rational(1, 3) * delta)

    z = xi[-1]
    R = tc.TensorCalculusArray([[sy.cos(omega * z), -sy.sin(omega * z), 0],
                                [sy.sin(omega * z), sy.cos(omega * z), 0],
                                [0, 0, 1]])

    RT = R.transpose()

    EL1 = sy.Rational(1, 2) * tc.grad(R*Q*RT, xi).ip( tc.grad(R*Q*RT, xi) ).subs(z, 0)
    EL2 = sy.Rational(1, 2) * L2 * ( tc.div(R*Q*RT, xi) * tc.div(R*Q*RT, xi) ).subs(z, 0)
    EL3 = sy.Rational(1, 2) * L3 * ( (R*Q*RT) ** ( tc.grad(R*Q*RT, xi) ** tc.transpose_3(tc.grad(R*Q*RT, xi)) ) ).subs(z, 0)

    E = sy.simplify(E)
    cubic = sy.simplify(cubic)
    TdS = sy.simplify(TdS)
    EL1 = sy.simplify(EL1)
    EL2 = sy.simplify(EL2)
    EL3 = sy.simplify(EL3)

    return E, cubic, TdS, EL1, EL2, EL3

def calc_singular_potential_surface_energy(Q, Lambda, xi, alpha, L2, L3, Z):

    raise NotImplementedError
