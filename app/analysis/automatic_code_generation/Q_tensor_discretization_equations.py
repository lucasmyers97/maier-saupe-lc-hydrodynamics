"""
Holds functions which symbolically produce the residual and jacobian terms for 
different discretization techniques.
"""

import sympy as sy
from ..utilities import tensor_calculus as tc

def E1(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E1 = sy.zeros(vec_dim)

    for i in range(vec_dim):
        E1[i] = -tc.grad(Phi_i[i], xi).ip( tc.grad(Q, xi) )

    return sy.simplify(E1)



def E2(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E2 = sy.zeros(vec_dim)

    for i in range(vec_dim):
        E2[i] = -tc.transpose_3( tc.grad(Phi_i[i], xi) ).ip( tc.grad(Q, xi) )

    return sy.simplify(E2)



def E31(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E31 = sy.zeros(vec_dim)

    for i in range(vec_dim):
        E31[i] = -tc.grad(Phi_i[i], xi).ip( Q * tc.grad(Q, xi) )

    return sy.simplify(E31)



def E32(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E32 = sy.zeros(vec_dim)

    for i in range(vec_dim):
        E32[i] = -sy.Rational(1, 2) * Phi_i[i].ip( tc.grad(Q, xi) ** tc.transpose_3(tc.grad(Q, xi)) )

    return sy.simplify(E32)



def dE1(Phi_i, Phi_j, xi):
    vec_dim = len(Phi_i)
    dE1 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE1[i, j] = -tc.grad(Phi_i[i], xi).ip( tc.grad(Phi_j[j], xi) )

    return sy.simplify(dE1)



def dE2(Phi_i, Phi_j, xi):
    vec_dim = len(Phi_i)
    dE2 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE2[i, j] = -tc.grad(Phi_i[i], xi).ip( tc.transpose_3(tc.grad(Phi_j[j], xi)) )

    return sy.simplify(dE2)



def dE31(Phi_i, Phi_j, Q, xi):
    vec_dim = len(Phi_i)
    dE31 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE31[i, j] = -tc.grad(Phi_i[i], xi).ip( Phi_j[j] * tc.grad(Q, xi) + Q * tc.grad(Phi_j[j], xi) )

    return sy.simplify(dE31)



def dE32(Phi_i, Phi_j, Q, xi):
    vec_dim = len(Phi_i)
    dE32 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE32[i, j] = -Phi_i[i].ip( tc.grad(Phi_j[j], xi) ** tc.transpose_3(tc.grad(Q, xi)) )

    return sy.simplify(dE32)




def calc_singular_potential_convex_splitting_residual(Phi_i, Q, Q0, Lambda, xi, alpha, L2, L3, dt):

    vec_dim = len(Phi_i)
    RQ = sy.zeros(vec_dim)
    RLambda = sy.zeros(vec_dim)

    for i in range(vec_dim):
        RQ[i] = Phi_i[i].ip(Q - (1 + alpha * dt) * Q0)
        RLambda[i] = -dt * (-Phi_i[i].ip(Lambda))

    RQ = sy.simplify(RQ)
    RLambda = sy.simplify(RLambda)

    RE1 = -dt * E1(Phi_i, Q, xi)
    RE2 = -dt * L2 * E2(Phi_i, Q, xi)
    RE31 = -dt * L3 * E31(Phi_i, Q, xi)
    RE32 = -dt * L3 * E32(Phi_i, Q, xi)

    return RQ, RLambda, RE1, RE2, RE31, RE32



def calc_singular_potential_convex_splitting_jacobian(Phi_i, Phi_j, xi, Q, dLambda, L2, L3, dt):

    vec_dim = len(Phi_i)
    dRQ = sy.zeros(vec_dim, vec_dim)
    dRLambda = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dRQ[i, j] = Phi_i[i].ip(Phi_j[j])
            dRLambda[i, j] = -dt * (-Phi_i[i].ip(dLambda[j]))

    dRQ = sy.simplify(dRQ) 
    dRLambda = sy.simplify(dRLambda) 

    dRE1 = -dt * dE1(Phi_i, Phi_j, xi)
    dRE2 = -dt * L2 * dE2(Phi_i, Phi_j, xi)
    dRE31 = -dt * L3 * dE31(Phi_i, Phi_j, Q, xi)
    dRE32 = -dt * L3 * dE32(Phi_i, Phi_j, Q, xi)

    return dRQ, dRLambda, dRE1, dRE2, dRE31, dRE32



def calc_singular_potential_semi_implicit_residual(Phi_i, xi, Q, Q0, Lambda, Lambda0, alpha, L2, L3, dt, theta):

    vec_dim = len(Phi_i)
    RQ = sy.zeros(vec_dim)
    RLambda = sy.zeros(vec_dim)

    for i in range(vec_dim):
        RQ[i] = Phi_i[i].ip( Q - Q0 - dt * (theta * alpha * Q0 + (1 - theta) * alpha * Q) )
        RLambda[i] = -dt * Phi_i[i].ip( theta * (-Lambda0) + (1 - theta) * (-Lambda) )

    RQ = sy.simplify(RQ)
    RLambda = sy.simplify(RLambda)

    RE1 = -dt * (theta * E1(Phi_i, Q0, xi) + (1 - theta) * E1(Phi_i, Q, xi))
    RE2 = -dt * (theta * L2 * E2(Phi_i, Q0, xi) + (1 - theta) * L2 * E2(Phi_i, Q, xi))
    RE31 = -dt * (theta * L3 * E31(Phi_i, Q0, xi) + (1 - theta) * L3 * E31(Phi_i, Q, xi))
    RE32 = -dt * (theta * L3 * E32(Phi_i, Q0, xi) + (1 - theta) * L3 * E32(Phi_i, Q, xi))

    return RQ, RLambda, RE1, RE2, RE31, RE32



def calc_singular_potential_semi_implicit_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3, dt, theta):

    vec_dim = len(Phi_i)
    dRQ = sy.zeros(vec_dim, vec_dim)
    dRLambda = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dRQ[i, j] = Phi_i[i].ip(Phi_j[j]) - dt * (1 - theta) * alpha * Phi_i[i].ip(Phi_j[j])
            dRLambda[i, j] = -dt * (1 - theta) * (-Phi_i[i].ip(dLambda[j]))

    dRQ = sy.simplify(dRQ) 
    dRLambda = sy.simplify(dRLambda) 

    dRE1 = -dt * (1 - theta) * dE1(Phi_i, Phi_j, xi)
    dRE2 = -dt * (1 - theta) * L2 * dE2(Phi_i, Phi_j, xi)
    dRE31 = -dt * (1 - theta) * L3 * dE31(Phi_i, Phi_j, Q, xi)
    dRE32 = -dt * (1 - theta) * L3 * dE32(Phi_i, Phi_j, Q, xi)

    return dRQ, dRLambda, dRE1, dRE2, dRE31, dRE32


def calc_singular_potential_newton_method_residual(Phi_i, xi, Q, Lambda, alpha, L2, L3):

    vec_dim = len(Phi_i)
    RQ = sy.zeros(vec_dim)
    RLambda = sy.zeros(vec_dim)

    for i in range(vec_dim):
        RQ[i] = Phi_i[i].ip(alpha * Q)
        RLambda[i] = Phi_i[i].ip(-Lambda)

    RQ = sy.simplify(RQ)
    RLambda = sy.simplify(RLambda)

    RE1 = E1(Phi_i, Q, xi)
    RE2 = L2 * E2(Phi_i, Q, xi)
    RE31 = L3 * E31(Phi_i, Q, xi)
    RE32 = L3 * E32(Phi_i, Q, xi)

    return RQ, RLambda, RE1, RE2, RE31, RE32



def calc_singular_potential_newton_method_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3):

    vec_dim = len(Phi_i)
    dRQ = sy.zeros(vec_dim, vec_dim)
    dRLambda = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dRQ[i, j] = Phi_i[i].ip(alpha * Phi_j[j])
            dRLambda[i, j] = Phi_i[i].ip(-dLambda[j])

    dRQ = sy.simplify(dRQ) 
    dRLambda = sy.simplify(dRLambda) 

    dRE1 = dE1(Phi_i, Phi_j, xi)
    dRE2 = L2 * dE2(Phi_i, Phi_j, xi)
    dRE31 = L3 * dE31(Phi_i, Phi_j, Q, xi)
    dRE32 = L3 * dE32(Phi_i, Phi_j, Q, xi)

    return dRQ, dRLambda, dRE1, dRE2, dRE31, dRE32



def calc_singular_potential_semi_implicit_surface_residual(Phi_i, Q, Q0, nu, S0, W1, W2, dt, theta):

    vec_dim = len(Phi_i)
    I = tc.TensorCalculusArray([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
    P = I - nu @ nu

    RQ = sy.zeros(vec_dim)
    RS = sy.zeros(vec_dim)

    for i in range(vec_dim):
        RQ[i] = -Phi_i[i].ip(Q - Q0)

        TS = (-2 * W1 * (Q - P * Q * P - sy.Rational(1, 3) * S0 * (nu @ nu))
              - 4 * W2 * ((Q ** Q) * Q - sy.Rational(2, 3) * S0**2 * Q) )
        TS = tc.TensorCalculusArray( sy.simplify(TS) )
        TS0 = (-2 * W1 * (Q0 - P * Q0 * P - sy.Rational(1, 3) * S0 * (nu @ nu))
               - 4 * W2 * ((Q0 ** Q0) * Q0 - sy.Rational(2, 3) * S0**2 * Q0) )
        TS0 = tc.TensorCalculusArray( sy.simplify(TS0) )

        RS[i] = Phi_i[i].ip( dt * (theta * TS0 + (1 - theta) * TS) )

    RQ = sy.simplify(RQ)
    RS = sy.simplify(RS)

    return RQ, RS



def calc_singular_potential_semi_implicit_surface_jacobian(Phi_i, Phi_j, Q, nu, S0, W1, W2, dt, theta):

    vec_dim = len(Phi_i)
    I = tc.TensorCalculusArray([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
    P = I - nu @ nu

    dRQ = sy.zeros(vec_dim, vec_dim)
    dRS = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):

            dTS = (-2 * W1 * (Phi_j[j] - P * Phi_j[j] * P)
                  - 4 * W2 * (2 * (Phi_j[j] ** Q) * Q 
                              + (Q ** Q) * Phi_j[j]
                              - sy.Rational(2, 3) * S0**2 * Phi_j[j]) )
            dTS = tc.TensorCalculusArray( sy.simplify(dTS) )

            dRQ[i, j] = Phi_i[i].ip(Phi_j[j])
            dRS[i, j] = Phi_i[i].ip( -dt * (1 - theta) * dTS )

    dRQ = sy.simplify(dRQ) 
    dRS = sy.simplify(dRS)

    return dRQ, dRS
