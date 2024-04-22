"""
Holds functions which symbolically produce the residual and jacobian terms for 
different discretization techniques.
"""

import sympy as sy
from ..utilities import tensor_calculus as tc

def E1(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E1 = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        E1[i] = -tc.grad(Phi_i[i], xi).ip( tc.grad(Q, xi) )

    return E1



def E2(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E2 = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        E2[i] = -tc.div(Phi_i[i], xi).ip( tc.div(Q, xi) )

    return E2



def E3(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E3 = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        E3[i] = -tc.grad(Phi_i[i], xi).ip( Q * tc.grad(Q, xi) )

    return E3



def E32(Phi_i, Q, xi):
    vec_dim = len(Phi_i)
    E32 = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        E32[i] = -sy.Rational(1, 2) * Phi_i[i].ip( tc.grad(Q, xi) ** tc.transpose_3(tc.grad(Q, xi)) )

    return E32



def dE1(Phi_i, Phi_j, xi):
    vec_dim = len(Phi_i)
    dE1 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE1[i, j] = -tc.grad(Phi_i[i], xi).ip( tc.grad(Phi_j[j], xi) )

    return dE1



def dE2(Phi_i, Phi_j, xi):
    vec_dim = len(Phi_i)
    dE2 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE2[i, j] = -tc.div(Phi_i[i], xi).ip( tc.div(Phi_j[j], xi) )

    return dE2



def dE3(Phi_i, Phi_j, Q, xi):
    vec_dim = len(Phi_i)
    dE3 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE3[i, j] = -tc.grad(Phi_i[i], xi).ip( Phi_j[j] * tc.grad(Q, xi) + Q * tc.grad(Phi_j[j], xi) )

    return dE3



def dE32(Phi_i, Phi_j, Q, xi):
    vec_dim = len(Phi_i)
    dE32 = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dE32[i, j] = -Phi_i[i].ip( tc.grad(Phi_j[j], xi) ** tc.transpose_3(tc.grad(Q, xi)) )

    return dE32



def TQ(Phi_i, Q, Lambda, xi, kappa, L3):

    vec_dim = len(Phi_i)
    mean_field = sy.zeros(vec_dim, 1)
    entropy = sy.zeros(vec_dim, 1)
    elastic = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        mean_field[i] = kappa * Phi_i[i].ip(Q)
        entropy[i] = -Phi_i[i].ip(Lambda)
        elastic[i] = -L3 / 2 * Phi_i[i].ip( tc.grad(Q, xi) ** tc.transpose_3(tc.grad(Q, xi)) )

    return mean_field, entropy, elastic



def TdQ(Phi_i, Q, xi, L2, L3):

    return E1(Phi_i, Q, xi), L2 * E2(Phi_i, Q, xi), L3 * E3(Phi_i, Q, xi)



def dTQ(Phi_i, Phi_j, xi, Q, dLambda, kappa, L3):

    vec_dim = len(Phi_i)
    mean_field = sy.zeros(vec_dim, vec_dim)
    entropy = sy.zeros(vec_dim, vec_dim)
    elastic = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            mean_field[i, j] = kappa * Phi_i[i].ip(Phi_j[j])
            entropy[i, j] = -Phi_i[i].ip(dLambda[j])
            elastic[i, j] = -L3 * Phi_i[i].ip( 
                                              tc.grad(Q, xi) 
                                              ** tc.transpose_3( tc.grad(Phi_j[j], xi) ) 
                                              )

    return mean_field, entropy, elastic



def dTdQ(Phi_i, Phi_j, Q, xi, L2, L3):

    return (
            dE1(Phi_i, Phi_j, xi),
            L2 * dE2(Phi_i, Phi_j, xi),
            L3 * dE3(Phi_i, Phi_j, Q, xi)
            )



def calc_singular_potential_convex_splitting_residual(Phi_i, Q, Q0, Lambda, xi, alpha, L2, L3, dt):

    vec_dim = len(Phi_i)
    RQ = sy.zeros(vec_dim, 1)
    RLambda = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        RQ[i] = Phi_i[i].ip(Q - (1 + alpha * dt) * Q0)
        RLambda[i] = -dt * (-Phi_i[i].ip(Lambda))

    RQ = sy.simplify(RQ)
    RLambda = sy.simplify(RLambda)

    RE1 = -dt * sy.simplify( E1(Phi_i, Q, xi) )
    RE2 = -dt * L2 * sy.simplify( E2(Phi_i, Q, xi) )
    RE31 = -dt * L3 * sy.simplify( E3(Phi_i, Q, xi) )
    RE32 = -dt * L3 * sy.simplify( E32(Phi_i, Q, xi) )

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

    dRE1 = -dt * sy.simplify( dE1(Phi_i, Phi_j, xi) )
    dRE2 = -dt * L2 * sy.simplify( dE2(Phi_i, Phi_j, xi) )
    dRE31 = -dt * L3 * sy.simplify( dE3(Phi_i, Phi_j, Q, xi) )
    dRE32 = -dt * L3 * sy.simplify( dE32(Phi_i, Phi_j, Q, xi) )

    return dRQ, dRLambda, dRE1, dRE2, dRE31, dRE32



def calc_singular_potential_semi_implicit_residual(Phi_i, xi, Q, Q0, Lambda, Lambda0, kappa, L2, L3, dt, theta):

    vec_dim = len(Phi_i)
    RQ = sy.Matrix([sy.simplify( Phi_i[i].ip( Q - Q0 ) )
                    for i in range(vec_dim)])

    TQ_terms = TQ(Phi_i, Q, Lambda, xi, kappa, L3)
    TQ0_terms = TQ(Phi_i, Q0, Lambda0, xi, kappa, L3)

    TdQ_terms = TdQ(Phi_i, Q, xi, L2, L3)
    TdQ0_terms = TdQ(Phi_i, Q0, xi, L2, L3)

    T_terms = tuple( -dt * (theta * sy.simplify(TQ0_term) 
                            + (1 - theta) * sy.simplify(TQ_term))
                     for TQ0_term, TQ_term in zip(TQ_terms, TQ0_terms) 
                    )
    Td_terms = tuple( -dt * (theta * sy.simplify(TdQ0_term) 
                             + (1 - theta) * sy.simplify(TdQ_term))
                     for TdQ0_term, TdQ_term in zip(TdQ_terms, TdQ0_terms) 
                     )

    return RQ, T_terms + Td_terms



def calc_singular_potential_semi_implicit_jacobian(Phi_i, Phi_j, xi, Q, dLambda, kappa, L2, L3, dt, theta):

    vec_dim = len(Phi_i)
    RQ = sy.zeros(vec_dim, vec_dim)
    for i in range(vec_dim):
        for j in range(vec_dim):
            RQ[i, j] = sy.simplify( Phi_i[i].ip( Phi_j[j] ) )

    dTQ_terms = dTQ(Phi_i, Phi_j, xi, Q, dLambda, kappa, L3) 
    dTdQ_terms = dTdQ(Phi_i, Phi_j, Q, xi, L2, L3) 

    dT_terms = tuple( -dt * (1 - theta) * sy.simplify(dTQ_term)
                      for dTQ_term in dTQ_terms 
                     )
    dTd_terms = tuple( -dt *  (1 - theta) * sy.simplify(dTdQ_term)
                      for dTdQ_term in dTdQ_terms
                      )

    return RQ, dT_terms + dTd_terms



def calc_singular_potential_newton_method_residual(Phi_i, xi, Q, Lambda, kappa, L2, L3):

    TQ_terms = TQ(Phi_i, Q, Lambda, xi, kappa, L3)

    TdQ_terms = TdQ(Phi_i, Q, xi, L2, L3)

    T_terms = tuple( sy.simplify(TQ_term) for TQ_term in TQ_terms )
    Td_terms = tuple( sy.simplify(TdQ_term) for TdQ_term in TdQ_terms )

    return T_terms + Td_terms



def calc_singular_potential_newton_method_jacobian(Phi_i, Phi_j, xi, Q, dLambda, kappa, L2, L3):

    dTQ_terms = dTQ(Phi_i, Phi_j, xi, Q, dLambda, kappa, L3) 
    dTdQ_terms = dTdQ(Phi_i, Phi_j, Q, xi, L2, L3) 

    dT_terms = tuple( sy.simplify(dTQ_term) for dTQ_term in dTQ_terms )
    dTd_terms = tuple( sy.simplify(dTdQ_term) for dTdQ_term in dTdQ_terms )

    return dT_terms + dTd_terms



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

        TS = (-2 * W1 * (Q - P * Q * P + sy.Rational(1, 3) * S0 * (nu @ nu))
              - 4 * W2 * ((Q ** Q) * Q - sy.Rational(2, 3) * S0**2 * Q) )
        TS = tc.TensorCalculusArray( sy.simplify(TS) )
        TS0 = (-2 * W1 * (Q0 - P * Q0 * P + sy.Rational(1, 3) * S0 * (nu @ nu))
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



def calc_singular_potential_semi_implicit_rotated_residual(Phi_i, xi, Q, Q0, Lambda, Lambda0, alpha, L2, L3, omega, dt, theta):

    vec_dim = len(Phi_i)
    RQ = sy.zeros(vec_dim, 1)
    RLambda = sy.zeros(vec_dim, 1)

    for i in range(vec_dim):
        RQ[i] = Phi_i[i].ip( Q - Q0 - dt * (theta * alpha * Q0 + (1 - theta) * alpha * Q) )
        RLambda[i] = -dt * Phi_i[i].ip( theta * (-Lambda0) + (1 - theta) * (-Lambda) )

    RQ = sy.simplify(RQ)
    RLambda = sy.simplify(RLambda)

    z = xi[-1]
    R = tc.TensorCalculusArray([[sy.cos(omega * z), -sy.sin(omega * z), 0],
                                [sy.sin(omega * z), sy.cos(omega * z), 0],
                                [0, 0, 1]])

    RT = R.transpose()

    Phi_i_rot = [R*Phi*RT for Phi in Phi_i]

    RE1 = -dt * (theta * E1(Phi_i_rot, R*Q0*RT, xi) + (1 - theta) * E1(Phi_i_rot, R*Q*RT, xi)).subs(z, 0)
    RE2 = -dt * (theta * L2 * E2(Phi_i_rot, R*Q0*RT, xi) + (1 - theta) * L2 * E2(Phi_i_rot, R*Q*RT, xi)).subs(z, 0)
    RE31 = -dt * (theta * L3 * E3(Phi_i_rot, R*Q0*RT, xi) + (1 - theta) * L3 * E31(Phi_i_rot, R*Q*RT, xi)).subs(z, 0)
    RE32 = -dt * (theta * L3 * E32(Phi_i_rot, R*Q0*RT, xi) + (1 - theta) * L3 * E32(Phi_i_rot, R*Q*RT, xi)).subs(z, 0)

    RE1 = sy.simplify(RE1)
    RE2 = sy.simplify(RE2)
    RE31 = sy.simplify(RE31)
    RE32 = sy.simplify(RE32)

    return RQ, RLambda, RE1, RE2, RE31, RE32


def calc_singular_potential_semi_implicit_rotated_jacobian(Phi_i, Phi_j, xi, Q, dLambda, alpha, L2, L3, omega, dt, theta):

    vec_dim = len(Phi_i)
    dRQ = sy.zeros(vec_dim, vec_dim)
    dRLambda = sy.zeros(vec_dim, vec_dim)

    for i in range(vec_dim):
        for j in range(vec_dim):
            dRQ[i, j] = Phi_i[i].ip(Phi_j[j]) - dt * (1 - theta) * alpha * Phi_i[i].ip(Phi_j[j])
            dRLambda[i, j] = -dt * (1 - theta) * (-Phi_i[i].ip(dLambda[j]))

    dRQ = sy.simplify(dRQ) 
    dRLambda = sy.simplify(dRLambda) 

    z = xi[-1]
    R = tc.TensorCalculusArray([[sy.cos(omega * z), -sy.sin(omega * z), 0],
                                [sy.sin(omega * z), sy.cos(omega * z), 0],
                                [0, 0, 1]])

    RT = R.transpose()

    Phi_i_rot = [R*Phi*RT for Phi in Phi_i]
    Phi_j_rot = [R*Phi*RT for Phi in Phi_j]

    dRE1 = -dt * (1 - theta) * dE1(Phi_i_rot, Phi_j_rot, xi).subs(z, 0)
    dRE2 = -dt * (1 - theta) * L2 * dE2(Phi_i_rot, Phi_j_rot, xi).subs(z, 0)
    dRE31 = -dt * (1 - theta) * L3 * dE3(Phi_i_rot, Phi_j_rot, R*Q*RT, xi).subs(z, 0)
    dRE32 = -dt * (1 - theta) * L3 * dE32(Phi_i_rot, Phi_j_rot, R*Q*RT, xi).subs(z, 0)

    dRE1 = sy.simplify(dRE1)
    dRE2 = sy.simplify(dRE2)
    dRE31 = sy.simplify(dRE31)
    dRE32 = sy.simplify(dRE32)

    return dRQ, dRLambda, dRE1, dRE2, dRE31, dRE32
