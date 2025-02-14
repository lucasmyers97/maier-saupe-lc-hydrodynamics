"""
This script checks the energies and equations of motion of the 2D twisted
configurations.
First it calculates the energy by applying a rotation matrix and evaluating at 
zero against the calculation that we did by hand (on the rotated energy).
Then it takes the (correct) energy and calculates the Euler-Lagrange equation
automatically then checks against what we did by hand.

As it stands right now, everything should be correct.
"""

from sympy import *
init_printing()

vec_dim = 5
dim = 3

x, y, z = symbols('x, y, z', real=True)
xi = Matrix([x, y, z])

omega = symbols('omega', real=True)
Q_vec = Matrix([Function('Q_{}'.format(i))(x, y) for i in range(vec_dim)])

# Q_mat = Matrix([[Q_vec[0], Q_vec[1], Q_vec[2]],
#                 [Q_vec[1], Q_vec[3], Q_vec[4]],
#                 [Q_vec[2], Q_vec[4], -(Q_vec[0] + Q_vec[3])]])

Q_mat = Matrix([[Function('Q_{{{}{}}}'.format(i, j))(x, y) 
                 for j in range(dim)]
                for i in range(dim)])

Q_subs = { Q_mat[0, 0]: Q_vec[0],
           Q_mat[0, 1]: Q_vec[1],
           Q_mat[0, 2]: Q_vec[2],
           Q_mat[1, 1]: Q_vec[3],
           Q_mat[1, 2]: Q_vec[4],
           Q_mat[1, 0]: Q_vec[1],
           Q_mat[2, 0]: Q_vec[2],
           Q_mat[2, 1]: Q_vec[4],
           Q_mat[2, 2]: -(Q_vec[0] + Q_vec[3]) }

dxQ_subs = { k.diff(x): v.diff(x) for k, v in Q_subs.items() }
dyQ_subs = { k.diff(y): v.diff(y) for k, v in Q_subs.items() }

R = Matrix([[cos(omega * z), -sin(omega * z), 0],
            [sin(omega * z), cos(omega * z), 0],
            [0, 0, 1]])

A = Matrix([[0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]])

P = Matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])

delta = eye(dim)

def ST(mat):

    return Rational(1, 2) * (mat + mat.transpose()) - Rational(1, 3) * mat.trace() * delta

def check_energies(print_output=False):

    L1_term = sum( (R[i, m] * Q_mat[m, n] * R[j, n]).diff(xi[k])
                  * (R[i, alpha] * Q_mat[alpha, beta] * R[j, beta]).diff(xi[k])
                  for i in range(dim)
                  for j in range(dim)
                  for k in range(dim)
                  for m in range(dim)
                  for n in range(dim)
                  for alpha in range(dim)
                  for beta in range(dim) ) * Rational(1, 2)

    L1_term = expand( simplify(L1_term) )

    L1_term_calc = ( sum(Q_mat[i, j].diff(xi[k]) * Q_mat[i, j].diff(xi[k])
                         for i in range(dim)
                         for j in range(dim)
                         for k in range(dim))
                    +
                    omega**2 * sum((A[i, k] * Q_mat[k, j] - Q_mat[i, k] * A[k, j])
                                   * (A[i, l] * Q_mat[l, j] - Q_mat[i, l] * A[l, j])
                                   for i in range(dim)
                                   for j in range(dim)
                                   for k in range(dim)
                                   for l in range(dim))
                    ) * Rational(1, 2)

    L1_term_calc = expand( simplify(L1_term_calc) )

    L2_term = sum( (R[j, m] * Q_mat[m, n] * R[i, n]).diff(xi[j])
                  * (R[k, alpha] * Q_mat[alpha, beta] * R[i, beta]).diff(xi[k])
                  for i in range(dim)
                  for j in range(dim)
                  for k in range(dim)
                  for m in range(dim)
                  for n in range(dim)
                  for alpha in range(dim)
                  for beta in range(dim) ) * Rational(1, 2)

    L2_term = expand( simplify(L2_term.subs(z, 0)) )

    L2_term_calc = ( sum(Q_mat[j, i].diff(xi[j]) * Q_mat[k, i].diff(xi[k])
                         for i in range(dim)
                         for j in range(dim)
                         for k in range(dim))
                    -
                    2 * omega * sum( Q_mat[j, i].diff(xi[j]) * Q_mat[2, m] * A[m, i]
                                    for i in range(dim)
                                    for j in range(dim)
                                    for m in range(dim))
                    +
                    omega**2 * sum( Q_mat[2, m] * A[m, i] * Q_mat[2, n] * A[n, i]
                                   for i in range(dim)
                                   for m in range(dim)
                                   for n in range(dim))
                    ) * Rational(1, 2)

    L2_term_calc = expand( simplify(L2_term_calc) )

    L3_term = sum( R[k, m] * Q_mat[m, n] * R[l, n]
                   * (R[i, alpha] * Q_mat[alpha, beta] * R[j, beta]).diff(xi[k])
                   * (R[i, gamma] * Q_mat[gamma, delta] * R[j, delta]).diff(xi[l])
                  for i in range(dim)
                  for j in range(dim)
                  for k in range(dim)
                  for l in range(dim)
                  for m in range(dim)
                  for n in range(dim)
                  for alpha in range(dim)
                  for beta in range(dim)
                  for gamma in range(dim)
                  for delta in range(dim) ) * Rational(1, 2)

    L3_term = expand( simplify(L3_term.subs(z, 0)) )

    L3_term_calc = ( sum( Q_mat[k, l] * Q_mat[i, j].diff(xi[k]) * Q_mat[i, j].diff(xi[l])
                         for i in range(dim)
                         for j in range(dim)
                         for k in range(dim)
                         for l in range(dim) )
                    +
                    2 * omega * sum( Q_mat[2, k] * Q_mat[i, j].diff(xi[k]) * (A[i, m] * Q_mat[m, j] - Q_mat[i, m] * A[m, j])
                                    for i in range(dim)
                                    for j in range(dim)
                                    for k in range(dim)
                                    for m in range(dim) )
                    +
                    omega**2 * Q_mat[2, 2] * sum( (A[i, m] * Q_mat[m, j] - Q_mat[i, m] * A[m, j])
                                             * (A[i, n] * Q_mat[n, j] - Q_mat[i, n] * A[n, j])
                                             for i in range(dim)
                                             for j in range(dim)
                                             for n in range(dim)
                                             for m in range(dim) )
                    ) * Rational(1, 2)


    L3_term_calc = expand( simplify(L3_term_calc) )

    if print_output:
        preview(L1_term.subs(Q_subs) - L1_term_calc.subs(Q_subs), output='svg')
        preview(L2_term.subs(Q_subs) - L2_term_calc.subs(Q_subs), output='svg')
        preview(L3_term.subs(Q_subs) - L3_term_calc.subs(Q_subs), output='svg')

    return L1_term, L2_term, L3_term



def check_equations_of_motion(L1_term, L2_term, L3_term):

    L1_eom = zeros(dim)
    for i in range(dim):
        for j in range(dim):
            L1_eom[i, j] = (-L1_term.diff(Q_mat[i, j]) 
                            + L1_term.diff(Q_mat[i, j].diff(x)).diff(x)
                            + L1_term.diff(Q_mat[i, j].diff(y)).diff(y)
                            )

    L2_eom = zeros(dim)
    for i in range(dim):
        for j in range(dim):
            L2_eom[i, j] = (-L2_term.diff(Q_mat[i, j]) 
                            + L2_term.diff(Q_mat[i, j].diff(x)).diff(x)
                            + L2_term.diff(Q_mat[i, j].diff(y)).diff(y)
                            )

    L3_eom = zeros(dim)
    for i in range(dim):
        for j in range(dim):
            L3_eom[i, j] = (-L3_term.diff(Q_mat[i, j]) 
                            + L3_term.diff(Q_mat[i, j].diff(x)).diff(x)
                            + L3_term.diff(Q_mat[i, j].diff(y)).diff(y)
                            )

    L1_eom = ST( simplify( L1_eom.subs(Q_subs) ) )
    L2_eom = ST( simplify( L2_eom.subs(Q_subs) ) )
    L3_eom = ST( simplify( L3_eom.subs(Q_subs) ) )

    L1_eom_calc = ( Q_mat.diff(x, 2) + Q_mat.diff(y, 2)
                   - omega**2 * ( P * Q_mat + Q_mat * P + 2 * A * Q_mat * A )
                   ).subs(Q_subs)

    L2_eom_calc = zeros(dim)
    for i in range(dim):
        for j in range(dim):
            L2_eom_calc[i, j] = ( sum(delta[k, j] * Q_mat[l, i].diff(xi[l]).diff(xi[k])
                                      for k in range(dim)
                                      for l in range(dim))
                                 - omega * (
                                     sum(delta[i, 2] * Q_mat[k, m].diff(xi[k]) * A[m, j]
                                         for k in range(dim)
                                         for m in range(dim))
                                     +
                                     sum(Q_mat[2, n].diff(xi[i]) * A[n, j]
                                         for n in range(dim))
                                     )
                                 - omega**2 * (
                                     sum(Q_mat[2, m] * P[m, j] * delta[i, 2]
                                         for m in range(dim))
                                     )
                                 ).subs(Q_subs)

    L3_eom_calc = zeros(dim)
    for i in range(dim):
        for j in range(dim):
            L3_eom_calc[i, j] = (
                    -Rational(1, 2) * sum( Q_mat[k, l].diff(xi[i]) * Q_mat[k, l].diff(xi[j])
                                          for k in range(dim)
                                          for l in range(dim) )
                    +
                    sum( (Q_mat[k, l] * Q_mat[i, j].diff(xi[l])).diff(xi[k])
                        for k in range(3)
                        for l in range(3) )
                    + omega * (
                        - sum(Q_mat[l, m].diff(xi[j]) * (A[m, n] * Q_mat[n, l] - Q_mat[m, n] * A[n, l]) * delta[i, 2]
                              for l in range(dim) 
                              for m in range(dim) 
                              for n in range(dim)
                              )
                        - sum(Q_mat[2, k] * (Q_mat[j, m].diff(xi[k]) * A[m, i] - Q_mat[m, i].diff(xi[k]) * A[j, m])
                              for k in range(dim)
                              for m in range(dim)
                              )
                        +
                        sum( (Q_mat[2, k] * (A[i, p] * Q_mat[p, j] - Q_mat[i, p] * A[p, j])).diff(xi[k])
                            for k in range(dim)
                            for p in range(dim)
                            )
                        )
                    - omega**2 / 2 * (
                        delta[i, 2] * delta[j, 2] * sum(
                            (A[k, m] * Q_mat[m, l] - Q_mat[k, m] * A[m, l])
                            * (A[k, n] * Q_mat[n, l] - Q_mat[k, n] * A[n, l])
                            for k in range(dim)
                            for l in range(dim)
                            for m in range(dim)
                            for n in range(dim)
                            )
                        +
                        2 * Q_mat[2, 2] * (
                            sum(
                                P[i, n] * Q_mat[n, j] + Q_mat[i, n] * P[n, j] 
                                for n in range(dim)
                                )
                            +
                            sum(2 * A[i, k] * Q_mat[k, n] * A[n, j]
                                for n in range(dim)
                                for k in range(dim)
                                )
                            )
                        )
                    ).subs(Q_subs)

    L1_eom_calc = ST(L1_eom_calc)
    L2_eom_calc = ST(L2_eom_calc)
    L3_eom_calc = ST(L3_eom_calc)

    preview( expand(simplify( L1_eom - L1_eom_calc)), output='svg')
    preview( expand(simplify( L2_eom - L2_eom_calc)), output='svg') # need to fix this
    preview( expand(simplify( L3_eom - L3_eom_calc)), output='svg')

    return L1_eom, L2_eom, L3_eom



def main():

    L1_term, L2_term, L3_term = check_energies()
    L1_eom, L2_eom, L3_eom = check_equations_of_motion(L1_term, L2_term, L3_term)
    

if __name__ == '__main__':
    main()
