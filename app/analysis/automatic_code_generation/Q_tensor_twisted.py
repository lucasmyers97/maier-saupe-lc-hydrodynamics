import sympy as sy

from ..utilities import Q_tensor_math as qtm
from ..utilities import tensor_calculus as tc

space_dim = 2
dim = 3
basis = qtm.QTensorMath.Basis.component_wise

m = qtm.QTensorMath(space_dim, basis)

Q = m.tensors.Q
Q0 = m.tensors.Q0
A = m.tensors.A
P = m.tensors.P
I = m.tensors.I

Q_vec = m.fe_fields.Q_vec
Q0_vec = m.fe_fields.Q0_vec
tau = sy.symbols('tau', real=True)
jac_subs = {Q_vec[i]: Q_vec[i] + tau * Q0_vec[i] for i in range(m.domain.vec_dim)}

xi = m.domain.xi
x = xi[0]
y = xi[1]

z = tc.TensorCalculusArray([0, 0, 1])

L2 = m.parameters.L2
L3 = m.parameters.L3
omega = m.parameters.omega

ST = lambda M: m.ST(M)

expr = (
        -omega * (
            L2 * ( z @ tc.div(Q*A, xi) + tc.grad(z * Q * A, xi) )
            +
            L3 * (
                z @ (tc.grad(Q, xi) ** (A*Q - Q*A))
                - z * Q * tc.grad(A*Q - Q*A, xi)
                - tc.div((Q*z) @ (A*Q - Q*A), xi)
                )
            )
        -
        omega**2 * (
            P*Q + Q*P + 2*A*Q*A
            +
            L2 * (z @ (z*Q*P))
            +
            L3 / 2 * (
                (A*Q - Q*A).ip(A*Q - Q*A) * (z@z)
                + 2 * (z*Q*z) * (P*Q + Q*P + 2*A*Q*A)
                )
            )
        )

expr = ST(expr).simplify()


L1_eom_calc = sy.zeros(dim)
for i in range(dim):
    for j in range(dim):
        L1_eom_calc[i, j] = (
                - omega**2 * ( P * Q + Q * P + 2 * A * Q * A )[i, j]
                   )

L2_eom_calc = sy.zeros(dim)
for i in range(dim):
    for j in range(dim):
        L2_eom_calc[i, j] = ( 
                             - omega * (
                                 sum(I[i, 2] * Q[k, m].diff(xi[k]) * A[m, j]
                                     for k in range(dim)
                                     for m in range(dim))
                                 +
                                 sum(Q[2, n].diff(xi[i]) * A[n, j]
                                     for n in range(dim))
                                 )
                             - omega**2 * (
                                 sum(Q[2, m] * P[m, j] * I[i, 2]
                                     for m in range(dim))
                                 )
                             )

L3_eom_calc = sy.zeros(dim)
for i in range(dim):
    for j in range(dim):
        L3_eom_calc[i, j] = (
                omega * (
                    - sum(Q[l, m].diff(xi[j]) * (A[m, n] * Q[n, l] - Q[m, n] * A[n, l]) * I[i, 2]
                          for l in range(dim) 
                          for m in range(dim) 
                          for n in range(dim)
                          )
                    - sum(Q[2, k] * (Q[j, m].diff(xi[k]) * A[m, i] - Q[m, i].diff(xi[k]) * A[j, m])
                          for k in range(dim)
                          for m in range(dim)
                          )
                    +
                    sum( (Q[2, k] * (A[i, p] * Q[p, j] - Q[i, p] * A[p, j])).diff(xi[k])
                        for k in range(dim)
                        for p in range(dim)
                        )
                    )
                - omega**2 / 2 * (
                    I[i, 2] * I[j, 2] * sum(
                        (A[k, m] * Q[m, l] - Q[k, m] * A[m, l])
                        * (A[k, n] * Q[n, l] - Q[k, n] * A[n, l])
                        for k in range(dim)
                        for l in range(dim)
                        for m in range(dim)
                        for n in range(dim)
                        )
                    +
                    2 * Q[2, 2] * (
                        sum(
                            P[i, n] * Q[n, j] + Q[i, n] * P[n, j] 
                            for n in range(dim)
                            )
                        +
                        sum(2 * A[i, k] * Q[k, n] * A[n, j]
                            for n in range(dim)
                            for k in range(dim)
                            )
                        )
                    )
                )

expr2 = ST( tc.TensorCalculusArray( L1_eom_calc + L2* L2_eom_calc + L3*L3_eom_calc ) ).simplify()

diff = expr - expr2

# for i in range(dim):
#     for j in range(dim):
#         sy.preview( sy.expand(diff[i, j]), output='svg')

jexpr = (
        -omega * (
            L2 * (z @ tc.div(Q0*A, xi) + tc.grad(z * Q0 * A, xi))
            +
            L3 * (z @ tc.grad(Q0, xi) ** (A*Q - Q*A)
                  + z @ tc.grad(Q, xi) ** (A*Q0 - Q0*A)
                  - (z * Q0) * tc.grad(A*Q - Q*A, xi)
                  - (z * Q) * tc.grad(A*Q0 - Q0*A, xi)
                  - tc.div((Q0*z) @ (A*Q - Q*A), xi)
                  - tc.div((Q*z) @ (A*Q0 - Q0*A), xi)
                  )
            )
        -
        omega**2 * (
            P*Q0 + Q0*P + 2*A*Q0*A
            + 
            L2 * (
                z @ (z * Q0 * P)
                )
            +
            L3 / 2 * (
                2 * (A*Q0 - Q0*A) ** (A*Q - Q*A) * (z@z)
                + 2 * (z * Q0 * z) * (P*Q + Q*P + 2*A*Q*A)
                + 2 * (z * Q * z) * (P*Q0 + Q0*P + 2*A*Q0*A)
                )
            )
        ).simplify()

jexpr = ST(jexpr).simplify()

jexpr2 = tc.TensorCalculusArray.zeros(dim, dim)
for i in range(dim):
    for j in range(dim):
        jexpr2[i, j] = expr[i, j].subs(jac_subs).diff(tau).subs(tau, 0)

jdiff = jexpr - jexpr2
# sy.preview( sy.expand(jexpr2[0, 0]), output='svg' )
# sy.preview( sy.expand(jdiff[0, 0]), output='svg' )
for i in range(dim):
    for j in range(dim):
        sy.preview( sy.expand(jdiff[i, j]), output='svg' )
