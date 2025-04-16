from sympy import *

L2_val = 10.0
desired_K2 = 1/6

S, L2, L3, K2, K3 = symbols('S L_2 L_3 K_2 K_3')

K1_expr = 4 * S**2 + 2 * L2 * S**2 - Rational(4, 3) * L3 * S**2
K2_expr = 4 * S**2 - Rational(4, 3) * L3 * S**3
K3_expr = 4 * S**2 + 2 * L2 * S**2 + Rational(8, 3) * L3 * S**3

# preview( simplify(K2_expr/K1_expr), output='svg' )
# preview( simplify(K3_expr/K1_expr), output='svg' )

# 
# print('L2 = {}'.format(L2_val))
# print()
# for L3_val in (0.0, 1.0, 2.0, 3, 3.5, 4):
#     subs_dict = {S: 0.6751, L2: L2_val, L3: L3_val}
# 
#     print('For L3 = {}'.format(L3_val))
#     print( 'K2/K1 is given by: {}'.format(simplify(K2_expr/K1_expr).subs(subs_dict)), end=' ' )
#     print( 'K3/K1 is given by: {}'.format(simplify(K3_expr/K1_expr).subs(subs_dict)) )
#     print()
# 
# calc_L2_val = solve( simplify((K2_expr / K1_expr).subs(L3, 0) - K2), L2 )[0].subs(K2, desired_K2)
# print('For L3 = 0, K3/K1 = 0, and K2/K1 = {}, L3 = {}'.format(desired_K2, calc_L2_val) )

# Solve for L2 and L3
solution = solve([K2_expr/K1_expr - K2, K3_expr/K1_expr - K3], [L2, L3], dict=True)[0]

# preview(simplify(solution[L2]), output='svg')
preview(simplify(solution[L3]), output='svg')

# preview( simplify(solution[L2]), output='svg' )
# preview( solution[L3], output='svg' )

# K2_val = 1/6
# print(K2_val)
# for K3_val in (0.8, 0.9, 1, 1.25, 1.5, 2, 2.5):
#     subs_dict = {S: 0.6751, K2: K2_val, K3: K3_val}
#     
#     print(subs_dict)
#     print( 'L2: {}'.format(solution[L2].subs(subs_dict)), end=' ' )
#     print( 'L3: {}'.format(solution[L3].subs(subs_dict)), end='\n\n' )
# 
# 
subs_dict = {S: 0.6751, L2: 1.5, L3: 0.0}
print( 'K2/K1 is given by: {}'.format(simplify(K2_expr/K1_expr).subs(subs_dict)), end=' ' )
