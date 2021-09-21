import sympy as sp

def int_S2(expr):
    
    theta, phi = sp.symbols('theta phi')
    return sp.integrate(
                sp.integrate(expr*sp.sin(theta), 
                             (theta, 0, sp.pi)),
                        (phi, 0, 2*sp.pi) )



def Jac(m, n):

    theta, phi = sp.symbols('theta phi')
    x = sp.cos(phi)*sp.sin(theta)
    y = sp.sin(phi)*sp.sin(theta)
    z = sp.cos(theta)

    xi = [x, y, z]
    i = [0, 0, 0, 1, 1]
    j = [0, 1, 2, 1, 2]

    if n == 0 or n == 3:
        return (
                1 / (4*sp.pi) * int_S2( xi[i[m]]*xi[j[m]] 
                                        * (xi[i[n]]**2 - xi[2]**2) )
                 - 1 / (16*sp.pi**2) * int_S2( xi[i[m]]*xi[j[m]]) 
                                               * int_S2(xi[i[n]] - xi[2] )
                )
    else:
        return (
                1 / (2*sp.pi) * int_S2( xi[i[m]]*xi[j[m]] 
                                        * xi[i[n]]*xi[j[n]] )
                 - 1 / (8*sp.pi**2) * int_S2(xi[i[m]]*xi[j[m]]) 
                                    * int_S2(xi[i[n]]*xi[j[n]])
                )



if __name__ == "__main__":

    vec_dim = 5

    J = sp.zeros(vec_dim)
    for k in range(vec_dim):
        for l in range(vec_dim):
            J[k, l] = Jac(k, l)

    J = sp.simplify(J)
    print(J)
    print(sp.latex(J))