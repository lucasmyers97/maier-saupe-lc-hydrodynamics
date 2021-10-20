import numpy as np
import matplotlib.pyplot as plt

b = -1
c = 1

S = np.linspace(-1, 4, num=1000)

Q = np.zeros((3, 3, 1000))
Q[0, 0, :] = S
Q[1, 1, :] = -S / 2
Q[2, 2, :] = -S / 2

f1 = sum(Q[i, j, :]*Q[i, j, :]
         for i in range(3)
         for j in range(3))

f2 = sum(Q[i, j, :]*Q[j, k, :]*Q[k, i, :]
         for i in range(3)
         for j in range(3)
         for k in range(3))

f3 = sum((Q[i, j, :]*Q[i, j, :])**2
         for i in range(3)
         for j in range(3))

a_coeff_array = [0.9, 1, 1.1]


for a_coeff in a_coeff_array:
    a = a_coeff*(2 * b**2 / (9*c))
    
    A = (2 / 3) * a
    B = (4 / 3)*b
    C = (4 / 9)*c
    
    f = (1 / 2) * A*f1 + (1 / 3) * B*f2 + (1 / 3) * C*f3
    
    plt.plot(S, f, label="A = " + str(A))

# plt.ylim((-0.02, 0.02))
# plt.xlim(-0.1, 0.7)
plt.show()