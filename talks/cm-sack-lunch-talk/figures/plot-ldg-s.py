import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

mpl.rc('font', **font)

b = -1
c = 3/2

S = np.linspace(-0.2, 0.9, num=1000)

a_coeff_array = [1.5, 1, .4]
ls_array = ['solid', 'dotted', 'dashed']
labels = [r'$A > A_0$', r'$A = A_0$', r'$A < A_0$']

fig, ax = plt.subplots(1, 1)
for a_coeff, ls, label in zip(a_coeff_array, ls_array, labels):
    a = a_coeff * 2 * b**2 / (9 * c)
    
    f = a / 2 * S**2 + b / 3 * S**3 + 1 / 4 * c * S**4
    
    ax.plot(S, f, lw=4, ls=ls, label=label)
    ax.set_ylim((-0.02, 0.02))
    ax.set_xlim((-0.2, 0.9))

pt = (-2*b / (3*c))
xt = ax.get_xticks()
closest = np.argmin(np.abs(xt - pt))
xt = np.delete(xt, closest)
xt = np.append(xt, pt)

xtl = xt.round(1).tolist()
xtl[-1]=r'$\frac{-2B}{81C}$'
ax.set_xticks(xt)
ax.set_xticklabels(xtl)

plt.xlabel(r'$S$')
plt.ylabel(r'$f$')
fig.tight_layout()

plt.legend()
plt.show()