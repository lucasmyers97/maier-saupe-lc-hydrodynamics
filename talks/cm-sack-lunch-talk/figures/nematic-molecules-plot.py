import matplotlib.pyplot as plt
import numpy as np

def arrow_curve(length=1, width=0.15, head_length=0.25, head_width=0.25):

    r = np.zeros(4)
    z = np.zeros(4)

    r[0] = -width / 2
    r[1] = r[0]
    r[2] = - head_width / 2
    r[3] = 0

    z[1] = length - head_length / 2
    z[2] = z[1]
    z[3] = length + head_length / 2

    # want to center on origin
    z = z - length / 2

    return r, z

def standard_arrow_coords(length=1, width=0.15, head_length=0.25, head_width=0.25):

    r, z = arrow_curve(length, width, head_length, head_width)
    phi = np.linspace(0, 2*np.pi, num=1000)
    x = np.outer(r, np.cos(phi))
    y = np.outer(r, np.sin(phi))
    z = np.outer(z, np.ones(phi.shape))

    return x, y, z

def plot_arrow(u, v, x, y, length=1, width=0.5, head_length=0.25, head_width=0.4):

    

    return None

r, z = arrow_curve()

plt.plot(r, z)
plt.axis('equal')
plt.show()

x, y, z = standard_arrow_coords()
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(x, y, z)
ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1/np.sqrt(2), 1/np.sqrt(2)))
plt.show()

# x = np.linspace(x_min, x_max, n_x_arrows)
# y = x.copy()
# z = x.copy()

# X, Y, Z = np.meshgrid(x, y, z)

# U = np.ones(Z.shape)
# V = np.zeros(X.shape)
# W = np.zeros(Y.shape)

# ax.quiver(X, Y, Z, U, V, W, length=0.1, linewidths=5)

# plt.show()