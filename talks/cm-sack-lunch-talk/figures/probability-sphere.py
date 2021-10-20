import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import quadpy

def calc_sphere_coords(n=100):

    theta = np.linspace(0, np.pi / 2, num=n)
    phi = np.linspace(0, 2*np.pi, num=n)
    Theta, Phi = np.meshgrid(theta, phi)

    X = np.sin(Theta)*np.cos(Phi)
    Y = np.sin(Theta)*np.sin(Phi)
    Z = np.cos(Theta)

    return X, Y, Z

def p(X, Y):

    theta = np.arctan2(X, Y)
    R = np.sqrt(X**2 + Y**2)

    return 1 / (R + 1)


def p_gauss_int(theta_phi, sigma=1):
    theta, phi = theta_phi
    return np.exp(-theta**2 / sigma)

def P2_int(theta_phi):
    theta, phi = theta_phi
    return (1/2) * (3 * np.cos(theta)**2 - 1)

def p_gauss(X, Y, sigma=1):
    scheme = quadpy.u3.get_good_scheme(19)
    norm = scheme.integrate_spherical(
                lambda theta_phi : p_gauss_int(theta_phi, sigma))
    Z = np.sqrt(np.abs(1 - (X**2 + Y**2)))
    theta = np.arccos(Z)
    return (1 / norm) * np.exp(-theta**2 / sigma)

def p_const_int(theta_phi):
    theta, phi = theta_phi
    return (1 / (4*np.pi)) * np.ones(theta.shape)



def plot_sphere(n=100, sigma=1):

    offset = 0.3

    X, Y, Z = calc_sphere_coords(n)
    p_vals = p_gauss(X, Y, sigma)
    norm = mpl.colors.Normalize()
    cmap = plt.get_cmap('jet')
    C = cmap(norm(p_vals))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=C, cstride=1, rstride=1, vmax=2)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), 
           zlim=(-1/np.sqrt(2) + offset, 1/np.sqrt(2) + offset))

    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(C)
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.005, aspect=15)
    ax.axis("off")
    fig.tight_layout()

n = 500


sigma = 0.1
plot_sphere(n, sigma)
plt.savefig("prob-sphere-sigma" + str(sigma) + ".png")
plt.show()

scheme = quadpy.u3.get_good_scheme(19)
norm = scheme.integrate_spherical(lambda theta_phi : p_gauss_int(theta_phi, sigma))
val = scheme.integrate_spherical(lambda theta_phi : P2_int(theta_phi)*p_gauss_int(theta_phi, sigma))
print("S =", val / norm, "for sigma =", sigma)

sigma = 0.25
plot_sphere(n, sigma)
plt.savefig("prob-sphere-sigma" + str(sigma) + ".png")
plt.show()

scheme = quadpy.u3.get_good_scheme(19)
norm = scheme.integrate_spherical(lambda theta_phi : p_gauss_int(theta_phi, sigma))
val = scheme.integrate_spherical(lambda theta_phi : P2_int(theta_phi)*p_gauss_int(theta_phi, sigma))
print("S =", val / norm, "for sigma =", sigma)

sigma = 0.5
plot_sphere(n, sigma)
plt.savefig("prob-sphere-sigma" + str(sigma) + ".png")
plt.show()

scheme = quadpy.u3.get_good_scheme(19)
norm = scheme.integrate_spherical(lambda theta_phi : p_gauss_int(theta_phi, sigma))
val = scheme.integrate_spherical(lambda theta_phi : P2_int(theta_phi)*p_gauss_int(theta_phi, sigma))
print("S =", val / norm, "for sigma =", sigma)