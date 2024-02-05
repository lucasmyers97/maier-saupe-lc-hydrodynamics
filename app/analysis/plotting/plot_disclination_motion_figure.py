import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def calc_grad_perp(R, Phi, n):

    r_component = (1/R**2) * n * np.cos(n * Phi)
    phi_component = (1/R**2) * np.sin(n * Phi)

    x_component = r_component * np.cos(Phi) - phi_component * np.sin(Phi)
    y_component = r_component * np.sin(Phi) + phi_component * np.cos(Phi)

    return x_component, y_component



def main():
    
    m = 25
    n = 100

    s = np.s_[::m, n::n]

    r = np.linspace(3, 10, 1000)
    phi = np.linspace(0, 2*np.pi, 1000)

    R, Phi = np.meshgrid(r, phi)
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)

    xc1, yc1 = calc_grad_perp(R, Phi, 1)
    mag1 = np.sqrt(xc1**2 + yc1**2)
    xc1 /= mag1
    yc1 /= mag1

    xc2, yc2 = calc_grad_perp(R, Phi, 3)
    mag2 = np.sqrt(xc2**2 + yc2**2)
    xc2 /= mag2
    yc2 /= mag2

    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(5, 2), subplot_kw={'projection': 'polar'})

    norm = mpl.colors.Normalize(vmin=np.min(mag2), vmax=np.max(mag2))

    pcm1 = ax1.pcolormesh(Phi, R, norm(mag1))
    ax1.quiver(Phi[s], R[s], xc1[s], yc1[s], color='white')

    pcm2 = ax2.pcolormesh(Phi, R, norm(mag2))
    ax2.quiver(Phi[s], R[s], xc2[s], yc2[s], color='white')

    # fig.colorbar(pcm1, ax=ax1)
    fig.colorbar(pcm2, ax=[ax1, ax2])

    # style
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)

    fig.savefig('disclination_motion_figure.png')

    plt.show()


if __name__ == '__main__':
    main()
