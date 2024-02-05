import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

def main():
    
    Lx1, Lx2 = -2, 5
    Ly1, Ly2 = -2, 5
    n = 8
    q = 0.5
    scale = 15.0
    width = 0.0035

    angle_r = 1.5
    label_delta = 0.5

    sample_x_idx = -4
    sample_y_idx = -3

    fontsize = 16.0
    
    theta_r = 3.0

    x = np.linspace(Lx1, Lx2, n)
    y = np.linspace(Ly1, Ly2, n)

    n = [x[sample_x_idx], y[sample_y_idx]]

    X, Y = np.meshgrid(x, y, indexing='ij')
    Phi = np.arctan2(Y, X)
    Theta = q * Phi

    fig, ax = plt.subplots()
    ax.quiver(X, Y, np.cos(Theta), np.sin(Theta),
              pivot='mid', headlength=0, headaxislength=0,
              scale=scale, width=width)

    ax.plot(0, 0, marker='o', color='black')
    ax.plot([0, n[0]], [0, n[1]], color='black')

    theta2 = np.arctan2(n[1], n[0])
    arc = mpl.patches.Arc((0, 0), 2*angle_r, 2*angle_r, theta2=theta2*180/np.pi)
    ax.add_patch(arc)

    text_coords = [(angle_r + label_delta) * np.cos(theta2/2), 
                   (angle_r + label_delta) * np.sin(theta2/2)]
    ax.text(text_coords[0], text_coords[1], r'$\varphi$', ha='center', va='center',
            bbox=dict(facecolor='white', boxstyle='Square, pad=0', edgecolor='none'),
            fontsize=fontsize)

    theta0 = Theta[sample_x_idx, sample_y_idx]
    m = [n[0] + theta_r * np.cos(theta0), n[1] + theta_r * np.sin(theta0)]
    ax.plot([n[0], m[0]], [n[1], m[1]], color='black')
    ax.plot([n[0], m[0]], [n[1], n[1]], ls='--', color='black')

    arc = mpl.patches.Arc((n[0], n[1]), 2*angle_r, 2*angle_r, theta2=theta0*180/np.pi)
    ax.add_patch(arc)

    text_coords = [n[0] + (angle_r + label_delta) * np.cos(theta0/2), 
                   n[1] + (angle_r + label_delta) * np.sin(theta0/2)]
    ax.text(text_coords[0], text_coords[1], r'$\theta$', ha='center', va='center',
            bbox=dict(facecolor='white', boxstyle='Square, pad=0', edgecolor='none'),
            fontsize=fontsize)

    text_coords = [n[0]/2 - 0.1, 
                   n[1]/2]
    ax.text(text_coords[0], text_coords[1], r'$r$', ha='right', va='bottom',
            bbox=dict(facecolor='white', boxstyle='Square, pad=0', edgecolor='none'),
            fontsize=fontsize)

    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

    plt.show()


if __name__ == '__main__':
    main()
