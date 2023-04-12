"""
Module provides utility funtions for interfacing with and transforming
quantities associated with nematics, including Q-tensors, directors, director
angles and the like.
"""

import numpy as np

def vec_dim(dim):

    if (dim == 2):
        return 5
    elif (dim == 3):
        return 5

def Q_vec_to_mat(Q_vec):

    n_pts = Q_vec.shape[0]
    dim = 3

    Q_mat = np.zeros((n_pts, dim, dim))
    Q_mat[:, 0, 0] = Q_vec[:, 0]
    Q_mat[:, 1, 0] = Q_vec[:, 1]
    Q_mat[:, 2, 0] = Q_vec[:, 2]
    Q_mat[:, 1, 1] = Q_vec[:, 3]
    Q_mat[:, 2, 1] = Q_vec[:, 4]
    Q_mat[:, 0, 1] = Q_mat[:, 1, 0]    
    Q_mat[:, 0, 2] = Q_mat[:, 2, 0]    
    Q_mat[:, 1, 2] = Q_mat[:, 2, 1]    
    Q_mat[:, 2, 2] = -(Q_mat[:, 0, 0] + Q_mat[:, 1, 1])

    return Q_mat

def eigensystem_from_Q(Q):

    n_pts = Q.shape[0]
    dim = Q.shape[1]
    q1 = np.zeros(n_pts)
    q2 = np.zeros(n_pts)
    n = np.zeros((n_pts, dim))
    m = np.zeros((n_pts, dim))

    for i in range(Q.shape[0]):
        w, v = np.linalg.eigh(Q[i, :, :])
        q1[i] = w[-1]
        q2[i] = w[-2]
        n[i, :] = v[:, -1]
        m[i, :] = v[:, -2]

    return q1, q2, n, m



def sanitize_director_angle(phi):

    new_phi = np.copy(phi)

    # Fix jumps of over |pi/2| until there are no more jumps
    while True:

        dphi = np.diff(new_phi)

        jump_down_indices = np.nonzero(dphi < (-np.pi/2))
        jump_up_indices = np.nonzero(dphi > (np.pi/2))

        if (not jump_down_indices[0].size) and (not jump_up_indices[0].size):
            break

        for jump_index in jump_down_indices:
            if jump_index.shape[0] == 0:
                continue
            new_phi[(jump_index[0] + 1):] += np.pi

        for jump_index in jump_up_indices:
            if jump_index.shape[0] == 0:
                continue
            new_phi[(jump_index[0] + 1):] -= np.pi

    # Make sure mean is in range [0, pi]
    while True:

        mean = np.mean(new_phi)
        if (mean < -1e-10):
            new_phi += np.pi
        elif (mean > (np.pi + 1e-10)):
            new_phi -= np.pi
        else:
            break

    return new_phi



def director_to_angle(n):

    phi = np.arctan2(n[:, 1], n[:, 0])
    return phi



def calc_eps(L3, S=0.6751, L2=0.0):

    return L3 * S / (2 + L2 + (1 / 3) * L3 * S)



def calc_L3(eps, S=0.6751, L2=0.0):

    return (eps / S) * (2 + L2) / (1 - eps / 3)



def calc_L2(eps, L3, S=0.6751):

    return ( L3 * S * (1 - eps / 3) - 2 * eps ) / eps



def split_defect_centers_by_charge(charge, t, x, y):

    pos_idx = np.nonzero(charge > 0)
    neg_idx = np.nonzero(charge < 0)

    pos_t = t[pos_idx]
    neg_t = t[neg_idx]

    pos_centers = np.vstack( (x[pos_idx], y[pos_idx]) ).transpose()
    neg_centers = np.vstack( (x[neg_idx], y[neg_idx]) ).transpose()

    return pos_t, neg_t, pos_centers, neg_centers



def match_times_to_points(times, t, x, y):

    n_coords = 2
    points = np.zeros((times.shape[0], n_coords))
    for i, time in enumerate(times):
        t_idx = np.argmin(np.abs(t - time))
        points[i, 0] = x[t_idx]
        points[i, 1] = y[t_idx]

    return points



def get_other_polar_angle(theta, r, d):
    """
    Note: d > 0 if other defect is to the left, d < 0 if other defect is to the right
    """
    return np.arctan2(r * np.sin(theta), d + r * np.cos(theta))



def pairwise_defect_director_near_defect(theta, r, d, m):
    """
    This function gets you director angle near one defect, given the influence
    of anther defect.
    Note: d > 0 if other defect is to the left, d < 0 if other defect is to the right
    """
    theta2 = get_other_polar_angle(theta, r, d)
    return m * theta - m * theta2



def pairwise_defect_director_at_midpoint(theta, r, d, m1, m2):
    """
    This function gets you director angle from the midpoint between two defects
    Note: m1 is to the right, and m2 is to the left. 
    """
    theta1 = np.arctan2(r * np.sin(theta), r * np.cos(theta) - 0.5 * d)
    theta2 = np.arctan2(r * np.sin(theta), r * np.cos(theta) + 0.5 * d)

    return m1 * theta1 + m2 * theta2



def get_d_from_defect_positions(x, y, t, charge, defect_charge, time, dt):
    """
    Gets distance between two defects. If the defect_charge that's passed in is
    on the left, d will be negative, otherwise it will be positive.
    Note: this function assumes both defects lie on the x-axis.
    """

    (pos_t, neg_t, 
     pos_centers, neg_centers) = split_defect_centers_by_charge(charge, t, x, y)

    pos_center = match_times_to_points(np.array([time * dt]), pos_t, 
                                       pos_centers[:, 0], pos_centers[:, 1])
    neg_center = match_times_to_points(np.array([time * dt]), neg_t, 
                                       neg_centers[:, 0], neg_centers[:, 1])

    if defect_charge > 0:
        return pos_center[0, 0] - neg_center[0, 0]
    else:
        return neg_center[0, 0] - pos_center[0, 0]

