"""
Module provides utility funtions for interfacing with and transforming
quantities associated with nematics, including Q-tensors, directors, director
angles and the like.
"""

import numpy as np

def sanitize_director_angle(phi):

    dphi = np.diff(phi)
    jump_down_indices = np.nonzero(dphi < (-np.pi/2))
    jump_up_indices = np.nonzero(dphi > (np.pi/2))

    new_phi = np.copy(phi)
    for jump_index in jump_down_indices:
        if jump_index.shape[0] == 0:
            continue
        new_phi[(jump_index[0] + 1):] += np.pi

    for jump_index in jump_up_indices:
        if jump_index.shape[0] == 0:
            continue
        new_phi[(jump_index[0] + 1):] -= np.pi

    return new_phi



def director_to_angle(n):

    phi = np.arctan2(n[:, 1], n[:, 0])
    return phi



def calc_eps(L3, S=0.6751):

    return L3 * S / (2 + (1 / 3) * L3 * S)



def calc_L3(eps, S=0.6751):

    return 2 * eps / ( S * (1 - (1/3) * eps) )



def split_defect_centers_by_charge(charge, t, x, y):

    pos_idx = np.nonzero(charge > 0)
    neg_idx = np.nonzero(charge < 0)

    pos_t = t[pos_idx]
    neg_t = t[neg_idx]

    pos_centers = np.vstack( (x[pos_idx], y[pos_idx]) ).transpose()
    neg_centers = np.vstack( (x[neg_idx], y[neg_idx]) ).transpose()

    return pos_t, neg_t, pos_centers, neg_centers
