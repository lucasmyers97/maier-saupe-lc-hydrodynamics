import argparse
import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('science')
mpl.rcParams['figure.dpi'] = 300

from scipy.optimize import curve_fit



def read_commandline_args():


    description = "Plot defect annihilation coalescence times vs. anisotropy"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='folder where coalescence time data lives')
    parser.add_argument('--data_filename', dest='data_filename',
                        help='name of file with coalescence time data')
    parser.add_argument('--output_folder', dest='output_folder',
                        help='folder which will hold output plots')
    parser.add_argument('--plot_filename', dest='plot_filename',
                        help='filename of t_c vs. epsilon plot')
    args = parser.parse_args()

    output_folder = args.output_folder
    if not output_folder:
        output_folder = args.data_folder

    data_filename = os.path.join(args.data_folder, args.data_filename)
    plot_filename = os.path.join(output_folder, args.plot_filename)

    return data_filename, plot_filename



def power_law(eps, a, lam):

    # return np.abs(eps)**lam
    return a * eps**lam



def power_law_jac(eps, lam):

    # return 0.5 * np.abs(eps)**lam * np.log(eps**2)
    return eps**lam



def fit_power_law(eps, t_c):

    popt, pcov = curve_fit(power_law, eps, t_c, p0=[2.0], jac=power_law_jac)
    return popt



def main():

    data_filename, plot_filename = read_commandline_args()
    data = pd.read_excel(data_filename)

    eps = data['eps'].values
    t_f = data['t_f'].values

    eps_mod = np.copy(eps)
    t_f_mod = np.copy(t_f)
    neg_eps_idx = np.where(eps_mod < 0)[0]
    zero_eps_idx = np.where(eps_mod == 0)[0][0]

    eps_mod[neg_eps_idx] *= -1
    eps_mod = np.delete(eps_mod, zero_eps_idx)
    t_f_mod -= t_f[zero_eps_idx]
    t_f_mod = np.delete(t_f_mod, zero_eps_idx)

    popts, _ = curve_fit(power_law, eps_mod, t_f_mod, p0=[8000, 2.0])

    eps_ref = np.linspace(-np.max(eps_mod), np.max(eps_mod), num=1000)
    t_f_ref = popts[0] * np.abs(eps_ref)**popts[1] + t_f[zero_eps_idx]

    fig, ax = plt.subplots()

    ax.plot(eps, t_f, linestyle='', marker='o')
    fit_plot, = ax.plot(eps_ref, t_f_ref)

    ax.legend([fit_plot],
              [(r'$A | \epsilon |^\lambda + B$,\\'
                  r'$A = {:.2e}$,\\'
                  r'$\lambda = {:.2e}$,\\'
                  r'$B = {:.2e}$').format(popts[0], popts[1], t_f[zero_eps_idx])])
    ax.set_title(r'Coalescence time vs. anisotropy for $R_0 = 40$')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$t_c$')
    fig.tight_layout()

    fig.savefig(plot_filename)

    plt.show()

    # Check power in a different way
    pos_eps_idx = np.where(eps >= 0)[0]
    neg_eps_idx = np.where(eps <= 0)[0]

    popts_pos, _ = curve_fit(power_law, 
                             eps[pos_eps_idx], 
                             t_f[pos_eps_idx] - t_f[zero_eps_idx], 
                             p0=[8000, 2.0])

    popts_neg, _ = curve_fit(power_law, 
                             -eps[neg_eps_idx], 
                             t_f[neg_eps_idx] - t_f[zero_eps_idx], 
                             p0=[8000, 2.0])

    print(popts_pos)
    print(popts_neg)



if __name__ == "__main__":
    main()
