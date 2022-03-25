import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import logsumexp
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from RandomWalkSampler import RandomWalkSampler2D

def calc_model_log_pdf(w, logsigma, x_N, t_N):
    ''' Compute log pdf of all modeled random variables

    Args
    ----
    w : scalar float
    logsigma : scalar float
    x_N : 1D array, shape (N,)
    t_N : 1D array, shape (N,)

    Returns
    -------
    logpdf : float real scalar
        Log probability density function value at provided input
    '''
    log_pdf_t = np.sum(scipy.stats.norm.logpdf(t_N, w * x_N, np.exp(logsigma)))

    pi_w = np.asarray([0.8, 0.2])
    mu_w = np.asarray([0.0, 0.0])
    sigma_w = np.asarray([0.01, 1.0])
    log_pdf_w = logsumexp(
        np.log(pi_w) + scipy.stats.norm.logpdf(w, mu_w, sigma_w))

    log_pdf_logsigma = scipy.stats.norm.logpdf(logsigma, 0.0, 2.0)
    return log_pdf_t + log_pdf_w + log_pdf_logsigma


if __name__ == '__main__':
    n_samples = 40000
    n_keep = 30000

    N_grid = np.asarray([0, 1, 4, 512])
    rw_stddev_grid = np.asarray([0.5, 0.3, 0.1, 0.05])
    seed_grid = np.asarray([101, 101, 101, 101])
    G = len(N_grid)

    train_df = pd.read_csv("../data/toyline_train.csv")
    x_N = train_df['x'].values
    t_N = train_df['y'].values
    test_df = pd.read_csv("../data/toyline_test.csv")
    x_test_N = test_df['x'].values
    t_test_N = test_df['y'].values

    z_initA_D = np.asarray([0.02, 0.02])
    z_initB_D = np.asarray([1.0, -1.0])

    do_just_read_saved_results_if_available = False
    col_names = ['w', 'logsigma', 'did_accept_S']
    samples_csv_path_pattern = "problem3_results3/samples-N={N}-S={S}-rw_stddev={rw_stddev}-init_name={init_name}-seed={seed}.csv"

    H = 3 # height of one panel
    W = 2 # width of one panel
    _, ax_grid = plt.subplots(
        nrows=2, ncols=G, sharex=True, sharey=True,
        figsize=(W*G, H*2))
    name2color = {"A":"r", "B":"b"}
    # Loop over each column of plots

    for col_id, (N, rw_stddev, seed) in enumerate(zip(N_grid, rw_stddev_grid, seed_grid)):

        def calc_target_log_pdf(z_D):
            return calc_model_log_pdf(z_D[0], z_D[1], x_N[:N], t_N[:N])

        for row_id, (init_name, zinit_D) in enumerate(zip(['A', 'B'], [z_initA_D, z_initB_D])):

            csvpath = samples_csv_path_pattern.format(
                N=N, S=n_keep, rw_stddev=rw_stddev, init_name=init_name, seed=seed)

            if os.path.exists(csvpath) and do_just_read_saved_results_if_available:
                csv_df = pd.read_csv(csvpath)

            else:
                sampler = RandomWalkSampler2D(calc_target_log_pdf, rw_stddev, seed)
                z_list, sampler_info = sampler.draw_samples(zinit_D=zinit_D, n_samples=n_samples)
                z_SD = np.vstack(z_list[-n_keep:])
                csv_df = pd.DataFrame(z_SD.copy(), columns=col_names[:-1])
                csv_df['did_accept_S'] = sampler_info['did_accept_S'][-n_keep:].copy()
                csv_df.to_csv(csvpath, index=False)

            z_SD = csv_df[col_names].values
            accept_rate = np.mean(csv_df['did_accept_S'])

            ax_grid[row_id,col_id].plot(z_SD[:,0], z_SD[:,1], '.', alpha=0.05, color=name2color[init_name])

            ax_grid[0,col_id].set_title('N=%d' % N)
            ax_grid[1,col_id].set_xlabel('w')
            if col_id == 0:
                ax_grid[0,col_id].set_ylabel('$\\log \\sigma$')
                ax_grid[1,col_id].set_ylabel('$\\log \\sigma$')

            print("N %d | rw_stddev %7.3g | init_name %s | accept_rate (after burnin) %.3f" % (
                N, rw_stddev, init_name, accept_rate
                ))


    for ax in ax_grid.flatten():
        ax.set_xlim([-3, 3]);
        ax.set_ylim([-7, 5]);
        ax.set_aspect('equal', 'box');
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-6, -4, -2, 0, 2, 4])
    plt.tight_layout()
    plt.savefig("problem3_posterior_samples.pdf", bbox_to_inches='tight', pad_inches=0)
    plt.show()

    