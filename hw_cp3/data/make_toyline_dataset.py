import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(x):
    return -0.1 * x

def make_xy_dataset(N=10, noise_stddev=0.1, seed=0):
    prng = np.random.RandomState(seed)
    x_N = prng.uniform(low=-2.0, high=2.0, size=N)
    y_N = f(x_N) + noise_stddev * prng.randn(N)
    return x_N, y_N


def save_dataset_to_csv(filename, x_N, y_N):
    df = pd.DataFrame()
    df['x'] = x_N
    df['y'] = y_N
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    x_G = np.linspace(0, 1, 1001)

    # Make test dataset
    x_te_N, y_te_N = make_xy_dataset(N=512, seed=201)
    save_dataset_to_csv('toyline_test.csv', x_te_N, y_te_N)

    # Make train dataset
    x_N, y_N = make_xy_dataset(N=512, seed=101)
    save_dataset_to_csv('toyline_train.csv', x_N, y_N)

    fig_h, ax_grid = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
    for ax, N in zip(ax_grid, [0, 8, 64, 512]):
        ax.plot(x_N[:N], y_N[:N], 'k.')
        ax.plot(x_G, f(x_G), 'g-')

    plt.show()
