"""Create an easy-to-visualize non-linear classification dataset."""
import numpy as np


def make_moons(n=600, noise=0.12, seed=42):
    if n < 2:
        raise ValueError('n must be at least 2')
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0, np.pi, n0)
    t1 = rng.uniform(0, np.pi, n1)
    top = np.column_stack([np.cos(t0), np.sin(t0)])
    bottom = np.column_stack([1 - np.cos(t1), 0.5 - np.sin(t1)])
    x = np.vstack([top, bottom]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n0), np.ones(n1)].astype(int)
    perm = rng.permutation(n)
    return x[perm], y[perm]
