import numpy as np
from scipy.interpolate import CubicSpline

def add_noise(Xs, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    p = (Xs ** 2).mean(1, keepdims=True)
    return Xs + rng.standard_normal(Xs.shape) * np.sqrt(p / 10 ** (snr_db / 10))


def shift_freq(Xs, f_new):
    """time-scale the record so a 50 Hz system looks like f_new Hz at 4096 Hz sampling"""
    n = Xs.shape[1]; t = np.arange(n)
    out = np.empty_like(Xs)
    for i, x in enumerate(Xs):
        cs = CubicSpline(t, x, axis=0)
        tq = np.clip(t * f_new / 50.0, 0, n - 1)
        out[i] = cs(tq)
    return out


