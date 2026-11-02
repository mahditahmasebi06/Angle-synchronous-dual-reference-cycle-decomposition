"""Baseline representations, implemented with numpy/scipy only.
B1 raw statistics + FFT harmonics on the raw window
B2 DWT (db4, 6 levels) energy / entropy / std per channel  (classic wavelet pipeline)
B3 STFT-based synchrosqueezing (FSST) band energies + Renyi entropy
B4 MiniRocket-style random convolution kernels (PPV + max) with ridge classifier
B5 Teager-Kaiser energy operator on the cycle-difference (EMD-free TEO pipeline)
"""
import numpy as np
from scipy.signal import stft
from scipy.ndimage import correlate1d

FS = 4096.0
DB4 = np.array([-0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
                -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
                0.7148465705525415, 0.23037781330885523])          # pywt dec_lo


def _norm_raw(x):
    """Common 'naive' normalisation: each channel scaled by its own RMS of the first cycle."""
    s = np.sqrt((x[:82] ** 2).mean(0)) + 1e-6
    return x / s


def b1_stats(X):
    out = []
    for x in X:
        z = _norm_raw(x)
        f = []
        for c in range(7):
            v = z[:, c]
            sp = np.abs(np.fft.rfft(v * np.hanning(len(v))))
            fr = np.fft.rfftfreq(len(v), 1 / FS)
            h = [sp[np.argmin(abs(fr - 50 * k))] for k in (1, 2, 3, 5, 7)]
            f += [v.std(), np.abs(v).max(), ((v - v.mean()) ** 4).mean() / (v.var() + 1e-9) ** 2,
                  *np.log1p(h), np.log1p(sp[fr > 400].sum())]
        out.append(f)
    return np.array(out)


def dwt(v, levels=6, h=DB4):
    g = h[::-1] * (-1) ** np.arange(len(h))
    coeffs, a = [], v
    for _ in range(levels):
        ap = np.pad(a, (len(h), len(h)), mode='symmetric')
        lo = np.convolve(ap, h[::-1], 'same')[len(h):-len(h)][::2]
        hi = np.convolve(ap, g[::-1], 'same')[len(h):-len(h)][::2]
        coeffs.append(hi); a = lo
    coeffs.append(a)
    return coeffs


def b2_dwt(X):
    out = []
    for x in X:
        z = _norm_raw(x)
        f = []
        for c in range(7):
            cs = dwt(z[:, c])
            e = np.array([(cc ** 2).sum() for cc in cs]); p = e / e.sum()
            f += list(np.log1p(e)) + [-(p * np.log(p + 1e-12)).sum()] + [np.abs(cc).max() for cc in cs[:3]]
        out.append(f)
    return np.array(out)


def fsst(v, fs=FS, nper=128):
    """STFT-based synchrosqueezing (Oberlin/Auger FSST) with Gaussian window."""
    t = np.arange(nper) - nper / 2
    sig = nper / 8
    w = np.exp(-0.5 * (t / sig) ** 2)
    dw = -t / sig ** 2 * w
    f, tt, Z = stft(v, fs, window=w, nperseg=nper, noverlap=nper - 4, boundary=None, padded=False)
    _, _, Zd = stft(v, fs, window=dw, nperseg=nper, noverlap=nper - 4, boundary=None, padded=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        inst = f[:, None] - np.imag(Zd / Z) * fs / (2 * np.pi)
    df = f[1] - f[0]
    k = np.round(inst / df).astype(int)
    T = np.zeros_like(np.abs(Z))
    mask = (np.abs(Z) > 1e-6 * np.abs(Z).max()) & (k >= 0) & (k < len(f))
    cols = np.broadcast_to(np.arange(Z.shape[1]), Z.shape)
    np.add.at(T, (k[mask], cols[mask]), np.abs(Z[mask]))
    return f, tt, T


def b3_sst(X):
    out = []
    edges = [0, 30, 75, 125, 175, 275, 400, 800, 2048]
    for x in X:
        z = _norm_raw(x)
        feats = []
        for c in range(7):
            f, tt, T = fsst(z[:, c])
            E = T ** 2
            band = [E[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]
            p = E / (E.sum() + 1e-12)
            renyi = -np.log((p ** 3).sum() + 1e-30) / 2
            hf = E[f > 400].sum(0)
            feats += list(np.log1p(band)) + [renyi, np.log1p(hf.max()), (hf > 0.2 * hf.max()).mean()]
        out.append(feats)
    return np.array(out)


class MiniRocketLite:
    """Random dilated kernels (length 9, weights in {-1,2}) on multichannel input,
    PPV + max pooling. Close to MiniRocket in spirit, numpy implementation."""
    def __init__(self, n_kernels=1000, seed=0):
        self.n, self.rng = n_kernels, np.random.default_rng(seed)

    def fit(self, X):
        L = X.shape[1]
        self.k = []
        for _ in range(self.n):
            w = -np.ones(9); w[self.rng.choice(9, 3, replace=False)] = 2
            dil = int(2 ** self.rng.uniform(0, np.log2((L - 1) / 8)))
            ch = self.rng.choice(7, self.rng.integers(1, 4), replace=False)
            self.k.append((w, dil, ch))
        Z = self._conv(X[: min(len(X), 64)])
        self.bias = [np.quantile(z, self.rng.uniform(0.1, 0.9)) for z in Z]
        return self

    def _conv(self, X):
        Xn = np.stack([_norm_raw(x) for x in X])
        res = []
        for w, dil, ch in self.k:
            kern = np.zeros(8 * dil + 1); kern[::dil] = w
            res.append(correlate1d(Xn[:, :, ch].sum(-1), kern, axis=1, mode='constant'))
        return res

    def transform(self, X):
        Z = self._conv(X)
        out = np.empty((len(X), 2 * self.n))
        for i, (z, b) in enumerate(zip(Z, self.bias)):
            out[:, 2 * i] = (z > b).mean(1)
            out[:, 2 * i + 1] = z.max(1)
        return out


def b5_teo(X):
    """Cycle-difference residual + Teager-Kaiser energy (integer 82-sample delay)."""
    out = []
    for x in X:
        z = _norm_raw(x)
        r = z[82:] - z[:-82]
        f = []
        for c in range(7):
            v = r[:, c]
            psi = v[1:-1] ** 2 - v[:-2] * v[2:]
            a = np.abs(psi)
            f += [np.log1p(a.max()), np.log1p(a.mean()), (a > 0.1 * a.max()).mean(),
                  np.log1p(np.abs(v).max()), np.log1p(v.std())]
        out.append(f)
    return np.array(out)


def b1_stats_fs(X, fs):
    """B1 for an arbitrary sampling rate (first-cycle RMS scaling over round(fs/50) samples)."""
    L = int(round(fs / 50.0)); out = []
    for x in X:
        z = x / (np.sqrt((x[:L] ** 2).mean(0)) + 1e-6)
        f = []
        for c in range(7):
            v = z[:, c]
            sp = np.abs(np.fft.rfft(v * np.hanning(len(v))))
            fr = np.fft.rfftfreq(len(v), 1 / fs)
            h = [sp[np.argmin(abs(fr - 50 * k))] for k in (1, 2, 3, 5, 7)]
            f += [v.std(), np.abs(v).max(), ((v - v.mean()) ** 4).mean() / (v.var() + 1e-9) ** 2,
                  *np.log1p(h), np.log1p(sp[fr > 400].sum())]
        out.append(f)
    return np.array(out)
