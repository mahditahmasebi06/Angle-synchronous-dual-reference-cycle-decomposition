"""
Angle-Synchronous Dual-Reference Cycle Decomposition (AS-DRCD)
for incipient-fault waveform analysis (Xiong et al. dataset, 4096 Hz, 50 Hz, 81.92 samples/cycle).

Pipeline per record x[n, ch]  (ch = Ia Ib Ic In Va Vb Vc)
 1. robust fundamental-frequency estimate (cycle-to-cycle phasor rotation, median)
 2. angle-domain (order-tracked) resampling to exactly M samples per cycle  -> C[ch, k, m]
 3. cycle-novelty map N_k = C_k - C_{k-1}   (exact one-cycle delay; no fractional-delay leakage)
 4. change-point onset k0 on the novelty energy (MAD-thresholded CUSUM)
 5. robust pre-event template T = median(C_0..C_{k0-1}); superimposed map S_k = C_k - T
 6. symmetric-component self-calibration of voltages (per-phase gains from pre-event phasors)
 7. Cycle-Angle Residual Map (CARM) descriptors from S and N
"""
import numpy as np
from scipy.interpolate import CubicSpline

FS = 4096.0
F0 = 50.0
M = 128                    # samples per cycle after angle resampling
A = np.exp(2j * np.pi / 3)


# ----------------------------------------------------------------- frequency
def _est_freq(v, fs=FS, f0=F0, iters=3):
    """Median cycle-to-cycle rotation of the positive-sequence voltage phasor.
    Positive sequence is used because zero-sequence oscillations (neutral-voltage beating
    in resonant-earthed networks) bias per-phase phasor rotation."""
    f = f0
    n = np.arange(v.shape[0])
    for _ in range(iters):
        L = int(round(fs / f))
        ph = np.exp(-2j * np.pi * f * n / fs)
        Z = np.stack([np.convolve(v[:, ch] * ph, np.ones(L), 'valid')[::L] for ch in range(3)])
        if Z.shape[1] < 3:
            return f0
        _, z1, z2 = seq(Z)
        z = z1 if np.abs(z1).sum() >= np.abs(z2).sum() else z2
        r = z[1:] * np.conj(z[:-1])
        w = np.abs(r)
        keep = w > 0.8 * w.max()
        df = np.median(np.angle(r[keep])) / (2 * np.pi * L / fs)
        f = f + np.clip(df, -1.0, 1.0)
    return float(f) if 47.5 <= f <= 52.5 else f0


def estimate_frequency(v, fs=FS, f0=F0):
    return _est_freq(v, fs, f0)


# ----------------------------------------------------------------- resampling
def angle_resample(x, f, fs=FS, m=M):
    """Resample to an exact integer number of samples per fundamental cycle.
    Returns C[ch, K, m] (K full cycles) and the time vector used."""
    t = np.arange(x.shape[0]) / fs
    K = int(np.floor(t[-1] * f))
    tq = np.arange(K * m) / (m * f)
    cs = CubicSpline(t, x, axis=0)
    xr = cs(tq)                                  # (K*m, ch)
    return xr.T.reshape(x.shape[1], K, m), tq


def naive_cycles(x, fs=FS, f0=F0):
    """Ablation: integer-sample cycles (82 samples) without order tracking."""
    L = int(round(fs / f0))
    K = x.shape[0] // L
    return x[:K * L].T.reshape(x.shape[1], K, L)


# ----------------------------------------------------------------- phasors
def cycle_phasor(C):
    """Fundamental phasor of each cycle: (ch, K) complex, peak amplitude scale."""
    m = C.shape[-1]
    e = np.exp(-2j * np.pi * np.arange(m) / m)
    return 2.0 / m * (C * e).sum(-1)


# ----------------------------------------------------------------- onset
def detect_onset(C, chans=(0, 1, 2, 3), min_pre=1):
    """Change point on cycle-novelty energy of currents (and neutral)."""
    N = np.diff(C[list(chans)], axis=1)                  # (c, K-1, m)
    base = np.sqrt((C[list(chans)] ** 2).mean(-1)).max(0) + 1e-9  # cycle RMS envelope
    en = np.sqrt((N ** 2).mean(-1)).sum(0)              # novelty RMS summed over channels
    ref = np.median(en[:3]) if len(en) >= 3 else en[0]
    mad = np.median(np.abs(en[:3] - ref)) + 1e-9
    thr = max(ref + 6 * 1.4826 * mad, 3 * ref, 0.02 * np.median(base))
    idx = np.where(en > thr)[0]
    if len(idx) == 0:
        k0 = int(np.argmax(en)) + 1
    else:
        k0 = int(idx[0]) + 1                              # novelty k compares cycle k+1 vs k
    return max(k0, min_pre), en, thr


# ----------------------------------------------------------------- calibration
def ll_calibrate(Vph, seq_fix=True):
    """Neutral-shift-invariant self-calibration.
    Unknown per-channel gains g (real>0). True phase voltages may contain a neutral
    displacement Vn (single-phase earth faults in non-effectively earthed MV networks),
    but line-to-line voltages E_p - E_q stay balanced.  Choose g_b/g_a, g_c/g_a that
    null the negative-sequence of the reconstructed line voltages:
        min_g |V2(LL)|^2 / |V1(LL)|^2.
    Returns gains scaled to 1 pu phase voltage, the phase order, and the residual."""
    from scipy.optimize import minimize
    order = np.array([0, 1, 2])
    V = Vph.copy()
    if seq_fix:
        _, v1, v2 = seq(V[:, None])
        if abs(v2[0]) > abs(v1[0]):          # ACB labelled records -> swap b, c
            order = np.array([0, 2, 1]); V = V[order]
    mag = np.abs(V) + 1e-9
    g0 = np.log(mag.mean() / mag)

    def ll(g):
        u = np.exp(g) * V
        L = np.array([u[0] - u[1], u[1] - u[2], u[2] - u[0]])
        _, l1, l2 = seq(L[:, None])
        return l1[0], l2[0]

    def obj(p):
        l1, l2 = ll(np.r_[0.0, p])
        return abs(l2) ** 2 / (abs(l1) ** 2 + 1e-12) + 1e-4 * np.sum((np.r_[0.0, p] - g0 + g0[0]) ** 2)

    # Gain ratios are bounded (|log ratio| <= log 4): with a solid earth fault the set with one
    # channel gain -> 0 also balances the line voltages, a degenerate root outside any real VT
    # mismatch. Multi-start inside the box, keep the best.
    lim = np.log(4.0); bnds = [(-lim, lim)] * 2
    starts = [np.clip((g0 - g0[0])[1:], -lim, lim)] + [np.array([a, b]) for a in (-0.7, 0, 0.7) for b in (-0.7, 0, 0.7)]
    best = None
    for x0 in starts:
        r_ = minimize(obj, x0, method='L-BFGS-B', bounds=bnds, options=dict(ftol=1e-14, gtol=1e-10))
        if best is None or r_.fun < best.fun - 1e-12:
            best = r_
    g = np.exp(np.r_[0.0, best.x])
    l1, l2 = ll(np.log(g))
    g = g * np.sqrt(3) / (abs(l1) + 1e-12)      # |V1_LL| = sqrt(3) pu  ->  phase 1 pu
    gg = np.empty(3); gg[order] = g             # gains in original channel order
    return gg, order, abs(l2) / (abs(l1) + 1e-12)


def voltage_gains(Vph):
    """Vph: (3,) complex pre-event phasors. Gains that equalise magnitudes (balanced
    pre-event assumption) and express voltages in per-unit of the positive-sequence."""
    mag = np.abs(Vph) + 1e-9
    g = mag.mean() / mag
    return g / mag.mean()          # => |g_p V_p| = 1 pu


def seq(P):
    """Symmetrical components of a (3, ...) phasor array -> (V0, V1, V2)."""
    a, b, c = P
    return ((a + b + c) / 3, (a + A * b + A * A * c) / 3, (a + A * A * b + A * c) / 3)


# ----------------------------------------------------------------- main decomposition
def decompose(x, use_angle=True, calibrate=True, fs=FS):
    x = np.asarray(x, float)
    if use_angle:
        f = estimate_frequency(x[:, 4:7], fs)
        if not np.isfinite(f):
            f = F0
        C, _ = angle_resample(x, f, fs)
    else:
        f = F0
        C = naive_cycles(x, fs)
    K = C.shape[1]
    k0, nov_en, thr = detect_onset(C)
    k0 = min(k0, K - 2)
    P = cycle_phasor(C)                                    # (7, K)
    # reference window: pre-event cycles, or (energisation case) the last cycles
    pre_rms = np.sqrt((C[4:7, :max(k0, 1)] ** 2).mean())
    post_rms = np.sqrt((C[4:7, -3:] ** 2).mean())
    energised = pre_rms < 0.2 * post_rms
    ref = slice(0, max(k0, 1)) if not energised else slice(K - 3, K)
    T = np.median(C[:, ref], axis=1)                       # (7, m) template
    # voltage calibration
    Vref = P[4:7, ref].mean(1)
    vorder = np.array([0, 1, 2]); cal_res = np.nan
    if calibrate == 'll':
        g, vorder, cal_res = ll_calibrate(Vref)
    elif calibrate:                              # magnitude equalisation (common practice)
        g = voltage_gains(Vref)
    else:
        g = np.ones(3) / (np.abs(Vref).mean() + 1e-9)
    C = C.copy(); T = T.copy(); P = P.copy()
    C[4:7] *= g[:, None, None]; T[4:7] *= g[:, None]; P[4:7] *= g[:, None]
    # current scaling: per-unit of pre-event load current (floor to avoid div by ~0)
    Iref = np.abs(P[0:3, ref]).mean()
    Ibase = max(Iref, 5.0)
    S = C - T[:, None, :]                                  # superimposed map
    N = np.concatenate([np.zeros_like(C[:, :1]), np.diff(C, axis=1)], axis=1)  # novelty map
    # broadband noise floor of the novelty map, from its high orders (30..M/2) in the
    # median cycle, extrapolated to the full band (white-noise assumption)
    m = C.shape[-1]
    Pn = np.abs(np.fft.rfft(N[:, 1:], axis=-1)) ** 2 / m ** 2 * 2     # (ch, K-1, bins)
    hf = slice(30, m // 2 + 1)
    noise_ms = np.median(Pn[:, :, hf].sum(-1), axis=1) * (m // 2 + 1) / (m // 2 + 1 - 30)
    return dict(C=C, T=T, S=S, N=N, noise_ms=noise_ms, P=P, k0=k0, f=f, g=g, Ibase=Ibase, Iref=Iref,
                energised=energised, ref=ref, vorder=vorder, cal_res=cal_res, nov_en=nov_en, thr=thr)
