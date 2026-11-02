"""CARM descriptors computed from the AS-DRCD decomposition (see carm.py).
All phase-wise quantities are aggregated with permutation-invariant statistics
(max / mean / min over phases) so the faulted-phase label never enters the model."""
import numpy as np
import carm

EPS = 1e-9


def _runs(mask):
    """lengths of consecutive True runs"""
    r, c = [], 0
    for v in mask:
        if v:
            c += 1
        elif c:
            r.append(c); c = 0
    if c:
        r.append(c)
    return r


def carm_features(x, use_angle=True, calibrate='ll', dual=True, profile=True, denoise=True, fs=carm.FS,
                  nf_band='inband', k0_fix=None):
    d = carm.decompose(x, use_angle=use_angle, calibrate=calibrate, fs=fs, nf_band=nf_band, k0_fix=k0_fix)
    C, S, N, P, T = d['C'], d['S'], d['N'], d['P'], d['T']
    K, m = C.shape[1], C.shape[2]
    k0, Ib = d['k0'], d['Ibase']
    pre = slice(0, max(k0, 1)) if not d['energised'] else slice(K - 3, K)
    F = {}

    # ---------------- per-cycle residual energies (pu of pre-event load current)
    nz = d['noise_ms'][:, None] if denoise else np.zeros((7, 1))
    eS = np.sqrt(np.clip((S[0:4] ** 2).mean(-1) - nz[0:4] / 2, 0, None)) / Ib   # (4, K): Ia Ib Ic In
    eN = np.sqrt(np.clip((N[0:4] ** 2).mean(-1) - nz[0:4], 0, None)) / Ib
    cI = np.sqrt((C[0:3] ** 2).mean(-1)) / Ib        # cycle RMS of phase currents
    eSp, eNp = eS[:3].max(0), eN[:3].max(0)          # worst phase per cycle
    noise = np.median(eNp[1:max(k0, 2)]) if k0 > 1 else np.median(eNp[1:])
    tau = max(4 * noise, 0.05)
    post = slice(k0, K)

    # 1. duration / recurrence (novelty = change, superimposed = deviation from pre-event)
    act_N = eNp[post] > tau
    act_S = eSp[post] > tau
    rN, rS = _runs(act_N), _runs(act_S)
    F['nov_cycles'] = act_N.sum()
    F['sup_cycles'] = act_S.sum()
    F['nov_bursts'] = len(rN)
    F['sup_bursts'] = len(rS)
    F['sup_longest'] = max(rS) if rS else 0
    idx = np.where(act_N)[0]
    F['nov_span'] = (idx[-1] - idx[0] + 1) if len(idx) else 0
    F['sup_frac_tail'] = act_S[-4:].mean() if len(act_S) >= 4 else act_S.mean()
    if dual:   # dual-reference contrast: persistent change (S high, N low) vs transient
        F['dual_ratio_tail'] = eSp[-3:].mean() / (eNp[-3:].mean() + tau)
        F['dual_ratio_peak'] = eSp.max() / (eNp.max() + EPS)
        F['dual_lag'] = int(np.argmax(eSp)) - int(np.argmax(eNp))

    # 2. persistence
    F['sup_tail_over_peak'] = eSp[-3:].mean() / (eSp.max() + EPS)
    F['I_post_over_pre'] = cI[:, -3:].mean() / (cI[:, pre].mean() + EPS)
    F['I_peakcycle_over_pre'] = cI.max() / (cI[:, pre].mean() + EPS)

    # 3. magnitude
    F['sup_peak_pu'] = np.log10(eSp.max() + EPS)
    F['nov_peak_pu'] = np.log10(eNp.max() + EPS)
    F['abs_peak_A'] = np.log10(np.abs(S[0:3]).max() + 1)
    F['Ibase_A'] = np.log10(d['Iref'] + 1)
    F['n_phases'] = (eS[:3].max(1) > 0.3 * eS[:3].max()).sum()
    F['In_over_phase'] = eS[3].max() / (eS[:3].max() + EPS)
    F['In_peak_pu'] = np.log10(eS[3].max() + EPS)

    # 4. waveform shape of the residual in its peak cycle (worst phase)
    pk = int(np.argmax(eSp)); ph = int(np.argmax(eS[:3, pk]))
    s = S[ph, pk]
    rms = np.sqrt((s ** 2).mean()) + EPS
    F['crest'] = np.abs(s).max() / rms
    F['kurt'] = ((s - s.mean()) ** 4).mean() / (s.var() + EPS) ** 2
    ep, en = (s[s > 0] ** 2).sum(), (s[s < 0] ** 2).sum()
    F['halfcycle_asym'] = abs(ep - en) / (ep + en + EPS)
    F['dc_frac'] = abs(s.mean()) / rms
    # sub-cycle duration of the residual burst (in ms) on the concatenated worst-phase S
    sw = S[ph].ravel()
    env = np.abs(sw)
    thr = max(0.25 * env.max(), 3 * np.median(env[: max(k0, 1) * m]) + EPS)
    above = env > thr
    # duration = samples between first and last crossing within the strongest run of cycles
    ia = np.where(above)[0]
    dur = (ia[-1] - ia[0]) if len(ia) else 0
    f = d['f']
    F['burst_ms'] = 1000.0 * dur / (m * f)
    F['active_ms'] = 1000.0 * above.sum() / (m * f)
    F['active_ratio'] = above.sum() / (dur + 1)

    # 5. inception angle relative to the worst-phase pre-event voltage
    pkN = int(np.argmax(eNp)); phN = int(np.argmax(eN[:3, pkN]))
    i_ang = int(np.argmax(np.abs(N[phN, pkN])))
    vph = np.angle(P[4 + phN, pre].mean())          # fundamental phase of matching voltage
    th = 2 * np.pi * i_ang / m + vph                # residual angle w.r.t. voltage cosine peak
    F['incep_cos'] = np.cos(th); F['incep_sin'] = abs(np.sin(th))

    # 6. order spectrum of residual over the active cycles (angle domain => exact orders)
    act = np.where(eSp > tau)[0]
    act = act if len(act) else np.array([pk])
    spec = (np.abs(np.fft.rfft(S[:3][:, act], axis=-1)) ** 2).sum((0, 1))
    tot = spec.sum() + EPS
    for name, sl in [('h0', slice(0, 1)), ('h1', slice(1, 2)), ('h2', slice(2, 3)), ('h3', slice(3, 4)),
                     ('h4_6', slice(4, 7)), ('h7_15', slice(7, 16)), ('h16_63', slice(16, m // 2))]:
        F['ord_' + name] = spec[sl].sum() / tot
    orders = np.arange(len(spec))
    F['ord_centroid'] = np.log10((orders * spec).sum() / tot + 1)
    pn = spec / tot
    F['ord_entropy'] = -(pn * np.log(pn + EPS)).sum()
    # novelty-burst order spectrum at the inception cycle (sharp transient content)
    sn = np.abs(np.fft.rfft(N[:3, pkN], axis=-1)) ** 2
    F['nov_hf_frac'] = sn[:, 7:].sum() / (sn.sum() + EPS)

    # 7. voltage (self-calibrated, sequence-fixed)
    V = P[4:7][d['vorder']]
    V0, V1, V2 = carm.seq(V)
    a0, a1, a2 = abs(V0), abs(V1) + EPS, abs(V2)
    F['V0_pre'] = a0[pre].mean()
    F['V0_max'] = a0[k0:].max()
    F['V0_post'] = a0[-3:].mean()
    F['V0_delta'] = F['V0_max'] - F['V0_pre']
    F['V2_max'] = (a2 / a1)[k0:].max()
    Vm = np.abs(V)
    F['Vmin_phase'] = Vm[:, k0:].min()
    F['Vmax_phase'] = Vm[:, k0:].max()
    F['V1_min'] = a1[k0:].min()
    F['V1_post'] = a1[-3:].mean()
    eV = np.sqrt(np.clip((N[4:7] ** 2).mean(-1) - nz[4:7], 0, None))   # voltage novelty (pu)
    F['Vnov_peak'] = np.log10(eV.max() + EPS)
    F['Vnov_cycles'] = (eV.max(0)[k0:] > max(4 * np.median(eV.max(0)[1:max(k0, 2)]), 0.02)).sum()
    F['energised'] = float(d['energised'])
    F['onset_cycle'] = k0

    if profile:
        # 8. whole-record cycle-novelty distribution (onset-free; robust when the record
        #    starts inside a fault, e.g. sustained arcing earth faults)
        en_all = eNp[1:]
        for q in (50, 75, 90):
            F[f'novq{q}'] = np.log10(np.percentile(en_all, q) + 1e-4)
        F['nov_max_over_med'] = np.log10(max(en_all.max(), 1e-4) / (np.median(en_all) + 1e-4))   # finite when all cycles are below the noise floor
        for th in (0.05, 0.15, 0.5):
            F[f'nov_gt{th}'] = (en_all > th).mean()
            F[f'sup_gt{th}'] = (eSp > th).mean()
        eIn = eN[3, 1:]
        F['In_novq50'] = np.log10(np.median(eIn) + 1e-4)
        F['In_nov_gt0.05'] = (eIn > 0.05).mean()
        eVm = eV.max(0)[1:]
        F['Vnov_q50'] = np.log10(np.median(eVm) + 1e-5)
        for th in (0.02, 0.05, 0.15):
            F[f'Vnov_gt{th}'] = (eVm > th).mean()
        # cyclostationary novelty order spectrum over all cycles (currents, voltages)
        for tag, sl in (('I', slice(0, 3)), ('V', slice(4, 7))):
            sp = (np.abs(np.fft.rfft(N[sl, 1:], axis=-1)) ** 2).sum((0, 1))
            tt = sp.sum() + EPS
            F[f'novspec{tag}_h1'] = sp[1] / tt
            F[f'novspec{tag}_h2_6'] = sp[2:7].sum() / tt
            F[f'novspec{tag}_h7_15'] = sp[7:16].sum() / tt
            F[f'novspec{tag}_h16'] = sp[16:].sum() / tt
        # steady-state (template) distortion, exact harmonic bins thanks to angle resampling
        for tag, sl in (('I', slice(0, 3)), ('V', slice(4, 7))):
            sp = np.abs(np.fft.rfft(T[sl], axis=-1))
            h1 = sp[:, 1] + EPS
            F[f'tmpl{tag}_thd'] = np.log10(np.sqrt((sp[:, 2:20] ** 2).sum(1)).max() / h1.min() + 1e-4)
            F[f'tmpl{tag}_h3'] = (sp[:, 3] / h1).max()
            F[f'tmpl{tag}_h5'] = (sp[:, 5] / h1).max()
        # zero-sequence voltage trajectory (only meaningful once V0 is preserved by the
        # neutral-shift-invariant calibration): step-and-hold (solid earth fault), fluctuating
        # (arcing earth fault), decaying (self-extinguished fault in a resonant-earthed network),
        # return-to-baseline excursions (multi-cycle self-clearing faults)
        r0 = a0 / a1
        base = np.median(r0[pre]) if not d['energised'] else np.median(r0[-3:])
        post0 = r0[k0:] if K - k0 >= 3 else r0[-3:]
        exc = post0 - base
        pkv = int(np.argmax(exc)); peakv = exc[pkv] + EPS
        after = exc[pkv:]
        F['V0traj_peak'] = exc.max()
        F['V0traj_return'] = after.min() / peakv if peakv > 0.02 else 1.0
        F['V0traj_end'] = after[-2:].mean() / peakv if peakv > 0.02 else 1.0
        F['V0traj_var'] = np.median(np.abs(np.diff(post0))) / (post0.mean() + 0.02)
        F['V0traj_elev_frac'] = (exc > 0.1).mean()
        F['V0traj_excursions'] = len(_runs(exc > max(0.1, 0.3 * peakv)))
        if len(after) >= 3 and peakv > 0.05:
            tt = np.arange(len(after))
            F['V0traj_slope'] = np.polyfit(tt, np.log(np.clip(after, 0.01, None) / peakv), 1)[0]
        else:
            F['V0traj_slope'] = 0.0
        # sorted-phase residual energies at the peak cycle (permutation invariant, keeps phase count shape)
        srt = np.sort(eS[:3, pk])[::-1] / (eS[:3, pk].max() + EPS)
        F['peak_mid_ratio'] = srt[1]; F['peak_min_ratio'] = srt[2]
    return F, d


def oss_features(x, use_angle=True, calibrate='ll', d=None):
    """Order-domain spectral statistics: whole-record statistics of each channel after
    angle resampling (exact integer harmonic orders, no leakage for any K) and voltage
    self-calibration, in pu of the pre-event fundamental, sorted across phases."""
    if d is None:
        d = carm.decompose(x, use_angle=use_angle, calibrate=calibrate)
    C, P = d['C'], d['P']
    K, m = C.shape[1], C.shape[2]
    ref = d['ref']
    out = np.zeros((7, 9))
    for ch in range(7):
        base = np.abs(P[ch, ref]).mean() + (1e-9 if ch >= 4 else 1.0)
        if ch < 4:
            base = max(np.abs(P[0:3, ref]).mean(), 5.0)          # common current base
        v = C[ch].ravel() / base
        sp = np.abs(np.fft.rfft(v)) / len(v) * 2
        h = [sp[o * K] if o * K < len(sp) else 0 for o in (1, 2, 3, 5, 7)]
        hf = sp[8 * K:].sum()
        out[ch] = [v.std(), np.abs(v).max(), ((v - v.mean()) ** 4).mean() / (v.var() + 1e-12) ** 2,
                   *np.log1p(h), np.log1p(hf)]
    I = -np.sort(-out[0:3], 0); V = -np.sort(-out[4:7], 0)
    return np.concatenate([I, out[3:4], V]).ravel()


def full_features(x, **kw):
    F, d = carm_features(x, **{k: v for k, v in kw.items() if k in ('use_angle', 'calibrate', 'dual', 'profile', 'denoise')})
    o = oss_features(x, d=d)
    return np.r_[np.array(list(F.values()), float), o], list(F.keys()) + [f'oss_{i}' for i in range(len(o))]


def feature_matrix(X, **kw):
    rows = [carm_features(x, **kw)[0] for x in X]
    names = list(rows[0].keys())
    return np.array([[r[n] for n in names] for r in rows], float), names
