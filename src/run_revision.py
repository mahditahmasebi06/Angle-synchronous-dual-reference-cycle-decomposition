"""Analyses added in the first revision (all numbers -> results/revision.json).

 1. fold-level scores of the main representations (same splits and seeds as run_final.py) and the
    corrected resampled t-test over J = 50 folds (Nadeau and Bengio 2003)
 2. per-class metrics and one-vs-rest AUC from the same 10 event-grouped shuffles as Table 7 / Fig. 9
 3. end-to-end recall when the Stage II rejections are counted as missed events
 4. onset fallback: onset distribution, pre-onset neutral displacement, accuracy on fallback records
 5. ablations of the fused model and joint permutation of the onset-dependent descriptors
 6. 20 dB noise: in-band noise floor, no noise-floor subtraction, onset taken from the clean record
 7. acquisition characteristics within and between event groups
 8. OscGrid: C1 false alarms without switching windows, C2 false alarms under field-range VT gains
 9. classification under field-range VT gain mismatch of the test records
10. paired differences for the phase-relabelling table, closed-form calibration, batch inference time
"""
import os, json, time
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import paths, carm, features, baselines as B, evaluate as E
from invariant import sort_phases
from scenarios import add_noise

X, y, ys, G = E.X, E.y, E.ys, E.groups
OUT = os.path.join(paths.RESULTS, 'revision.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}       # resume: finished sections are skipped


def save():
    json.dump(res, open(OUT, 'w'), indent=1, default=float)


def run_folds(F, protocol, kind='et', reps=10, Fte=None):
    """Same splits and seeds as evaluate.run; keeps fold-level scores and per-record outputs."""
    Ft = F if Fte is None else Fte
    pred = np.zeros((reps, len(y)), int); proba = np.zeros((reps, len(y), 4))
    fold_f1, ratio = [], []
    for j, (tr, te) in enumerate(E.splits(protocol, reps)):
        clf = E.model(kind, seed=j).fit(F[tr], y[tr])
        p = clf.predict(Ft[te]); r = j // 5
        pred[r, te] = p
        if kind == 'et':
            proba[r, te] = clf.predict_proba(Ft[te])
        fold_f1.append(f1_score(y[te], p, labels=np.unique(y[te]), average='macro'))
        ratio.append(len(te) / len(tr))
    rep_f1 = [f1_score(y, pred[r], average='macro') for r in range(reps)]
    return dict(fold_f1=fold_f1, ratio=ratio, rep_f1=rep_f1, pred=pred, proba=proba)


def finite(F):
    """Variants with a larger noise floor can zero a novelty profile (log10(0)); clip to finite values."""
    return np.nan_to_num(np.asarray(F, float), nan=0.0, posinf=6.0, neginf=-6.0)


def corrected_t(a, b):
    """Corrected resampled t-test over the J fold-level scores (Nadeau & Bengio 2003; Bouckaert & Frank 2004)."""
    d = np.array(a['fold_f1']) - np.array(b['fold_f1']); J = len(d)
    rho = float(np.mean(a['ratio']))                       # n_test / n_train, averaged over the folds
    t = d.mean() / np.sqrt((1.0 / J + rho) * d.var(ddof=1) + 1e-18)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return dict(mean_diff=float(d.mean()), t=float(t), df=J - 1, p=float(p), ratio=rho, J=J)


# ------------------------------------------------------------------ features
t0 = time.time()
C0, names = features.feature_matrix(X)
B1s = sort_phases(B.b1_stats(X), 9); B1r = B.b1_stats(X)
F0 = np.hstack([C0, B1s])
groups = {
    'Duration and recurrence': ['nov_cycles', 'sup_cycles', 'nov_bursts', 'sup_bursts', 'sup_longest', 'nov_span', 'sup_frac_tail', 'burst_ms', 'active_ms', 'active_ratio'],
    'Dual-reference contrast': ['dual_ratio_tail', 'dual_ratio_peak', 'dual_lag'],
    'Persistence': ['sup_tail_over_peak', 'I_post_over_pre', 'I_peakcycle_over_pre'],
    'Magnitude': ['sup_peak_pu', 'nov_peak_pu', 'abs_peak_A', 'Ibase_A', 'n_phases', 'peak_mid_ratio', 'peak_min_ratio'],
    'Earth involvement': ['In_over_phase', 'In_peak_pu'],
    'Residual shape': ['crest', 'kurt', 'halfcycle_asym', 'dc_frac'],
    'Order spectrum': [n for n in names if n.startswith('ord_')] + ['nov_hf_frac'],
    'Inception angle': ['incep_cos', 'incep_sin'],
    'Calibrated voltage': ['V0_pre', 'V0_max', 'V0_post', 'V0_delta', 'V2_max', 'Vmin_phase', 'Vmax_phase', 'V1_min', 'V1_post', 'Vnov_peak', 'Vnov_cycles', 'energised', 'onset_cycle'],
    'Zero-sequence trajectory': [n for n in names if n.startswith('V0traj')],
    'Whole-record profile': [n for n in names if n.startswith(('novq', 'nov_gt', 'sup_gt', 'In_nov', 'Vnov_q', 'Vnov_gt', 'nov_max', 'novspec', 'tmpl'))],
}
prof = [names.index(c) for c in groups['Whole-record profile']]
onset_dep = [i for i in range(len(names)) if i not in prof]          # 61 descriptors that use k0 or T
print('features', round(time.time() - t0), 's', len(onset_dep), 'onset-dependent', flush=True)

# ------------------------------------------------------------------ 1. fold-level scores, corrected t-test
reps = {}
NEED_REPS = not all(k in res for k in ('ttest', 'perclass', 'end_to_end', 'fused_ablation'))
for proto in (('strat', 'group') if NEED_REPS else ()):
    for nm, F in [('fusion', F0), ('asdrcd', C0), ('b1_sorted', B1s), ('b1_raw', B1r)]:
        reps[(proto, nm)] = run_folds(F, proto)
        print(proto, nm, round(np.mean(reps[(proto, nm)]['rep_f1']), 4), flush=True)
if NEED_REPS: res['ttest'] = {}
for proto in (('strat', 'group') if NEED_REPS else ()):
    for a, b in [('fusion', 'b1_sorted'), ('fusion', 'b1_raw'), ('asdrcd', 'b1_sorted'), ('asdrcd', 'b1_raw'), ('fusion', 'asdrcd')]:
        res['ttest'][f'{proto}|{a}|{b}'] = corrected_t(reps[(proto, a)], reps[(proto, b)])
if NEED_REPS:
    res['rep_f1'] = {f'{p}|{n}': v['rep_f1'] for (p, n), v in reps.items()}
    res['fold_f1_mean'] = {f'{p}|{n}': float(np.mean(v['fold_f1'])) for (p, n), v in reps.items()}
    res['ratio_range'] = {p: [float(min(reps[(p, 'fusion')]['ratio'])), float(max(reps[(p, 'fusion')]['ratio']))] for p in ('strat', 'group')}
    save()

# ------------------------------------------------------------------ 2. per-class metrics and AUC, same shuffles
if NEED_REPS: res['perclass'] = {}
for nm in (('fusion', 'asdrcd', 'b1_sorted') if NEED_REPS else ()):
    r = reps[('group', nm)]
    yt = np.tile(y, 10); yp = r['pred'].ravel()
    P, R, F1, _ = precision_recall_fscore_support(yt, yp, labels=range(4), zero_division=0)
    auc = np.mean([[roc_auc_score(y == c, r['proba'][s][:, c]) for c in range(4)] for s in range(10)], 0)
    res['perclass'][nm] = dict(P=P.tolist(), R=R.tolist(), F1=F1.tolist(), AUC=auc.tolist(),
                               cm=confusion_matrix(yt, yp, labels=range(4)).tolist())
save()

# ------------------------------------------------------------------ 3. end-to-end recall with Stage II
det = json.load(open(os.path.join(paths.RESULTS, 'detect_stats2.json')))['events']
v = np.array(det); detected = ((v[:, 0] >= 3) & (v[:, 1] >= 0.2)) | (v[:, 2] >= 0.3)
r = reps.get(('group', 'fusion'))
e2e = {}
if not NEED_REPS:
    e2e = res['end_to_end']
for c, nm in (enumerate(['SIF', 'MIF', 'PF', 'TD']) if NEED_REPS else ()):
    cond = np.mean([(r['pred'][s][y == c] == c).mean() for s in range(10)])
    endto = np.mean([((r['pred'][s] == c) & detected)[y == c].mean() for s in range(10)])
    e2e[nm] = dict(conditional=float(cond), end_to_end=float(endto), missed=int((~detected & (y == c)).sum()))
if NEED_REPS:
    yp_e2e = np.where(detected[None, :], r['pred'], -1)
    e2e['macro_f1_end_to_end'] = float(np.mean([f1_score(y, yp_e2e[s], labels=range(4), average='macro') for s in range(10)]))
    res['end_to_end'] = e2e; save()

# ------------------------------------------------------------------ 4. onset fallback
info = [carm.decompose(x, calibrate='ll') for x in X]
fallback = np.array([not np.any(d['nov_en'] > d['thr']) for d in info])
k0 = np.array([d['k0'] for d in info])
r0pre = []
for d in info:
    V = d['P'][4:7][d['vorder']]; V0, V1, V2 = carm.seq(V)
    r0 = np.abs(V0) / np.maximum(np.abs(V1), 1e-9)
    r0pre.append(float(np.median(r0[d['ref']])))
r0pre = np.array(r0pre)
ef = (ys == 5) & (r0pre > 0.2)                                  # arcing earth faults active from the start
acc_fb = [np.mean([(r['pred'][s][m] == y[m]).mean() for s in range(10)]) for m in (fallback, ~fallback)] if r is not None else \
    [res['onset']['fusion_acc_fallback'], res['onset']['fusion_acc_threshold']]
res['onset'] = dict(n_fallback=int(fallback.sum()), k0_counts=np.bincount(k0, minlength=16).tolist(),
                    k0_counts_fallback=np.bincount(k0[fallback], minlength=16).tolist(),
                    k0_counts_threshold=np.bincount(k0[~fallback], minlength=16).tolist(),
                    fallback_by_class=[int(fallback[y == c].sum()) for c in range(4)],
                    n_ref_cycles_fallback=np.bincount(np.minimum(k0[fallback], 16), minlength=16).tolist(),
                    earth_fault_start=dict(n=int(ef.sum()), pre_r0_median=float(np.median(r0pre[ef])),
                                           pre_r0_q10=float(np.quantile(r0pre[ef], .1)), pre_r0_q90=float(np.quantile(r0pre[ef], .9)),
                                           fallback=int((ef & fallback).sum())),
                    fusion_acc_fallback=float(acc_fb[0]), fusion_acc_threshold=float(acc_fb[1]))
save(); print('onset', res['onset'], flush=True)

# ------------------------------------------------------------------ 5. ablations of the fused model
abl = {'B1 + whole-record profile (30)': np.hstack([C0[:, prof], B1s]),
       'B1 + onset-dependent descriptors (61)': np.hstack([C0[:, onset_dep], B1s])}
if NEED_REPS: res['fused_ablation'] = {}
for nm, F in (abl.items() if NEED_REPS else ()):
    for proto in ('strat', 'group'):
        rr = run_folds(F, proto)
        res['fused_ablation'][f'{proto}|{nm}'] = dict(f1=float(np.mean(rr['rep_f1'])), sd=float(np.std(rr['rep_f1'])),
                                                      t_vs_fusion=corrected_t(reps[(proto, 'fusion')], rr))
        print('ablation', proto, nm, round(np.mean(rr['rep_f1']), 4), flush=True)
rng = np.random.default_rng(0); drops = []
for rp in (range(5) if 'joint_permutation_onset_dependent' not in res else ()):
    from sklearn.model_selection import StratifiedGroupKFold
    base = np.zeros(len(y), int); perm = [np.zeros(len(y), int) for _ in range(3)]
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rp).split(F0, y, G):
        m = E.model('et', seed=rp).fit(F0[tr], y[tr]); base[te] = m.predict(F0[te])
        for k in range(3):
            Ft = F0[te].copy(); idx = rng.permutation(len(te)); Ft[:, onset_dep] = Ft[idx][:, onset_dep]
            perm[k][te] = m.predict(Ft)
    fb = f1_score(y, base, average='macro')
    drops.append(np.mean([fb - f1_score(y, p, average='macro') for p in perm]))
if drops:
    res['joint_permutation_onset_dependent'] = dict(mean=float(np.mean(drops)), sd=float(np.std(drops)), n_features=len(onset_dep))
save(); print('joint permutation', res['joint_permutation_onset_dependent'], flush=True)

# ------------------------------------------------------------------ 6. 20 dB noise diagnostics (train clean, test noisy)
Xn = add_noise(X, 20)                                            # same realisation as run_final.py
B1n = sort_phases(B.b1_stats(Xn), 9)
variants = {'paper setting': dict(), 'orders 30-64 (first submission)': dict(nf_band='fixed'), 'no noise-floor subtraction': dict(denoise=False)}
res['noise20'] = {}
for nm, kw in variants.items():
    if nm == 'paper setting' and 'paper setting|fusion' in res.get('noise20_done', {}):
        continue
    Ctr = C0 if not kw else finite(features.feature_matrix(X, **kw)[0])
    Cte = finite(features.feature_matrix(Xn, **kw)[0])
    for rep, (Ftr, Fte) in {'asdrcd': (Ctr, Cte), 'fusion': (np.hstack([Ctr, B1s]), np.hstack([Cte, B1n]))}.items():
        rr = run_folds(Ftr, 'group', reps=5, Fte=Fte)
        res['noise20'][f'{nm}|{rep}'] = float(np.mean(rr['rep_f1']))
    if kw:
        for proto in ('strat', 'group'):
            rr = run_folds(Ctr, proto); rf = run_folds(np.hstack([Ctr, B1s]), proto)
            res['noise20'][f'{nm}|clean|{proto}|asdrcd'] = float(np.mean(rr['rep_f1']))
            res['noise20'][f'{nm}|clean|{proto}|fusion'] = float(np.mean(rf['rep_f1']))
    print('noise', nm, {k: round(v, 3) for k, v in res['noise20'].items() if k.startswith(nm)}, flush=True); save()
Cfix = finite([np.array(list(features.carm_features(x, k0_fix=k)[0].values()), float) for x, k in zip(Xn, k0)])
k0n = np.array([carm.decompose(x, calibrate='ll')['k0'] for x in Xn])
for rep, (Ftr, Fte) in {'asdrcd': (C0, Cfix), 'fusion': (F0, np.hstack([Cfix, B1n]))}.items():
    res['noise20'][f'clean onset|{rep}'] = float(np.mean(run_folds(Ftr, 'group', reps=5, Fte=Fte)['rep_f1']))
res['noise20']['onset_changed'] = int((k0n != k0).sum())
save(); print('noise', res['noise20'], flush=True)

# ------------------------------------------------------------------ 7. acquisition characteristics within / between event groups
lI = np.log10(np.array([d['Iref'] for d in info]) + 1)
gr = np.array([np.log(d['g'][1] / d['g'][0]) for d in info]); gc = np.array([np.log(d['g'][2] / d['g'][0]) for d in info])
fq = np.array([d['f'] for d in info])
same = G[1:] == G[:-1]
res['groups'] = dict(n_groups=int(len(np.unique(G))), sizes=np.bincount(G).tolist())
for nm, v in [('log10_load_current', lI), ('log_gain_b_a', gr), ('log_gain_c_a', gc), ('frequency_Hz', fq)]:
    dv = np.abs(np.diff(v))
    res['groups'][nm] = dict(within=float(np.median(dv[same])), between=float(np.median(dv[~same])),
                             p_mannwhitney=float(stats.mannwhitneyu(dv[same], dv[~same], alternative='less').pvalue))
save(); print('groups', res['groups'], flush=True)

# ------------------------------------------------------------------ 8. OscGrid detector checks
osc = np.load(paths.OSC_NPZ)
wins = []
for i in range(len(osc['names'])):
    w0 = osc[f'r{i}']
    if not np.isfinite(w0[:, 0:6]).all():
        continue
    rms = np.sqrt((w0[:, 3:6] ** 2).mean(0))
    for s in range(0, len(w0) - 16 * 128, 16 * 128):
        w = w0[s:s + 16 * 128]
        if np.sqrt((w[:, 3:6] ** 2).mean(0)).min() < 0.8 * rms.max(): continue
        if np.sqrt((w[:, 0:3] ** 2).mean(0)).min() < 1e-4: continue
        wins.append(np.column_stack([w[:, 0:3], np.zeros(len(w)), w[:, 3:6]]))
wins = np.array(wins)


def stat(x, fs=4096.0, cal='ll'):
    d = carm.decompose(x, calibrate=cal, fs=fs)
    en = d['nov_en']; q = np.quantile(en, 0.25)
    R = en.max() / max(q, 1e-9); A = en.max() / max(d['Iref'], 1e-6)
    V = d['P'][4:7][d['vorder']]; V0, V1, V2 = carm.seq(V)
    r0 = np.abs(V0) / np.maximum(np.maximum(np.abs(V1), np.abs(V2)), 1e-9)
    on = np.abs(V1) > 0.5 * np.abs(V1).max()
    return [float(R), float(A), float(r0[on].max())]


h = np.array([stat(w, 6400.0) for w in wins])
c1 = (h[:, 0] >= 3) & (h[:, 1] >= 0.2); switching = c1 & (h[:, 1] > 2.0)
rng = np.random.default_rng(11)
Gw = np.exp(rng.uniform(-np.log(3.8) / 2, np.log(3.8) / 2, (len(wins), 3)))      # max/min gain ratio up to 3.8
wg = wins.copy(); wg[:, :, 4:7] *= Gw[:, None, :]
osc_res = dict(n=len(wins), C1=int(c1.sum()), C1_switching=int(switching.sum()),
               C1_rate_without_switching=float((c1 & ~switching).sum() / (len(wins) - switching.sum())),
               gain_ratio_median=float(np.median(Gw.max(1) / Gw.min(1))), gain_ratio_max=float((Gw.max(1) / Gw.min(1)).max()))
for cal, nm in (('ll', 'self-calibration'), (True, 'magnitude equalisation'), (False, 'none')):
    hg = np.array([stat(w, 6400.0, cal) for w in wg])
    osc_res[f'C2|{nm}'] = int((hg[:, 2] >= 0.3).sum()); osc_res[f'rho0_p95|{nm}'] = float(np.quantile(hg[:, 2], .95))
res['oscgrid_detector'] = osc_res; save(); print('oscgrid', osc_res, flush=True)

# ------------------------------------------------------------------ 9. classification with field-range VT gain mismatch in the test records
rng = np.random.default_rng(21)
Gx = np.exp(rng.uniform(-np.log(3.8) / 2, np.log(3.8) / 2, (len(X), 3)))
Xg = X.copy(); Xg[:, :, 4:7] *= Gx[:, None, :]
B1g = sort_phases(B.b1_stats(Xg), 9)
res['gain_test'] = dict(ratio_median=float(np.median(Gx.max(1) / Gx.min(1))))
for cal, nm in (('ll', 'self-calibration'), (True, 'magnitude equalisation'), (False, 'none')):
    Ctr = C0 if cal == 'll' else finite(features.feature_matrix(X, calibrate=cal)[0])
    Cte = finite(features.feature_matrix(Xg, calibrate=cal)[0])
    res['gain_test'][f'{nm}|clean|asdrcd'] = float(np.mean(run_folds(Ctr, 'group', reps=5)['rep_f1']))
    res['gain_test'][f'{nm}|gains|asdrcd'] = float(np.mean(run_folds(Ctr, 'group', reps=5, Fte=Cte)['rep_f1']))
    res['gain_test'][f'{nm}|gains|fusion'] = float(np.mean(run_folds(np.hstack([Ctr, B1s]), 'group', reps=5, Fte=np.hstack([Cte, B1g]))['rep_f1']))
    print('gains', nm, {k: round(v, 3) for k, v in res['gain_test'].items() if k.startswith(nm)}, flush=True); save()

# ------------------------------------------------------------------ 10. paired relabelling differences, calibration, inference time
fin = json.load(open(os.path.join(paths.RESULTS, 'final.json')))
res['relabel_paired_original'] = {k: float(np.mean(fin[f'main|group|{k}']['f1_per_rep'][:n]))
                                  for k, n in [('AS-DRCD/CARM (proposed)', 5), ('B1 raw stats+FFT', 5), ('B5 cycle-diff TEO', 5),
                                               ('B2 DWT db4', 5), ('B3 FSST', 5), ('B4 MiniRocket-lite (ridge)', 3)]}


def closed_form(Vref):
    """|V2(LL)| = 0 as a 2x2 real linear system in g_b/g_a, g_c/g_a (ACB channels swapped first)."""
    V = Vref.copy()
    _, v1, v2 = carm.seq(V[:, None])
    if abs(v2[0]) > abs(v1[0]):
        V = V[[0, 2, 1]]
    a2, a = carm.A ** 2, carm.A
    Amat = np.array([[(a2 * V[1]).real, (a * V[2]).real], [(a2 * V[1]).imag, (a * V[2]).imag]])
    return np.linalg.solve(Amat, -np.array([V[0].real, V[0].imag]))


dif, tcf, tit = [], [], []
for d in info:
    Vref = (d['P'][4:7] / d['g'][:, None])[:, d['ref']].mean(1)         # uncalibrated reference phasor
    s0 = time.perf_counter(); rcf = closed_form(Vref); s1 = time.perf_counter()
    g, order, _ = carm.ll_calibrate(Vref); s2 = time.perf_counter()
    gi = g[order]; rit = np.array([gi[1] / gi[0], gi[2] / gi[0]])
    if np.all(rcf > 0):
        dif.append(np.abs(np.log(rcf / rit)).max())
    tcf.append((s1 - s0) * 1000); tit.append((s2 - s1) * 1000)
m = E.model('et', seed=0).fit(F0, y)
tb = []
for _ in range(5):
    s0 = time.perf_counter(); m.predict_proba(F0[:47]); tb.append((time.perf_counter() - s0) * 1000 / 47)
res['calibration_closed_form'] = dict(n=len(dif), max_abs_log_ratio_diff=float(np.max(dif)), median=float(np.median(dif)),
                                      ms_closed=float(np.median(tcf)), ms_iterative=float(np.median(tit)))
res['inference_ms_per_record_batch47'] = float(np.median(tb))
save(); print('done', round(time.time() - t0), 's', flush=True)
