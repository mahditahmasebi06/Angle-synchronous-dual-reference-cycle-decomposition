"""Additional analyses for the revised manuscript (all numbers -> results/extra.json)."""
import os
import paths
import numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
import evaluate as E, features, baselines as B, carm
from invariant import sort_phases
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedGroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scenarios import add_noise

X, y, ys, G = E.X, E.y, E.ys, E.groups
OUT = os.path.join(paths.RESULTS, 'extra.json')
res = {}


def save():
    json.dump(res, open(OUT, 'w'), indent=1, default=float)


def fusion(Xs, fs=4096.0):
    C = np.array([np.array(list(features.carm_features(x, fs=fs)[0].values()), float) for x in Xs])
    B1 = sort_phases(B.b1_stats_fs(Xs, fs), 9)
    return C, B1


def score(F, reps=5):
    out = {}
    for p in ('strat', 'group'):
        r = E.run(E.precomputed(F), p, 'et', reps=reps)
        out[p] = dict(f1=r['f1'][0], f1_sd=r['f1'][1], acc=r['acc'][0])
    return out


C0, B10 = fusion(X)
F0 = np.hstack([C0, B10])
names = features.feature_matrix(X[:1])[1]

# ---------------- 1. observation window after onset
t = time.time(); res['window'] = {}
info = [carm.decompose(x, calibrate='ll') for x in X]
for Nw in [1, 2, 3, 4, 6, 8, 10, 12]:
    Xt = []
    for x, d in zip(X, info):
        n = int(round((d['k0'] + Nw + 1) * 4096.0 / d['f']))
        Xt.append(x[:min(max(n, 5 * 82), len(x))])
    C, B1 = fusion(Xt)
    pre_ms = float(np.median([d['k0'] for d in info]) * 20.0)
    res['window'][Nw] = dict(post_ms=Nw * 20.0, total_ms_median=pre_ms + (Nw + 1) * 20.0,
                             fusion=score(np.hstack([C, B1])), carm=score(C), b1=score(B1))
    print('window', Nw, res['window'][Nw]['fusion'], flush=True); save()
res['window']['full'] = dict(post_ms=None, fusion=score(F0), carm=score(C0), b1=score(B10)); save()
print('window done', time.time() - t, flush=True)

# ---------------- 2. sampling rate
res['fs'] = {}
for q in [1, 2, 4]:
    fs = 4096.0 / q
    Xd = [resample_poly(x, 1, q, axis=0) for x in X] if q > 1 else list(X)
    C, B1 = fusion(Xd, fs)
    res['fs'][int(fs)] = dict(samples_per_cycle=fs / 50, fusion=score(np.hstack([C, B1])), carm=score(C), b1=score(B1))
    print('fs', fs, res['fs'][int(fs)]['fusion'], flush=True); save()

# ---------------- 3. per-class metrics and one-vs-rest AUC
res['perclass'] = {}
reps = {'fusion': F0, 'carm': C0, 'b1_sorted': B10}
for nm, F in reps.items():
    for proto in ('strat', 'group'):
        P = np.zeros((len(y), 4)); yp = []; yt = []; aucs = []
        sp = (list(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(F, y, G) for r in range(10))
              if proto == 'group' else
              [list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=r).split(F, y)) for r in range(10)])
        for r, folds in enumerate(sp):
            prob = np.zeros((len(y), 4))
            for tr, te in folds:
                m = E.model('et', seed=r).fit(F[tr], y[tr]); prob[te] = m.predict_proba(F[te])
            aucs.append([roc_auc_score((y == c).astype(int), prob[:, c]) for c in range(4)])
            yp.append(prob.argmax(1)); yt.append(y)
        yp = np.concatenate(yp); yt = np.concatenate(yt)
        p_, r_, f_, s_ = precision_recall_fscore_support(yt, yp, labels=range(4))
        res['perclass'][f'{nm}|{proto}'] = dict(precision=p_.tolist(), recall=r_.tolist(), f1=f_.tolist(),
                                                support=(s_ // 10).tolist(), auc=np.mean(aucs, 0).tolist(),
                                                auc_sd=np.std(aucs, 0).tolist())
        print('perclass', nm, proto, np.round(f_, 3), np.round(np.mean(aucs, 0), 3), flush=True)
save()

# ---------------- 4. permutation importance by descriptor group (fusion, grouped CV)
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
assert sum(len(v) for v in groups.values()) == len(names), (sum(len(v) for v in groups.values()), len(names))
gidx = {g: [names.index(c) for c in cols] for g, cols in groups.items()}
gidx['Spectral statistics (B1, sorted)'] = list(range(C0.shape[1], F0.shape[1]))
rng = np.random.default_rng(0); drops = {g: [] for g in gidx}
for r in range(5):
    base = np.zeros(len(y), int); perm = {g: [np.zeros(len(y), int) for _ in range(3)] for g in gidx}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(F0, y, G):
        m = E.model('et', seed=r).fit(F0[tr], y[tr]); base[te] = m.predict(F0[te])
        for g, cols in gidx.items():
            for k in range(3):
                Ft = F0[te].copy(); idx = rng.permutation(len(te)); Ft[:, cols] = Ft[idx][:, cols]
                perm[g][k][te] = m.predict(Ft)
    fb = f1_score(y, base, average='macro')
    for g in gidx:
        drops[g].append(np.mean([fb - f1_score(y, p, average='macro') for p in perm[g]]))
res['importance'] = {g: dict(mean=float(np.mean(v)), sd=float(np.std(v)), n_features=len(gidx[g])) for g, v in drops.items()}
print('importance', {g: round(v['mean'], 3) for g, v in res['importance'].items()}, flush=True); save()

# ---------------- 5. learning curve (fraction of training event groups, grouped CV)
res['learning'] = {}
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    out = {}
    for nm, F in [('fusion', F0), ('carm', C0), ('b1_sorted', B10)]:
        f1s = []
        for r in range(5):
            pred = np.zeros(len(y), int); rr = np.random.default_rng(100 + r)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(F, y, G):
                for _ in range(50):
                    gs = np.unique(G[tr]); keep = rr.choice(gs, max(1, int(round(frac * len(gs)))), replace=False)
                    sub = tr[np.isin(G[tr], keep)]
                    if len(np.unique(y[sub])) == 4: break
                pred[te] = E.model('et', seed=r).fit(F[sub], y[sub]).predict(F[te])
            f1s.append(f1_score(y, pred, average='macro'))
        out[nm] = dict(f1=float(np.mean(f1s)), sd=float(np.std(f1s)))
    res['learning'][frac] = out; print('learning', frac, out, flush=True); save()

# ---------------- 6. t-SNE embedding of the fused descriptors
Z = TSNE(2, perplexity=30, random_state=0, init='pca').fit_transform(StandardScaler().fit_transform(F0))
res['tsne'] = dict(x=Z[:, 0].tolist(), y=Z[:, 1].tolist(), label=y.tolist(), subtype=ys.tolist()); save()

# ---------------- 7. detection: rate on events, false alarms on healthy windows, timing
def fired(d):
    return bool(np.any(d['nov_en'] > d['thr']))

det = {'events_detected': int(sum(fired(d) for d in info)), 'events_total': len(X),
       'onset_cycle_counts': np.bincount([d['k0'] for d in info]).tolist()}
for snr in (30, 20):
    Xn = add_noise(X, snr, seed=7)
    det[f'events_detected_snr{snr}'] = int(sum(fired(carm.decompose(x, calibrate='ll')) for x in Xn))
osc = np.load(paths.OSC_NPZ)
wins = []
for i in range(len(osc['names'])):
    v = osc[f'r{i}']
    if not np.isfinite(v[:, 0:6]).all():
        continue
    rms = np.sqrt((v[:, 3:6] ** 2).mean(0))
    for s in range(0, len(v) - 16 * 128, 16 * 128):
        w = v[s:s + 16 * 128]
        if np.sqrt((w[:, 3:6] ** 2).mean(0)).min() < 0.8 * rms.max():
            continue
        cur = np.sqrt((w[:, 0:3] ** 2).mean(0))
        if cur.min() < 1e-4:                       # de-energised feeder: no load current
            continue
        wins.append(np.column_stack([w[:, 0:3], np.zeros(len(w)), w[:, 3:6]]))
fa = [fired(carm.decompose(w, calibrate='ll', fs=6400.0)) for w in wins]
det['healthy_windows'] = len(wins); det['false_alarms'] = int(sum(fa))
for snr in (30, 20):
    wn = add_noise(np.array(wins), snr, seed=9)
    det[f'false_alarms_snr{snr}'] = int(sum(fired(carm.decompose(w, calibrate='ll', fs=6400.0)) for w in wn))
res['detection'] = det; print('detection', det, flush=True); save()

# timing (per record, single core)
tm = {'frequency_ms': [], 'resample_ms': [], 'onset_ms': [], 'calibration_ms': [], 'descriptors_ms': [], 'spectral_ms': []}
for x in X[:60]:
    t0 = time.perf_counter(); f = carm.estimate_frequency(x[:, 4:7]); t1 = time.perf_counter()
    Cc, _ = carm.angle_resample(x, f); t2 = time.perf_counter()
    carm.detect_onset(Cc); t3 = time.perf_counter()
    carm.ll_calibrate(carm.cycle_phasor(Cc)[4:7, :3].mean(1)); t4 = time.perf_counter()
    features.carm_features(x); t5 = time.perf_counter()
    B.b1_stats([x]); t6 = time.perf_counter()
    for k, a, b in [('frequency_ms', t0, t1), ('resample_ms', t1, t2), ('onset_ms', t2, t3), ('calibration_ms', t3, t4),
                    ('descriptors_ms', t4, t5), ('spectral_ms', t5, t6)]:
        tm[k].append((b - a) * 1000)
m = E.model('et', seed=0).fit(F0, y); t0 = time.perf_counter()
for i in range(100):
    m.predict(F0[i % len(y)][None])
tm['classifier_ms'] = [(time.perf_counter() - t0) * 10]
res['timing'] = {k: float(np.median(v)) for k, v in tm.items()}
print('timing', res['timing'], flush=True); save()
print('DONE', flush=True)
