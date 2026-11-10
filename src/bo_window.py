"""Bayesian search over the time-window parameters of AS-DRCD and their effect on classification.

Parameters (paper defaults in brackets)
  onset_ref     novelty values e_N(1..n) in the MAD reference of the onset rule        [3]      2..5
  mad_k         MAD multiplier of the onset threshold                                  [6]      3..10
  tmpl_cycles   pre-onset cycles in the template / calibration reference (4 = all)     [all]    1..4
  energ_cycles  last cycles used as reference for an energisation                      [3]      2..5
  post_cycles   cycles kept after the located onset (13 = full record)                 [full]   1..13
  b1_cycles     cycles of the RMS window that scales the B1 channels                   [1]      1..3

Three analyses (results/window_search.json):
  A. Gaussian-process Bayesian optimisation (expected improvement) of event-grouped macro-F1 of the fused
     model on all records, as a map of the response surface (optimistic by construction);
  B. one-at-a-time sweeps around the paper setting (effect of each parameter alone);
  C. nested evaluation: the search is repeated inside the training groups of each outer event-grouped fold
     and the selected setting is scored on the held-out groups, against the paper setting on the same folds.
Features use no labels, so they are computed once per setting for all records and cached.
"""
import os, sys, json, time, warnings
import paths
import numpy as np
warnings.filterwarnings('ignore')
import evaluate as E, features, carm
from invariant import sort_phases
from sklearn.model_selection import StratifiedGroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

X, y, G = E.X, E.y, E.groups
OUT = os.path.join(paths.RESULTS, 'window_search.json')
SPACE = [('onset_ref', 2, 5, int), ('mad_k', 3.0, 10.0, float), ('tmpl_cycles', 1, 4, int),
         ('energ_cycles', 2, 5, int), ('post_cycles', 1, 13, int), ('b1_cycles', 1, 3, int)]
DEFAULT = dict(onset_ref=3, mad_k=6.0, tmpl_cycles=4, energ_cycles=3, post_cycles=13, b1_cycles=1)
NAMES = [s[0] for s in SPACE]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, 'w'), indent=1, default=float)


def clean(p):
    q = {}
    for n, lo, hi, t in SPACE:
        v = min(max(p[n], lo), hi)
        q[n] = int(round(v)) if t is int else round(float(v) * 4) / 4      # mad_k on a 0.25 grid
    return q


def key(p):
    return tuple(p[n] for n in NAMES)


def encode(p):
    return np.array([(p[n] - lo) / (hi - lo) for n, lo, hi, _ in SPACE])


def sample(rng):
    return clean({n: (rng.integers(lo, hi + 1) if t is int else rng.uniform(lo, hi)) for n, lo, hi, t in SPACE})


# ------------------------------------------------------------------ features for one setting
def b1(Xs, ncyc):
    out = []
    for x in Xs:
        z = x / (np.sqrt((x[:82 * ncyc] ** 2).mean(0)) + 1e-6)
        f = []
        for c in range(7):
            v = z[:, c]
            sp = np.abs(np.fft.rfft(v * np.hanning(len(v))))
            fr = np.fft.rfftfreq(len(v), 1 / 4096.0)
            h = [sp[np.argmin(abs(fr - 50 * k))] for k in (1, 2, 3, 5, 7)]
            f += [v.std(), np.abs(v).max(), ((v - v.mean()) ** 4).mean() / (v.var() + 1e-9) ** 2,
                  *np.log1p(h), np.log1p(sp[fr > 400].sum())]
        out.append(f)
    return sort_phases(np.array(out), 9)


CACHE = {}


def feats(p):
    k = key(p)
    if k in CACHE:
        return CACHE[k]
    carm.WIN.update(onset_ref=p['onset_ref'], mad_k=p['mad_k'],
                    tmpl_cycles=None if p['tmpl_cycles'] == 4 else p['tmpl_cycles'], energ_cycles=p['energ_cycles'])
    Xs = X
    if p['post_cycles'] < 13:                       # cut the record post_cycles after the located onset
        Xs = []
        for x in X:
            d = carm.decompose(x)
            n = int(round((d['k0'] + p['post_cycles'] + 1) * 4096.0 / d['f']))
            Xs.append(x[:min(max(n, 5 * 82), len(x))])
    C = np.array([np.array(list(features.carm_features(x)[0].values()), float) for x in Xs])
    C = np.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6)
    F = (C, np.hstack([C, b1(Xs, p['b1_cycles'])]))
    carm.WIN.update(onset_ref=3, mad_k=6.0, tmpl_cycles=None, energ_cycles=3)
    CACHE[k] = F
    return F


# ------------------------------------------------------------------ scoring
def score(F, idx, groups, protocol='group', reps=3, nfold=5, seed0=0):
    """Macro-F1 of pooled predictions per shuffle/repeat, averaged; idx = records used."""
    Fi, yi = F[idx], y[idx]
    out = []
    for r in range(reps):
        if protocol == 'group':
            sp = StratifiedGroupKFold(nfold, shuffle=True, random_state=seed0 + r).split(Fi, yi, groups[idx])
        else:
            sp = RepeatedStratifiedKFold(n_splits=nfold, n_repeats=1, random_state=seed0 + r).split(Fi, yi)
        pred = np.zeros(len(yi), int)
        for j, (tr, te) in enumerate(sp):
            pred[te] = E.model('et', seed=r * nfold + j).fit(Fi[tr], yi[tr]).predict(Fi[te])
        out.append(f1_score(yi, pred, average='macro'))
    return float(np.mean(out))


# ------------------------------------------------------------------ GP-EI Bayesian optimisation
def bo(objective, n_init, n_iter, rng, log=None):
    hist = []
    pts = [dict(DEFAULT)] + [sample(rng) for _ in range(n_init)]
    for p in pts:
        hist.append((p, objective(p)))
        if log: log(len(hist), p, hist[-1][1])
    for it in range(n_iter):
        Z = np.array([encode(p) for p, _ in hist]); v = np.array([s for _, s in hist])
        mu0, sd0 = v.mean(), v.std() + 1e-9
        kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(len(SPACE), 0.5),
                                                         length_scale_bounds=(0.05, 20), nu=2.5) \
            + WhiteKernel(0.1, (1e-4, 1.0))
        gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=3, random_state=it)
        gp.fit(Z, (v - mu0) / sd0)
        seen = {key(p) for p, _ in hist}
        cands = [c for c in (sample(rng) for _ in range(4000)) if key(c) not in seen]
        Zc = np.array([encode(c) for c in cands])
        m, s = gp.predict(Zc, return_std=True)
        best = (v.max() - mu0) / sd0
        zz = (m - best - 0.01) / (s + 1e-9)
        ei = (m - best - 0.01) * norm.cdf(zz) + s * norm.pdf(zz)
        p = cands[int(np.argmax(ei))]
        hist.append((p, objective(p)))
        if log: log(len(hist), p, hist[-1][1])
    return hist, gp


ALL = np.arange(len(y))
t0 = time.time()
stage = sys.argv[1] if len(sys.argv) > 1 else 'ABC'

# ---------------- A. response surface on all records
if 'A' in stage and 'A' not in res:
    rng = np.random.default_rng(1)
    trials = []

    def objA(p):
        C, Fu = feats(p)
        r = dict(params=p, fusion=score(Fu, ALL, G), asdrcd=score(C, ALL, G))
        trials.append(r); res['A_trials'] = trials; save()
        return r['fusion']

    hist, gp = bo(objA, 12, 48, rng,
                  log=lambda i, p, s: print(f'A {i:2d} {s:.3f} {p} {time.time() - t0:.0f}s', flush=True))
    Z = np.array([encode(p) for p, _ in hist])
    pdp = {}
    for d, (n, lo, hi, t) in enumerate(SPACE):          # GP partial dependence over the visited settings
        grid = np.arange(lo, hi + 1) if t is int else np.linspace(lo, hi, 15)
        vals = []
        for g in grid:
            Zg = Z.copy(); Zg[:, d] = (g - lo) / (hi - lo)
            vals.append(float(gp.predict(Zg).mean()))
        v = np.array([s for _, s in hist])
        vals = np.array(vals) * (v.std() + 1e-9) + v.mean()
        pdp[n] = dict(grid=[float(g) for g in grid], f1=vals.tolist(), range=float(vals.max() - vals.min()))
    res['A'] = dict(best=max(trials, key=lambda r: r['fusion']), default=trials[0], pdp=pdp,
                    length_scales=dict(zip(NAMES, gp.kernel_.k1.k2.length_scale.tolist())), n=len(trials))
    save()

# ---------------- B. one-at-a-time sweeps around the paper setting
if 'B' in stage and 'B' not in res:
    grids = dict(onset_ref=[2, 3, 4, 5], mad_k=[3.0, 4.0, 5.0, 6.0, 8.0, 10.0], tmpl_cycles=[1, 2, 3, 4],
                 energ_cycles=[2, 3, 4, 5], post_cycles=[1, 2, 3, 4, 6, 8, 10, 12, 13], b1_cycles=[1, 2, 3])
    B = res.get('B_partial', {})
    memo = {}
    for n, grid in grids.items():
        B.setdefault(n, {})
        for g in grid:
            if str(g) in B[n]:
                continue
            p = clean(dict(DEFAULT, **{n: g}))
            if key(p) not in memo:
                C, Fu = feats(p)
                memo[key(p)] = dict(fusion_group=score(Fu, ALL, G, reps=5), asdrcd_group=score(C, ALL, G, reps=5),
                                    fusion_strat=score(Fu, ALL, G, 'strat', reps=5))
            B[n][str(g)] = memo[key(p)]
            print('B', n, g, B[n][str(g)], f'{time.time() - t0:.0f}s', flush=True)
            res['B_partial'] = B; save()
    res['B'] = B; save()

# ---------------- C. nested evaluation of the search
if 'C' in stage and 'C' not in res:
    outer = res.get('C_outer', [])
    splits = [(s, k, tr, te) for s in (100, 101)
              for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, G))]
    for s, k, tr, te in splits[len(outer):]:
        rng = np.random.default_rng(s * 10 + k)

        def objC(p):
            return score(feats(p)[1], tr, G, reps=2, nfold=4, seed0=50)

        hist, _ = bo(objC, 8, 16, rng)
        best = max(hist, key=lambda h: h[1])[0]
        row = dict(shuffle=s, fold=k, best=best, inner_best=max(h[1] for h in hist),
                   inner_default=hist[0][1], test=te.tolist())
        for nm, p in (('tuned', best), ('default', DEFAULT)):
            Fu = feats(p)[1]
            row['pred_' + nm] = E.model('et', seed=k).fit(Fu[tr], y[tr]).predict(Fu[te]).tolist()
        outer.append(row); res['C_outer'] = outer; save()
        print('C', s, k, best, f"inner {row['inner_default']:.3f}->{row['inner_best']:.3f}",
              f'{time.time() - t0:.0f}s', flush=True)
    summ = {}
    for s in (100, 101):
        rows = [r for r in outer if r['shuffle'] == s]
        idx = np.concatenate([r['test'] for r in rows])
        for nm in ('tuned', 'default'):
            pr = np.concatenate([r['pred_' + nm] for r in rows])
            summ.setdefault(nm, []).append(f1_score(y[idx], pr, average='macro'))
    res['C'] = dict(tuned=summ['tuned'], default=summ['default'],
                    tuned_mean=float(np.mean(summ['tuned'])), default_mean=float(np.mean(summ['default'])),
                    selected=[r['best'] for r in outer])
    save()
print('done', f'{time.time() - t0:.0f}s', flush=True)
