"""Final consolidated experiments (all numbers in the draft come from results/final.json)."""
import os
import paths
import numpy as np, json, time, sys
import evaluate as E, features, baselines as B
from invariant import sort_phases
from scenarios import add_noise, shift_freq

X, y = E.X, E.y
OUT = os.path.join(paths.RESULTS, 'final.json')
res = {}


def log(key, r):
    res[key] = r
    print(f"{key:70s} acc {r['acc'][0]:.3f}±{r['acc'][1]:.3f}  mF1 {r['f1'][0]:.3f}±{r['f1'][1]:.3f}  bAcc {r['bacc'][0]:.3f}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1, default=float)


carm = lambda Z, **kw: features.feature_matrix(Z, **kw)[0]
fb = np.load(os.path.join(paths.CACHE, 'feat_baselines.npz'))
nper = {'B1_raw_stats+FFT': 9, 'B2_DWT_db4': 11, 'B3_FSST': 11, 'B5_cycle-diff_TEO': 5}
F = {'AS-DRCD/CARM (proposed)': carm(X)}
for k, n in nper.items():
    F[k.replace('_', ' ')] = fb[k]
    F[k.replace('_', ' ') + ' (phase-sorted)'] = sort_phases(fb[k], n)
F['AS-DRCD + B1 phase-sorted (fusion)'] = np.hstack([F['AS-DRCD/CARM (proposed)'], F['B1 raw stats+FFT (phase-sorted)']])
MR = np.load(os.path.join(paths.CACHE, 'feat_minirocket.npy'))

# ---------------- 1. main comparison
for proto in ['strat', 'group']:
    for k, FF in F.items():
        log(f'main|{proto}|{k}', E.run(E.precomputed(FF), proto, 'et'))
    log(f'main|{proto}|B4 MiniRocket-lite (ridge)', E.run(E.precomputed(MR), proto, 'ridge'))

# ---------------- 2. ablations of the proposed pipeline
abl = {'no angle resampling (integer 82-sample cycles)': dict(use_angle=False),
       'magnitude-equalisation voltage normalisation': dict(calibrate=True),
       'no voltage normalisation': dict(calibrate=False),
       'no dual-reference descriptors': dict(dual=False),
       'no whole-record profile, V0 trajectory, template harmonics': dict(profile=False),
       'no noise-floor correction': dict(denoise=False),
       'noise floor from orders 30-64 (first submission)': dict(nf_band='fixed')}
for name, kw in abl.items():
    FF = carm(X, **kw)
    for proto in ['strat', 'group']:
        log(f'ablation|{proto}|{name}', E.run(E.precomputed(FF), proto, 'et'))

# ---------------- 3. phase relabelling of test records (train on original labels)
rng = np.random.default_rng(1)
perms = [(1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
Xp = X.copy()
for i, j in enumerate(rng.integers(0, 5, len(X))):
    p = list(perms[j])
    Xp[i][:, [0, 1, 2]] = X[i][:, p]
    Xp[i][:, [4, 5, 6]] = X[i][:, [4 + q for q in p]]
Fp = {'AS-DRCD/CARM (proposed)': carm(Xp), 'B1 raw stats+FFT': B.b1_stats(Xp)}
for proto in ['strat', 'group']:
    for k, FFp in Fp.items():
        FF = F[k]
        log(f'relabel|{proto}|{k}', E.run(lambda a, b, tr, te: (FF[tr], FFp[te]), proto, 'et', reps=5))

# ---------------- 4. robustness: train clean, test corrupted (grouped protocol)
cor = {'SNR 30 dB': add_noise(X, 30), 'SNR 20 dB': add_noise(X, 20),
       'f = 49.5 Hz': shift_freq(X, 49.5), 'f = 50.5 Hz': shift_freq(X, 50.5)}
fns = {'AS-DRCD/CARM (proposed)': (F['AS-DRCD/CARM (proposed)'], carm),
       'AS-DRCD without angle resampling': (carm(X, use_angle=False), lambda Z: carm(Z, use_angle=False)),
       'B1 raw stats+FFT (phase-sorted)': (F['B1 raw stats+FFT (phase-sorted)'], lambda Z: sort_phases(B.b1_stats(Z), 9)),
       'B5 cycle-diff TEO (phase-sorted)': (F['B5 cycle-diff TEO (phase-sorted)'], lambda Z: sort_phases(B.b5_teo(Z), 5))}
for cn, Xc in cor.items():
    tst = {}
    for k, (Fc_, fn) in fns.items():
        tst[k] = fn(Xc)
        Fa, Fb = Fc_, tst[k]
        log(f'robust|{cn}|{k}', E.run(lambda a, b, tr, te: (Fa[tr], Fb[te]), 'group', 'et', reps=5))
    Fa = F['AS-DRCD + B1 phase-sorted (fusion)']
    Fb = np.hstack([tst['AS-DRCD/CARM (proposed)'], tst['B1 raw stats+FFT (phase-sorted)']])
    log(f'robust|{cn}|AS-DRCD + B1 phase-sorted (fusion)', E.run(lambda a, b, tr, te: (Fa[tr], Fb[te]), 'group', 'et', reps=5))
print('DONE', flush=True)
