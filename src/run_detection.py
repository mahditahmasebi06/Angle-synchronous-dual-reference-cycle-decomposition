"""Stage II of AS-DRCD, Eqs. (4)-(5): an event is declared when
  C1: current-novelty contrast R >= 3 and amplitude A >= 0.2 (pu of load current), or
  C2: neutral-voltage displacement max_k |V0|/|V1| >= 0.3 on the self-calibrated voltages.
Writes results/detect_stats2.json (R, A, rho0 per record/window) and results/detection_summary.json (Table 6)."""
import os
import paths
import numpy as np, json, carm, evaluate as E
from scenarios import add_noise
X, y = E.X, E.y
def stat(x, fs=4096.0, cal='ll'):
    d = carm.decompose(x, calibrate=cal, fs=fs)
    en = d['nov_en']; q = np.quantile(en, 0.25)
    R = en.max() / max(q, 1e-9); A = en.max() / max(d['Iref'], 1e-6)
    V = d['P'][4:7][d['vorder']]; V0, V1, V2 = carm.seq(V)
    r0 = np.abs(V0) / np.maximum(np.maximum(np.abs(V1), np.abs(V2)), 1e-9)
    on = np.abs(V1) > 0.5 * np.abs(V1).max()
    return [float(R), float(A), float(r0[on].max())]
osc = np.load(paths.OSC_NPZ)
wins = []
for i in range(len(osc['names'])):
    v = osc[f'r{i}']
    if not np.isfinite(v[:, 0:6]).all(): continue
    rms = np.sqrt((v[:, 3:6] ** 2).mean(0))
    for s in range(0, len(v) - 16 * 128, 16 * 128):
        w = v[s:s + 16 * 128]
        if np.sqrt((w[:, 3:6] ** 2).mean(0)).min() < 0.8 * rms.max(): continue
        if np.sqrt((w[:, 0:3] ** 2).mean(0)).min() < 1e-4: continue
        wins.append(np.column_stack([w[:, 0:3], np.zeros(len(w)), w[:, 3:6]]))
wins = np.array(wins)
rng = np.random.default_rng(5)
G = rng.uniform(0.7, 1.4, (len(wins), 3))                  # random VT gain mismatch on healthy windows
wg = wins.copy(); wg[:, :, 4:7] *= G[:, None, :]
res = {}
res['events'] = [stat(x) for x in X]
res['healthy'] = [stat(w, 6400.0) for w in wins]
for cal in ('ll', True, False):
    res[f'healthy_gain_{cal}'] = [stat(w, 6400.0, cal) for w in wg]
res['events_gain_mageq'] = [stat(x, cal=True) for x in X]
for snr in (30, 20):
    res[f'events_snr{snr}'] = [stat(x) for x in add_noise(X, snr, seed=7)]
    res[f'healthy_snr{snr}'] = [stat(w, 6400.0) for w in add_noise(wins, snr, seed=9)]
json.dump(res, open(os.path.join(paths.RESULTS, 'detect_stats2.json'), 'w'))
TH = dict(R=3, A=0.2, V0=0.3)


def fire(v, eR=TH['R'], eA=TH['A'], e0=TH['V0']):
    v = np.array(v); a = (v[:, 0] >= eR) & (v[:, 1] >= eA); b = v[:, 2] >= e0
    return a, b, a | b


summ = {}
for k, v in res.items():
    a, b, c = fire(v)
    summ[k] = dict(n=len(v), current=int(a.sum()), v0=int(b.sum()), either=int(c.sum()))
    if k.startswith('events'):
        summ[k]['per_class'] = [int(c[y == j].sum()) for j in range(4)]
    print(k, summ[k])
h = np.array(res['healthy']); a, _, _ = fire(h)
summ['healthy_flag_A'] = [round(float(x), 2) for x in h[a, 1]]
summ['thresholds'] = TH
summ['sens'] = {str(p): int(fire(res['events'], e0=p)[2].sum()) for p in (0.2, 0.25, 0.3, 0.35, 0.4)}
json.dump(summ, open(os.path.join(paths.RESULTS, 'detection_summary.json'), 'w'), indent=1)
print('pickup sensitivity', summ['sens'])
