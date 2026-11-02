"""Data behind Figs. 5, 8 and 12 that are not produced by the experiment scripts:
results/descriptor_dump.json (the 91 descriptors of every record and the impurity share of each descriptor group)
results/detect_example.json  (the two records of the detection illustration, Fig. 5)."""
import os, json
import numpy as np
import paths, carm, features, baselines as B, evaluate as E
from invariant import sort_phases

X, y, ys = E.X, E.y, E.ys

# ---- descriptors and group impurity shares (fused model and AS-DRCD alone, mean of 10 seeds)
C0 = np.array([np.array(list(features.carm_features(x)[0].values()), float) for x in X])
names = features.feature_matrix(X[:1])[1]
F0 = np.hstack([C0, sort_phases(B.b1_stats_fs(X, 4096.0), 9)])
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
gidx = {g: [names.index(c) for c in cols] for g, cols in groups.items()}
imp = np.mean([E.model('et', seed=s).fit(F0, y).feature_importances_ for s in range(10)], 0)
impC = np.mean([E.model('et', seed=s).fit(C0, y).feature_importances_ for s in range(10)], 0)
mdi = {g: float(imp[i].sum()) for g, i in gidx.items()}
mdi['Spectral statistics (B1, sorted)'] = float(imp[C0.shape[1]:].sum())
mdiC = {g: float(impC[i].sum()) for g, i in gidx.items()}
json.dump(dict(mdi=mdi, mdiC=mdiC, names=names, C=C0.tolist(), y=y.tolist()),
          open(os.path.join(paths.RESULTS, 'descriptor_dump.json'), 'w'))
print({g: round(v, 3) for g, v in mdi.items()})

# ---- detection illustration: SIF record 13 (index 12) and PF arcing earth fault record 41 (index 40)
ex = {}
for key, i in (('sif', 12), ('pf', 40)):
    d = carm.decompose(X[i], calibrate='ll')
    en = d['nov_en'] / d['Iref']
    V = d['P'][4:7][d['vorder']]; V0, V1, V2 = carm.seq(V)
    r0 = np.abs(V0) / np.maximum(np.maximum(np.abs(V1), np.abs(V2)), 1e-9)
    R = d['nov_en'].max() / np.quantile(d['nov_en'], 0.25); A = d['nov_en'].max() / d['Iref']
    ex[key] = dict(idx=i, sub=int(ys[i]), f=float(d['f']), k0=int(d['k0']), Iref=float(d['Iref']),
                   I=np.round(X[i][::2, :4], 2).tolist(), V=np.round((X[i][:, 4:7] * d['g'])[::2], 3).tolist(),
                   en=np.round(en, 4).tolist(), q25=float(np.quantile(en, 0.25)), r0=np.round(r0, 4).tolist(),
                   thr_on=float(d['thr'] / d['Iref']), R=float(R), A=float(A), fired=bool(R >= 3 and A >= 0.2))
json.dump(ex, open(os.path.join(paths.RESULTS, 'detect_example.json'), 'w'))
print('detect_example.json written')
