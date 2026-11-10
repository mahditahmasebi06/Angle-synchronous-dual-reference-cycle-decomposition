"""Data behind the time-window figure: the windows of SIF record 13 (index 12) at the paper setting and a
one-cycle zoom of the input samples against the angle grid. Writes results/window_example.json."""
import os, json
import numpy as np
import paths, carm

d0 = np.load(paths.DATA_NPZ, allow_pickle=True)
X = d0['X'] if 'X' in d0.files else d0[d0.files[0]]
i = 12
x = np.asarray(X[i], float)
d = carm.decompose(x, calibrate='ll')
f, k0 = float(d['f']), int(d['k0'])
C, tq = carm.angle_resample(x, f)
K, M = C.shape[1], C.shape[2]
ph = int(np.argmax(np.abs(C[0:3, k0]).max(-1)))          # most disturbed phase in the onset cycle
k = k0
t0, t1 = k / f, (k + 1) / f
n = np.arange(x.shape[0]); tn = n / carm.FS
sel = (tn >= t0) & (tn < t1)
out = dict(idx=i, f=f, k0=k0, K=K, M=M, fs=carm.FS, L=int(round(carm.FS / f)), phase='abc'[ph], zoom_cycle=k,
           raw_t=np.round((tn[sel] - t0) * 1000, 4).tolist(), raw_v=np.round(x[sel, ph], 2).tolist(),
           grid_t=np.round(np.arange(M) / (M * f) * 1000, 4).tolist(), grid_v=np.round(C[ph, k], 2).tolist(),
           I=np.round(x[::2, :4], 2).tolist(), nov_en=np.round(d['nov_en'] / d['Iref'], 4).tolist())
json.dump(out, open(os.path.join(paths.RESULTS, 'window_example.json'), 'w'))
print('window_example.json written', f, k0, K, M, out['L'], 'phase', out['phase'], 'raw', int(sel.sum()))
