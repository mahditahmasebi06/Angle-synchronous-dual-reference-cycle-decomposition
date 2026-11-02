"""Semi-synthetic validation of the voltage self-calibration on independent real waveforms
(OscGrid, unlabelled 50 Hz / 6400 Hz subset, busbar phase voltages).

For healthy 16-cycle segments we inject a known neutral displacement Vn = k * |V1| * e^{j phi}
(as an earth fault in a non-effectively earthed network does) and random channel gains, then
recover V0 with (a) no normalisation (common scale), (b) magnitude equalisation, (c) the proposed
line-voltage (LL) self-calibration. Error = |V0_est - V0_true| / |V1_true|."""
import os
import paths
import numpy as np, json, sys
import carm

FS = 6400.0; M = 128
d = np.load(paths.OSC_NPZ)
names = d['names']
rng = np.random.default_rng(0)

segs = []
for i in range(len(names)):
    v = d[f'r{i}'][:, 3:6]                         # UA UB UC busbar
    if not np.isfinite(v).all():
        continue
    rms = np.sqrt((v ** 2).mean(0))
    if rms.min() < 0.3 * rms.max():                # two-VT (open) connection: phase C missing
        continue
    for s in range(0, len(v) - 16 * M, 16 * M):
        w = v[s:s + 16 * M]
        r = np.sqrt((w ** 2).mean(0))
        if r.min() > 0.8 * rms.max():              # healthy, energised segment only
            segs.append((i, w))
print(len(segs), 'healthy 16-cycle segments from records', sorted(set(i for i, _ in segs)))


def phasors(w):
    C = w.T.reshape(3, -1, M)
    return carm.cycle_phasor(C).mean(1)            # one phasor per phase (16-cycle mean)


def v0_rel(P):
    V0, V1, V2 = carm.seq(P[:, None])
    if abs(V2[0]) > abs(V1[0]):
        V1 = V2
    return V0[0] / abs(V1[0])


res = {}
K = [0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0]
for k in K:
    errs = {'none': [], 'magnitude equalisation': [], 'LL self-calibration (proposed)': []}
    for (i, w) in segs:
        for rep in range(5):
            P = phasors(w)
            _, V1, V2 = carm.seq(P[:, None]); V1m = max(abs(V1[0]), abs(V2[0]))
            Vn = k * V1m * np.exp(1j * rng.uniform(0, 2 * np.pi))
            Pt = P + Vn                              # true phase voltages with neutral shift
            v0_true = v0_rel(Pt)
            g = rng.uniform(0.7, 1.4, 3)              # unknown channel gains
            Pm = Pt / g                              # what the recorder stores
            # (a) common scale only
            errs['none'].append(abs(v0_rel(Pm) - v0_true))
            # (b) magnitude equalisation of the (already shifted) phasors
            ge = carm.voltage_gains(Pm)
            errs['magnitude equalisation'].append(abs(v0_rel(Pm * ge) - v0_true))
            # (c) proposed: null negative sequence of line voltages
            gl, order, _ = carm.ll_calibrate(Pm)
            errs['LL self-calibration (proposed)'].append(abs(v0_rel(Pm * gl) - v0_true))
    res[str(k)] = {m: dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)))
                   for m, e in errs.items()}
    print(f'k={k:.1f}', ' | '.join(f"{m}: med {np.median(e):.3f} p90 {np.percentile(e, 90):.3f}" for m, e in errs.items()))

json.dump(dict(n_segments=len(segs), records=sorted(set(int(i) for i, _ in segs)), results=res),
          open(os.path.join(paths.RESULTS, 'oscgrid_calibration.json'), 'w'), indent=1)
