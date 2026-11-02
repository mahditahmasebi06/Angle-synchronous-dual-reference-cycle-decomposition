"""Compute and cache baseline representations used by run_final.py."""
import os
import paths
import numpy as np
import evaluate as E, baselines as B

X = E.X
feats = {'B1 raw stats+FFT': B.b1_stats(X), 'B2 DWT db4': B.b2_dwt(X),
         'B3 FSST': B.b3_sst(X), 'B5 cycle-diff TEO': B.b5_teo(X)}
np.savez(os.path.join(paths.CACHE, 'feat_baselines.npz'), **{k.replace(' ', '_'): v for k, v in feats.items()})
np.save(os.path.join(paths.CACHE, 'feat_minirocket.npy'), B.MiniRocketLite(1000, seed=0).fit(X).transform(X))
print({k: v.shape for k, v in feats.items()})
