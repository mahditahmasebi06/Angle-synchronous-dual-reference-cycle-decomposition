import numpy as np
def sort_phases(F, n_per_ch):
    """F columns grouped by channel: Ia Ib Ic In Va Vb Vc, n_per_ch features each.
    Sort each feature across the three current phases and across the three voltage phases
    (descending) -> phase-permutation-invariant version of a per-channel representation."""
    F = F.reshape(len(F), 7, n_per_ch)
    I = -np.sort(-F[:, 0:3], axis=1); V = -np.sort(-F[:, 4:7], axis=1)
    return np.concatenate([I, F[:, 3:4], V], 1).reshape(len(F), -1)
