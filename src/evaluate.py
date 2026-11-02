import os
import paths
import numpy as np, itertools, json, sys, time
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

D = np.load(paths.DATA_NPZ); X, y, ys = D['X'], D['y'], D['ys']
runs = [len(list(g)) for _, g in itertools.groupby(ys)]
groups = np.repeat(np.arange(len(runs)), runs)


def model(kind='et', seed=0):
    if kind == 'et':
        return ExtraTreesClassifier(300, class_weight='balanced', min_samples_leaf=1,
                                    max_features='sqrt', n_jobs=2, random_state=seed)
    if kind == 'rf':
        return RandomForestClassifier(300, class_weight='balanced', n_jobs=2, random_state=seed)
    if kind == 'hgb':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              class_weight='balanced', random_state=seed)
    if kind == 'ridge':
        return make_pipeline(StandardScaler(), RidgeClassifierCV(np.logspace(-3, 3, 10),
                                                                 class_weight='balanced'))


def splits(protocol, reps=10):
    if protocol == 'strat':
        yield from RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=0).split(X, y)
    else:
        for r in range(reps):
            yield from StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, groups)


def run(featfn, protocol='strat', kind='et', reps=10, transform=None, X_override=None):
    """featfn(Xtrain_raw, Xtest_raw) -> (Ftr, Fte); returns per-repeat metrics and pooled CM."""
    accs, f1s, bas, cm = [], [], [], np.zeros((4, 4), int)
    yt_all, yp_all = [], []
    sp = list(splits(protocol, reps))
    Xsrc = X if X_override is None else X_override
    for j, (tr, te) in enumerate(sp):
        Ftr, Fte = featfn(X[tr], Xsrc[te], tr, te)
        clf = model(kind, seed=j).fit(Ftr, y[tr])
        p = clf.predict(Fte)
        yt_all.append(y[te]); yp_all.append(p)
        if (j + 1) % 5 == 0:                       # one full 5-fold repeat
            yt, yp = np.concatenate(yt_all[-5:]), np.concatenate(yp_all[-5:])
            accs.append(accuracy_score(yt, yp)); f1s.append(f1_score(yt, yp, average='macro'))
            bas.append(balanced_accuracy_score(yt, yp))
            cm += confusion_matrix(yt, yp, labels=range(4))
    return dict(acc=(np.mean(accs), np.std(accs)), f1=(np.mean(f1s), np.std(f1s)),
                bacc=(np.mean(bas), np.std(bas)), cm=cm.tolist(),
                f1_per_rep=f1s, acc_per_rep=accs)


def precomputed(F):
    return lambda Xtr, Xte, tr, te: (F[tr], F[te])
