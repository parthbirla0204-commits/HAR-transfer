"""Subject-wise k-fold splits with a subject-disjoint validation set; labelled-fraction sampling; normalisation."""
import numpy as np


def subject_folds(subjects, n_folds=5, seed=0):
    subs = np.array(sorted(set(subjects)))
    np.random.RandomState(seed).shuffle(subs)
    return np.array_split(subs, min(n_folds, len(subs)))


def make_split(subj, folds, fold, seed, val_frac=0.2):
    test = set(folds[fold])
    rest = np.array([s for i, f in enumerate(folds) if i != fold for s in f])
    np.random.RandomState(seed + 1000 * fold).shuffle(rest)
    n_val = max(1, int(round(val_frac * len(rest))))
    val, train = set(rest[:n_val]), set(rest[n_val:])
    idx = lambda S: np.where(np.isin(subj, list(S)))[0]
    tr, va, te = idx(train), idx(val), idx(test)
    assert not (set(subj[tr]) & set(subj[va])) and not (set(subj[tr]) & set(subj[te])) and not (set(subj[va]) & set(subj[te]))
    return tr, va, te


def sample_fraction(y, idx, frac, seed, min_per_class=2):
    """Stratified random sample of `frac` of the windows in idx (>= min_per_class per class when available)."""
    if frac >= 1.0:
        return idx
    rng = np.random.RandomState(seed)
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = min(len(ci), max(min_per_class, int(round(frac * len(ci)))))
        out.append(rng.choice(ci, k, replace=False))
    return np.sort(np.concatenate(out))


class Normalizer:
    """Per-channel mean/std computed on a TRAINING split only."""
    def __init__(self, X):
        self.mu = X.mean((0, 2), keepdims=True).astype(np.float32)
        self.sd = (X.std((0, 2), keepdims=True) + 1e-6).astype(np.float32)

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)
