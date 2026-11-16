import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from har.data import make_synthetic, resample_to_fs, window_signal
from har.splits import subject_folds, make_split, Normalizer
from har.transfer import linear_cka, rbf_mmd2
import torch


def test_windows_and_resample():
    t = np.arange(0, 20, 1 / 20.0); sig = resample_to_fs(t, np.random.randn(len(t), 3))
    assert abs(len(sig) - 20 * 50) <= 2 and window_signal(sig).shape[1:] == (3, 128)


def test_subject_disjoint():
    d = make_synthetic(n_subjects=15, windows_per_class=2)
    folds = subject_folds(d.subj, 5, 0)
    for f in range(5):
        tr, va, te = make_split(d.subj, folds, f, 0)
        S = [set(d.subj[i]) for i in (tr, va, te)]
        assert not (S[0] & S[1] or S[0] & S[2] or S[1] & S[2])


def test_normalizer_uses_train_only():
    X = np.random.randn(50, 3, 128).astype(np.float32) * 3 + 1
    n = Normalizer(X[:25]); assert np.allclose(n.mu.ravel(), X[:25].mean((0, 2)), atol=1e-5)


def test_measures():
    a = torch.randn(100, 8); assert rbf_mmd2(a, a + 0) < 1e-6 and rbf_mmd2(a, a + 3) > 0.1
    x = np.random.randn(100, 8); assert abs(linear_cka(x, x) - 1) < 1e-6
