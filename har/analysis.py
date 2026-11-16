"""Single-case study used by the app and make_report.py: embeddings before/after, confusion matrices, CKA drift."""
import numpy as np
from .config import TrainCfg
from .experiments import get_source, source_model
from .splits import subject_folds, make_split, sample_fraction, Normalizer
from .transfer import run_method, layer_features, representation_drift, get_device, PROPOSED, predict


def case_study(datasets, src_key, tgt_name, frac, seed, fold, methods, cfg=None, device=None, arch="resnet1d", n_folds=5):
    cfg = cfg or TrainCfg(); device = device or get_device()
    src, tgt = get_source(datasets, src_key), datasets[tgt_name]
    folds = subject_folds(tgt.subj, n_folds, seed); fold = min(fold, len(folds) - 1)
    tr, va, te = make_split(tgt.subj, folds, fold, seed, cfg.val_frac_subjects)
    tn = Normalizer(tgt.X[tr]); sm, Xsrc = source_model(src, src_key, arch, seed, cfg, device)
    sub = sample_fraction(tgt.y, tr, frac, seed * 100 + fold); vs = sample_fraction(tgt.y, va, frac, seed * 100 + fold, 5)
    args = (tn(tgt.X[sub]), tgt.y[sub], tn(tgt.X[vs]), tgt.y[vs], tn(tgt.X[te]), tgt.y[te])
    res = {m: run_method(m, sm, arch, *args, Xsrc, cfg, device, seed) for m in methods}
    Xte, yte = args[4], args[5]
    out = {"results": res, "Xte": Xte, "yte": yte, "source_model": sm, "Xsrc": Xsrc, "norm": tn}
    if PROPOSED in res:
        out["cka_drift"] = representation_drift(sm, res[PROPOSED]["model"], Xte, device)
        out["layer_names"] = sm.names
    return out


def penultimate(model, X, device):
    return layer_features(model, X, device)[-1].numpy()
