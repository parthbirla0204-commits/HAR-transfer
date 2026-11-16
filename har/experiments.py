"""Full experimental grid: pair x seed x fold x fraction x method, resumable, written row-by-row to CSV."""
import json, os
import numpy as np
import pandas as pd
from .config import TrainCfg, FRACTIONS
from .data import concat
from .splits import subject_folds, make_split, sample_fraction, Normalizer
from .transfer import pretrain, run_method, get_device, ALL_METHODS

_SRC_CACHE = {}


def source_model(src, src_key, arch, seed, cfg, device):
    """Pre-train once per (source, arch, seed); subject-disjoint validation set for early stopping."""
    k = (src_key, arch, seed)
    if k not in _SRC_CACHE:
        subs = np.array(sorted(set(src.subj))); np.random.RandomState(seed).shuffle(subs)
        nv = max(1, int(round(cfg.val_frac_subjects * len(subs))))
        va = np.where(np.isin(src.subj, subs[:nv]))[0]; tr = np.where(~np.isin(src.subj, subs[:nv]))[0]
        norm = Normalizer(src.X[tr])
        m, _ = pretrain(arch, norm(src.X[tr]), src.y[tr], norm(src.X[va]), src.y[va], cfg, device, seed)
        _SRC_CACHE[k] = (m, norm(src.X[tr]))
    return _SRC_CACHE[k]


def get_source(datasets, src_key):
    return concat([datasets[n] for n in src_key.split("+")], src_key)


def run_grid(datasets, pairs, out_csv, fractions=FRACTIONS, methods=None, n_folds=5, seeds=range(5),
             cfg=None, device=None, arch="resnet1d", progress=None, verbose=True):
    cfg = cfg or TrainCfg(); device = device or get_device(); methods = methods or ALL_METHODS
    done = set()
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv)
        done = set(zip(d.pair, d.frac, d.method, d.fold, d.seed))
    total = len(pairs) * len(list(seeds)) * n_folds * len(fractions) * len(methods); n = 0
    for src_key, tgt_name in pairs:
        pair = f"{src_key}->{tgt_name}"; src = get_source(datasets, src_key); tgt = datasets[tgt_name]
        for seed in seeds:
            folds = subject_folds(tgt.subj, n_folds, seed)
            for fold in range(len(folds)):
                tr, va, te = make_split(tgt.subj, folds, fold, seed, cfg.val_frac_subjects)
                tnorm = Normalizer(tgt.X[tr])                       # statistics from the TRAIN split only
                Xte, yte = tnorm(tgt.X[te]), tgt.y[te]
                sm, Xsrc = source_model(src, src_key, arch, seed, cfg, device)
                for frac in fractions:
                    sub = sample_fraction(tgt.y, tr, frac, seed * 100 + fold)
                    vsub = sample_fraction(tgt.y, va, frac, seed * 100 + fold, min_per_class=5)
                    Xtr, ytr, Xva, yva = tnorm(tgt.X[sub]), tgt.y[sub], tnorm(tgt.X[vsub]), tgt.y[vsub]
                    for method in methods:
                        n += 1
                        if (pair, frac, method, fold, seed) in done:
                            continue
                        r = run_method(method, sm, arch, Xtr, ytr, Xva, yva, Xte, yte, Xsrc, cfg, device, seed)
                        cm = np.zeros((5, 5), int)
                        np.add.at(cm, (yte, r["pred"]), 1)
                        row = dict(pair=pair, source=src_key, target=tgt_name, frac=frac, method=method, fold=fold,
                                   seed=seed, acc=r["acc"], macro_f1=r["macro_f1"], trainable_params=r["trainable_params"],
                                   train_time=r["train_time"], epochs=r["epochs"], n_train=len(sub),
                                   disc=r.get("disc", ""), selected=r.get("selected", ""), cm=json.dumps(cm.ravel().tolist()))
                        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                        if progress:
                            progress(n, total, f"{pair} seed={seed} fold={fold} frac={frac} {method}: F1={r['macro_f1']:.3f}")
                        elif verbose:
                            print(f"[{n}/{total}] {pair} s{seed} f{fold} {frac} {method}: acc={r['acc']:.3f} f1={r['macro_f1']:.3f}", flush=True)
    return pd.read_csv(out_csv)
