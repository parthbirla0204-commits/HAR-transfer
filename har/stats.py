"""Summaries (mean +- std over seeds), paired significance vs the strongest baseline, negative-transfer table."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from .transfer import BASELINES, PROPOSED


def per_seed(df):
    """Average the folds inside each seed; mean +- std are then taken across seeds."""
    return df.groupby(["pair", "frac", "method", "seed"], as_index=False)[["acc", "macro_f1", "trainable_params", "train_time"]].mean()


def summary(df):
    ps = per_seed(df)
    g = ps.groupby(["pair", "frac", "method"])
    out = g[["acc", "macro_f1"]].agg(["mean", "std"]).reset_index()
    out.columns = ["pair", "frac", "method", "acc_mean", "acc_std", "f1_mean", "f1_std"]
    out[["params", "time_s"]] = g[["trainable_params", "train_time"]].mean().reset_index(drop=True)
    out["acc"] = out.acc_mean.map("{:.3f}".format) + " ± " + out.acc_std.fillna(0).map("{:.3f}".format)
    out["macro_f1"] = out.f1_mean.map("{:.3f}".format) + " ± " + out.f1_std.fillna(0).map("{:.3f}".format)
    return out


def significance(df, proposed=PROPOSED, metric="macro_f1", baselines=BASELINES):
    """Per (pair, fraction): paired t-test and Wilcoxon of `proposed` vs the strongest baseline (highest mean F1).
    Pairing is over (seed, fold) runs, which share the exact same split and labelled subset."""
    rows = []
    for (pair, frac), g in df.groupby(["pair", "frac"]):
        b = g[g.method.isin(baselines)].groupby("method")[metric].mean()
        if b.empty or proposed not in set(g.method):
            continue
        best = b.idxmax()
        a = g[g.method == proposed].set_index(["seed", "fold"])[metric]
        c = g[g.method == best].set_index(["seed", "fold"])[metric]
        idx = a.index.intersection(c.index); a, c = a.loc[idx], c.loc[idx]
        if len(idx) < 3:
            continue
        t, pt = stats.ttest_rel(a, c)
        try:
            pw = stats.wilcoxon(a, c).pvalue
        except ValueError:
            pw = np.nan
        rows.append(dict(pair=pair, frac=frac, strongest_baseline=best, proposed=a.mean(), baseline=c.mean(),
                         delta=a.mean() - c.mean(), t=t, p_ttest=pt, p_wilcoxon=pw, n=len(idx)))
    out = pd.DataFrame(rows)
    if len(out):
        out["significant_0.05"] = (out.p_ttest < .05) & (out.p_wilcoxon < .05)
    return out


def negative_transfer(df, methods=("full_ft", "frozen_head", PROPOSED), metric="macro_f1"):
    """Cases where a transfer method is worse than training from scratch (mean over seeds+folds), with paired p-values."""
    rows = []
    for (pair, frac), g in df.groupby(["pair", "frac"]):
        if "scratch" not in set(g.method):
            continue
        s = g[g.method == "scratch"].set_index(["seed", "fold"])[metric]
        for m in methods:
            x = g[g.method == m].set_index(["seed", "fold"])[metric]
            idx = x.index.intersection(s.index)
            if len(idx) < 3:
                continue
            d = x.loc[idx] - s.loc[idx]
            p = stats.ttest_rel(x.loc[idx], s.loc[idx]).pvalue
            rows.append(dict(pair=pair, frac=frac, method=m, delta_vs_scratch=d.mean(), p=p, negative=d.mean() < 0))
    return pd.DataFrame(rows)


def mean_discrepancy(df):
    """Per pair: mean layer-wise MMD^2 recorded by the proposed method (a diagnostic for negative transfer)."""
    d = df[df.method == PROPOSED].dropna(subset=["disc"])
    rows = []
    for pair, g in d.groupby("pair"):
        arr = np.array([json.loads(x) for x in g.disc if isinstance(x, str) and x])
        if len(arr):
            rows.append({"pair": pair, **{f"layer{i}": v for i, v in enumerate(arr.mean(0))}})
    return pd.DataFrame(rows)


def confusion_from_runs(df, pair, frac, method):
    g = df[(df.pair == pair) & (df.frac == frac) & (df.method == method)]
    return np.sum([np.array(json.loads(c)).reshape(5, 5) for c in g.cm], 0) if len(g) else np.zeros((5, 5), int)
