"""Matplotlib figures used by the app and the report script."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import COMMON_ACTIVITIES


def confusion_fig(cm, title=""):
    cm = np.asarray(cm, float); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(COMMON_ACTIVITIES, rotation=45, ha="right"); ax.set_yticklabels(COMMON_ACTIVITIES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="w" if cmn[i, j] > .5 else "k", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); return fig


def embedding_fig(Z, y, title="", domain=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for c, name in enumerate(COMMON_ACTIVITIES):
        m = y == c
        if domain is None:
            ax.scatter(Z[m, 0], Z[m, 1], s=8, label=name, alpha=.7)
        else:
            for dom, mk in ((0, "o"), (1, "x")):
                mm = m & (domain == dom)
                ax.scatter(Z[mm, 0], Z[mm, 1], s=10, marker=mk, color=f"C{c}", alpha=.6, label=name if dom == 0 else None)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout(); return fig


def embed2d(F, seed=0):
    """UMAP if installed, otherwise t-SNE."""
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(F), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(2, perplexity=min(30, max(5, len(F) // 4)), random_state=seed, init="pca").fit_transform(F), "t-SNE"


def f1_vs_fraction_fig(summ, pair, methods=None, metric="f1"):
    s = summ[summ.pair == pair]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in (methods or sorted(s.method.unique())):
        g = s[s.method == m].sort_values("frac")
        if len(g):
            ax.errorbar(g.frac * 100, g[f"{metric}_mean"], yerr=g[f"{metric}_std"].fillna(0), marker="o", capsize=2,
                        label=m, lw=2.2 if m == "adapt_mmd" else 1.1)
    ax.set_xscale("log"); ax.set_xlabel("% of target training data"); ax.set_ylabel("macro-F1" if metric == "f1" else "accuracy")
    ax.set_title(pair); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); return fig


def drift_fig(cka, names):
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.bar(names, cka); ax.set_ylim(0, 1.05); ax.set_ylabel("CKA (pretrained vs adapted)")
    fig.tight_layout(); return fig
