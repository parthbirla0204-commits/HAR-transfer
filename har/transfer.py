"""Training loop, baselines, and the proposed method (AdaptMMD: discrepancy-guided layer selection + per-layer LR)."""
import copy, time, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from .config import TrainCfg
from .models import build_model, count_params


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X, device, bs=1024):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.as_tensor(X[i:i + bs], device=device)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], int)


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0))}


def fit(model, Xtr, ytr, Xva, yva, param_groups, trainable, cfg: TrainCfg, device, seed):
    """Identical optimiser / epochs / batch size / early stopping for every method.
    `trainable`: set of group indices that are updated (others frozen and kept in eval mode, so BN stats stay fixed)."""
    seed_all(seed); model.to(device)
    opt = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    Xt, yt = torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device)
    best, best_state, bad, ep_done = -1, copy.deepcopy(model.state_dict()), 0, 0
    t0 = time.time(); lossf = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(seed)
    for ep in range(cfg.max_epochs):
        model.train()
        for i, gm in enumerate(model.groups):
            if i not in trainable:
                gm.eval()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        for b in range(0, len(Xt), cfg.batch_size):
            idx = perm[b:b + cfg.batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(); lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
        ep_done = ep + 1
        f1 = metrics(yva, predict(model, Xva, device))["macro_f1"] if len(yva) else ep
        if f1 > best:
            best, bad, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return {"epochs": ep_done, "train_time": time.time() - t0}


def set_trainable(model, groups, head=True):
    for i, gm in enumerate(model.groups):
        for p in gm.parameters():
            p.requires_grad = i in groups
    for p in model.head.parameters():
        p.requires_grad = head


# ------------------------------------------------------------------ discrepancy measures
def rbf_mmd2(A, B):
    """Biased MMD^2 with an RBF kernel; bandwidth = median pairwise distance (scale invariant)."""
    Z = torch.cat([A, B]); D = torch.cdist(Z, Z) ** 2
    bw = D[D > 0].median() if (D > 0).any() else torch.tensor(1.0)
    K = torch.exp(-D / (bw + 1e-8)); n = len(A)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def linear_cka(X, Y):
    """Linear CKA between paired representations X (n,d1), Y (n,d2)."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (den + 1e-12))


@torch.no_grad()
def layer_features(model, X, device):
    model.eval()
    _, f = model(torch.as_tensor(X, device=device), return_feats=True)
    return [z.cpu() for z in f]


def layer_discrepancy(model, Xs, Xt, device, n=512, seed=0):
    """Per-layer MMD^2 between source and target representations (unpaired, so n_s != n_t is fine)."""
    rng = np.random.RandomState(seed)
    Xs = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    Xt = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)]
    fs, ft = layer_features(model, Xs, device), layer_features(model, Xt, device)
    return np.array([rbf_mmd2(a, b) for a, b in zip(fs, ft)])


def representation_drift(model_a, model_b, X, device):
    """Per-layer linear CKA between two models on the same inputs (1 = representation unchanged)."""
    fa, fb = layer_features(model_a, X, device), layer_features(model_b, X, device)
    return [linear_cka(a.numpy(), b.numpy()) for a, b in zip(fa, fb)]


# ------------------------------------------------------------------ source pre-training
def pretrain(arch, Xtr, ytr, Xva, yva, cfg, device, seed):
    m = build_model(arch)
    set_trainable(m, set(range(len(m.groups))))
    info = fit(m, Xtr, ytr, Xva, yva, [{"params": m.parameters(), "lr": cfg.lr_pretrain}],
               set(range(len(m.groups))), cfg, device, seed)
    return m, info


# ------------------------------------------------------------------ adaptation strategies
PROPOSED = "adapt_mmd"
BASELINES = ["scratch", "source_only", "frozen_head", "full_ft", "layerwise_lr", "deepconvlstm", "inceptiontime"]
ABLATIONS = ["adapt_no_select", "adapt_no_lrscale", "adapt_random_select", "adapt_inverse_select", "adapt_reinit_head"]
ALL_METHODS = BASELINES + [PROPOSED] + ABLATIONS


def _reinit_head(m, seed):
    torch.manual_seed(seed); m.head.reset_parameters()


def adapt_plan(method, disc, cfg, seed):
    """Returns (trainable group set, per-group LR list) for the proposed method and its ablations."""
    G = len(disc); r = disc / (disc.max() + 1e-12)
    sel = {i for i in range(G) if r[i] >= cfg.rho} or {int(r.argmax())}
    lr_scaled = [cfg.lr_max_adapt * (cfg.lr_floor + (1 - cfg.lr_floor) * r[i]) for i in range(G)]
    lr_flat = [cfg.lr_max_adapt] * G
    if method == "adapt_no_select":
        return set(range(G)), lr_scaled
    if method == "adapt_no_lrscale":
        return sel, lr_flat
    if method == "adapt_random_select":
        rng = np.random.RandomState(seed); return set(rng.choice(G, len(sel), replace=False).tolist()), lr_flat
    if method == "adapt_inverse_select":
        order = np.argsort(r); return set(order[:len(sel)].tolist()), lr_flat
    return sel, lr_scaled            # adapt_mmd, adapt_reinit_head


def run_method(method, src_model, arch_src, Xtr, ytr, Xva, yva, Xte, yte, Xsrc, cfg, device, seed):
    """All inputs are already normalised. Xtr/ytr = the labelled target fraction used for fine-tuning."""
    extra = {}
    t_extra = 0.0
    if method in ("scratch", "deepconvlstm", "inceptiontime"):
        arch = arch_src if method == "scratch" else method
        m = build_model(arch); G = len(m.groups); tr = set(range(G)); set_trainable(m, tr)
        pg = [{"params": m.parameters(), "lr": cfg.lr_scratch}]
    else:
        m = copy.deepcopy(src_model); G = len(m.groups)
        if method == "source_only":
            set_trainable(m, set(), head=False)
            p = predict(m.to(device), Xte, device)
            return {**metrics(yte, p), "trainable_params": 0, "train_time": 0.0, "epochs": 0, "pred": p}
        if method == "frozen_head":
            _reinit_head(m, seed); tr = set(); set_trainable(m, tr)
            pg = [{"params": m.head.parameters(), "lr": cfg.lr_head}]
        elif method == "full_ft":
            tr = set(range(G)); set_trainable(m, tr)
            pg = [{"params": m.parameters(), "lr": cfg.lr_finetune}]
        elif method == "layerwise_lr":       # discriminative fine-tuning: lr decays towards the input
            tr = set(range(G)); set_trainable(m, tr)
            pg = [{"params": m.groups[i].parameters(), "lr": cfg.lr_finetune * cfg.lr_decay ** (G - 1 - i)} for i in range(G)]
            pg.append({"params": m.head.parameters(), "lr": cfg.lr_finetune})
        elif method.startswith("freeze_"):   # ablation: first k groups frozen, rest + head fine-tuned (low LR)
            k = int(method.split("_")[1]); tr = set(range(k, G)); set_trainable(m, tr)
            pg = [{"params": m.groups[i].parameters(), "lr": cfg.lr_finetune} for i in tr] + \
                 [{"params": m.head.parameters(), "lr": cfg.lr_finetune}]
        elif method == PROPOSED or method in ABLATIONS:
            t0 = time.time()
            disc = layer_discrepancy(m.to(device), Xsrc, Xtr, device, cfg.mmd_samples, seed)
            tr, lrs = adapt_plan(method, disc, cfg, seed)
            if method == "adapt_reinit_head":
                _reinit_head(m, seed)
            set_trainable(m, tr)
            pg = [{"params": m.groups[i].parameters(), "lr": lrs[i]} for i in sorted(tr)] + \
                 [{"params": m.head.parameters(), "lr": cfg.lr_max_adapt}]
            t_extra = time.time() - t0
            extra = {"disc": json.dumps(np.round(disc, 5).tolist()), "selected": json.dumps(sorted(int(i) for i in tr))}
        else:
            raise ValueError(method)
    n_train = count_params(m)
    info = fit(m, Xtr, ytr, Xva, yva, pg, tr, cfg, device, seed)
    p = predict(m, Xte, device)
    return {**metrics(yte, p), "trainable_params": n_train, "train_time": info["train_time"] + t_extra,
            "epochs": info["epochs"], "pred": p, "model": m, **extra}
