"""Dataset loaders + shared preprocessing (50 Hz, 128-sample windows, 50 % overlap, 3-axis accelerometer in g)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .config import FS, WIN, STEP, G, COMMON_ACTIVITIES
from .labels import LABEL_MAP, to_index


@dataclass
class HARData:
    X: np.ndarray      # (N, 3, 128) float32, in g
    y: np.ndarray      # (N,) int, index into COMMON_ACTIVITIES
    subj: np.ndarray   # (N,) str, globally unique subject id ("<dataset>_<id>")
    name: str

    def __len__(self):
        return len(self.y)

    def save(self, path):
        np.savez_compressed(path, X=self.X, y=self.y, subj=self.subj.astype(str), name=self.name)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return HARData(z["X"].astype(np.float32), z["y"].astype(int), z["subj"].astype(str), str(z["name"]))


def concat(datasets, name=None):
    return HARData(np.concatenate([d.X for d in datasets]), np.concatenate([d.y for d in datasets]),
                   np.concatenate([d.subj for d in datasets]), name or "+".join(d.name for d in datasets))


# ------------------------------------------------------------------ signal utilities
def resample_to_fs(t, xyz, fs=FS):
    """Linear interpolation of (T,3) samples at times t (seconds, increasing) onto a uniform fs grid."""
    t, idx = np.unique(t, return_index=True)
    xyz = xyz[idx]
    if len(t) < 2:
        return np.empty((0, 3))
    tn = np.arange(t[0], t[-1], 1.0 / fs)
    return np.stack([np.interp(tn, t, xyz[:, i]) for i in range(3)], 1)


def window_signal(sig, win=WIN, step=STEP):
    n = (len(sig) - win) // step + 1
    if n <= 0:
        return np.empty((0, 3, win), np.float32)
    idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
    return sig[idx].transpose(0, 2, 1).astype(np.float32)


def split_on_gaps(t, max_gap=1.0):
    cuts = np.where(np.diff(t) > max_gap)[0] + 1
    return np.split(np.arange(len(t)), cuts)


def runs(labels):
    """Contiguous runs of equal values -> list of (start, end, value)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cuts = np.where(labels[1:] != labels[:-1])[0] + 1
    st = np.r_[0, cuts]; en = np.r_[cuts, len(labels)]
    return [(a, b, labels[a]) for a, b in zip(st, en)]


def build(segments, name):
    """segments: iterable of (subject, common_label_name|None, t|None, xyz[g], fs_in)."""
    X, y, s = [], [], []
    for subj, lab, t, xyz, fs_in in segments:
        li = to_index(lab) if lab is not None else None
        if li is None or len(xyz) < 2:
            continue
        if t is None and fs_in == FS:
            sig = xyz
        else:
            sig = resample_to_fs(np.arange(len(xyz)) / fs_in if t is None else t, xyz)
        w = window_signal(sig)
        if len(w):
            X.append(w); y.append(np.full(len(w), li)); s.append(np.full(len(w), f"{name}_{subj}"))
    if not X:
        raise ValueError(f"No windows produced for {name}; check paths / label names.")
    return HARData(np.concatenate(X), np.concatenate(y), np.concatenate(s), name)


# ------------------------------------------------------------------ loaders
def _txt(p):
    return pd.read_csv(p, sep=r"\s+", header=None).values


def load_uci_har(root):
    """root = 'UCI HAR Dataset'. Uses total_acc (gravity included, g); windows are already 50 Hz/128/50 %."""
    root = Path(root); Xs, ys, ss = [], [], []
    for split in ("train", "test"):
        d = root / split
        X = np.stack([_txt(d / "Inertial Signals" / f"total_acc_{a}_{split}.txt") for a in "xyz"], 1)
        y = _txt(d / f"y_{split}.txt").ravel().astype(int)
        s = _txt(d / f"subject_{split}.txt").ravel().astype(int)
        names = [LABEL_MAP["uci_har"][v] for v in y]
        keep = np.array([n is not None for n in names])
        Xs.append(X[keep])
        ys.append(np.array([to_index(n) for n in np.array(names, dtype=object)[keep]], dtype=int))
        ss.append(np.array([f"uci_har_{v}" for v in s[keep]]))
    return HARData(np.concatenate(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(ss), "uci_har")


def load_wisdm(path):
    """WISDM_ar_v1.1_raw.txt: user,activity,timestamp,x,y,z; (20 Hz, m/s^2)."""
    rows = []
    for line in open(path, errors="ignore"):
        for rec in line.strip().split(";"):
            p = rec.split(",")
            if len(p) >= 6:
                try:
                    rows.append((p[0], p[1], float(p[3]), float(p[4]), float(p[5])))
                except ValueError:
                    pass
    df = pd.DataFrame(rows, columns=["u", "a", "x", "y", "z"])
    segs = []
    for (u, a), g in df.groupby(["u", "a"], sort=False):
        segs.append((u, LABEL_MAP["wisdm"].get(a), None, g[["x", "y", "z"]].values / G, 20))
    return build(segs, "wisdm")


def load_motionsense(root):
    """root = folder containing 'A_DeviceMotion_data' (or the activity folders directly). 50 Hz, g."""
    root = Path(root)
    if (root / "A_DeviceMotion_data").exists():
        root = root / "A_DeviceMotion_data"
    segs = []
    for folder in sorted(root.glob("*_*")):
        if not folder.is_dir():
            continue
        act = LABEL_MAP["motionsense"].get(folder.name.split("_")[0])
        for f in folder.glob("sub_*.csv"):
            df = pd.read_csv(f)
            acc = df[[f"userAcceleration.{a}" for a in "xyz"]].values + df[[f"gravity.{a}" for a in "xyz"]].values
            segs.append((f.stem.split("_")[1], act, None, acc, 50))
    return build(segs, "motionsense")


def load_hhar(path):
    """Phones_accelerometer.csv from 'Activity recognition exp'; irregular 50-200 Hz, m/s^2."""
    df = pd.read_csv(path, usecols=["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"])
    df = df[df.gt.isin(LABEL_MAP["hhar"])]
    segs = []
    for (u, m, d, gt), g in df.groupby(["User", "Model", "Device", "gt"], sort=False):
        g = g.sort_values("Creation_Time")
        t = g.Creation_Time.values.astype(float) / 1e9
        xyz = g[["x", "y", "z"]].values / G
        for idx in split_on_gaps(t):
            if len(idx) > 10:
                segs.append((u, LABEL_MAP["hhar"][gt], t[idx], xyz[idx], None))
    return build(segs, "hhar")


def load_pamap2(root):
    """root = 'PAMAP2_Dataset/Protocol'. Chest IMU +-16g accelerometer (cols 21-23), 100 Hz, m/s^2."""
    segs = []
    for f in sorted(Path(root).glob("subject*.dat")):
        df = pd.read_csv(f, sep=r"\s+", header=None, usecols=[1, 21, 22, 23]).dropna()
        sid = f.stem.replace("subject", "")
        for a, b, act in runs(df[1].values.astype(int)):
            segs.append((sid, LABEL_MAP["pamap2"].get(int(act)), None, df.iloc[a:b, 1:4].values / G, 100))
    return build(segs, "pamap2")


def load_custom_csv(df, fs, units="g", label_map=None, subject_col="subject", label_col="label",
                    cols=("x", "y", "z"), time_col=None, name="custom"):
    """Your own recordings: one row per sample. label_map maps your labels -> common activity names."""
    scale = 1 / G if units.lower().startswith("m") else 1.0
    segs = []
    for subj, g in df.groupby(subject_col, sort=False):
        for a, b, lab in runs(g[label_col].values):
            lab_c = (label_map or {}).get(lab, lab if lab in COMMON_ACTIVITIES else None)
            t = g[time_col].values[a:b] if time_col else None
            segs.append((subj, lab_c, t, g.iloc[a:b][list(cols)].values.astype(float) * scale, fs))
    return build(segs, name)


LOADERS = {"uci_har": load_uci_har, "wisdm": load_wisdm, "motionsense": load_motionsense,
           "hhar": load_hhar, "pamap2": load_pamap2}


def load_dataset(name, path, cache_dir="data/cache"):
    cache = Path(cache_dir) / f"{name}.npz"
    if cache.exists():
        return HARData.load(cache)
    d = LOADERS[name](path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.save(cache)
    return d


# ------------------------------------------------------------------ synthetic data (demo + tests)
def make_synthetic(name="synthetic", n_subjects=12, windows_per_class=20, seed=0, shift=0.0):
    """Class-conditional signals with a per-dataset domain shift (axis rotation, scale, noise)."""
    rng = np.random.RandomState(seed)
    t = np.arange(WIN) / FS
    base = {0: (1.9, .55), 1: (1.6, .75), 2: (2.4, .95), 3: (0, 0), 4: (0, 0)}
    grav = {0: [0, 0, 1], 1: [0, .1, .98], 2: [0, -.1, .98], 3: [.35, 0, .93], 4: [0.02, 0, 1.0]}
    ang = shift * np.pi / 6
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    X, y, s = [], [], []
    for sub in range(n_subjects):
        f_j = rng.uniform(.9, 1.1); a_j = rng.uniform(.85, 1.15)
        for c in range(5):
            f, a = base[c]
            for _ in range(windows_per_class):
                ph = rng.uniform(0, 2 * np.pi, 3)
                sig = np.stack([a * a_j * np.sin(2 * np.pi * f * f_j * t + ph[i]) * (0.6 + .4 * i / 2) for i in range(3)]) \
                    if f else np.zeros((3, WIN))
                sig = sig + np.array(grav[c])[:, None] + rng.randn(3, WIN) * (.05 + .03 * shift)
                X.append((R @ sig) * (1 + .1 * shift)); y.append(c); s.append(f"{name}_{sub}")
    return HARData(np.array(X, np.float32), np.array(y), np.array(s), name)
