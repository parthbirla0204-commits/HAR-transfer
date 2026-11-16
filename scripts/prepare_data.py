"""Convert raw datasets to cached 50 Hz / 128-sample windows (data/cache/<name>.npz).
Example:
  python scripts/prepare_data.py --uci_har "data/UCI HAR Dataset" --wisdm data/WISDM_ar_v1.1_raw.txt \
      --motionsense data/MotionSense --hhar "data/Activity recognition exp/Phones_accelerometer.csv" --pamap2 data/PAMAP2/Protocol
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from har.data import LOADERS, load_dataset

ap = argparse.ArgumentParser()
for k in LOADERS:
    ap.add_argument(f"--{k}", help=f"path to raw {k}")
ap.add_argument("--cache", default="data/cache")
a = ap.parse_args()
for k in LOADERS:
    if getattr(a, k):
        d = load_dataset(k, getattr(a, k), a.cache)
        print(f"{k}: windows={len(d)} subjects={len(set(d.subj))} class counts={[(d.y == c).sum() for c in range(5)]}")
