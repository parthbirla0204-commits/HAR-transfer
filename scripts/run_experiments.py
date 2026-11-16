"""Run the full grid (methods x fractions x pairs x folds x seeds). Resumable: re-run the same command to continue.
Quick check : python scripts/run_experiments.py --synthetic --quick
Paper run   : python scripts/run_experiments.py --sources uci_har uci_har+pamap2 --targets wisdm motionsense hhar
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from har.config import TrainCfg, FRACTIONS
from har.data import HARData, make_synthetic
from har.experiments import run_grid
from har.transfer import ALL_METHODS, BASELINES, PROPOSED, get_device
from har.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/cache"); ap.add_argument("--out", default="results/results.csv")
ap.add_argument("--sources", nargs="+", default=["uci_har"]); ap.add_argument("--targets", nargs="+", default=["wisdm", "motionsense", "hhar"])
ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
ap.add_argument("--freeze_ablation", action="store_true", help="add freeze_0..freeze_G methods (number of frozen layers)")
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--quick", action="store_true", help="2 folds, 2 seeds, 6 epochs")
a = ap.parse_args()
if a.quick:
    a.folds, a.seeds, a.epochs = 2, 2, 6
names = {n for s in a.sources for n in s.split("+")} | set(a.targets)
if a.synthetic:
    datasets = {n: make_synthetic(n, 12, 10, seed=i, shift=float(i)) for i, n in enumerate(sorted(names))}
else:
    datasets = {n: HARData.load(Path(a.cache) / f"{n}.npz") for n in names}
methods = list(a.methods)
if a.freeze_ablation:
    methods += [f"freeze_{k}" for k in range(len(build_model('resnet1d').groups) + 1)]
cfg = TrainCfg(max_epochs=a.epochs)
pairs = [(s, t) for s in a.sources for t in a.targets if t not in s.split("+")]
print("device:", get_device(), "| pairs:", pairs)
run_grid(datasets, pairs, a.out, a.fractions, methods, a.folds, range(a.seeds), cfg)
print("done ->", a.out)
