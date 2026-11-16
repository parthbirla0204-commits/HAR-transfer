"""Turn results/results.csv into paper tables (CSV + LaTeX) and figures in results/report/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from har.config import TrainCfg
from har.labels import mapping_table
from har.stats import summary, significance, negative_transfer, mean_discrepancy, confusion_from_runs
from har.viz import f1_vs_fraction_fig, confusion_fig
from har.transfer import PROPOSED

src = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
out = Path("results/report"); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(src); S = summary(df)
S.to_csv(out / "summary.csv", index=False)
mapping_table().to_latex(out / "label_mapping.tex", index=False); mapping_table().to_csv(out / "label_mapping.csv", index=False)
pd.DataFrame(TrainCfg().table()).to_csv(out / "hyperparameters.csv", index=False)
for pair in S.pair.unique():
    tab = S[S.pair == pair].pivot(index="method", columns="frac", values="macro_f1")
    tab.to_latex(out / f"table_f1_{pair.replace('>', '').replace('-', '_')}.tex")
    f1_vs_fraction_fig(S, pair).savefig(out / f"f1_vs_fraction_{pair.replace('>', '').replace('-', '_')}.png", dpi=200)
    for m in ("scratch", "full_ft", PROPOSED):
        confusion_fig(confusion_from_runs(df, pair, 1.0, m), f"{pair} {m} (100 %)").savefig(
            out / f"cm_{pair.replace('>', '').replace('-', '_')}_{m}.png", dpi=200)
significance(df).to_csv(out / "significance_vs_strongest_baseline.csv", index=False)
negative_transfer(df).to_csv(out / "negative_transfer.csv", index=False)
mean_discrepancy(df).to_csv(out / "layer_discrepancy.csv", index=False)
cost = df.groupby("method")[["trainable_params", "train_time", "epochs"]].mean().round(2)
cost.to_csv(out / "params_and_time.csv")
print("report written to", out)
