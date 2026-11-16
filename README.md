# HAR Transfer Lab

Cross-dataset transfer learning for Human Activity Recognition (HAR): a reproducible research pipeline **and** a Streamlit web app
that implements every item of the paper protocol and lets you test your own accelerometer data.

**Proposed method – AdaptMMD (discrepancy-guided layer selection + per-layer learning rates).**
After pre-training on the source, the layer-wise RBF-MMD² between source and target representations is computed for every layer group
(`har/transfer.py: layer_discrepancy`). With `r_l = d_l / max_l d_l`, layer *l* is fine-tuned iff `r_l ≥ ρ` (otherwise frozen, BN in eval mode)
and gets learning rate `lr_max·(floor + (1-floor)·r_l)`; the classifier head always trains. Linear CKA is used as a *diagnostic*
(representation drift between pre-trained and adapted model, `representation_drift`). Only the labelled target subset is used to compute the discrepancy,
so the label/data budget is identical to every baseline.

## Protocol → code map
| Spec item | Where |
|---|---|
| Datasets, common activities, label-mapping table | `har/labels.py` (`mapping_table()`), loaders in `har/data.py` (UCI HAR, PAMAP2, WISDM, MotionSense, HHAR) |
| 50 Hz, 128-sample windows, 50 % overlap, common 3-axis accelerometer (g) | `har/data.py` (`resample_to_fs`, `window_signal`, unit conversion in each loader) |
| Normalisation from the training split only | `har/splits.py: Normalizer`, used in `har/experiments.py` |
| Subject-wise 5-fold CV, subject-disjoint validation for early stopping | `har/splits.py` (asserts disjointness), `fit()` in `har/transfer.py` |
| 1/5/10/20/100 % target data (stratified; validation labels are subsampled with the same fraction) | `sample_fraction`, `FRACTIONS` |
| Baselines: scratch, source-only, frozen+new head, full FT (low LR), + `layerwise_lr`, DeepConvLSTM, InceptionTime(-lite) | `har/transfer.py: run_method`, `har/models.py` |
| ≥5 seeds, acc + macro-F1, mean ± std | `har/experiments.py`, `har/stats.py: summary` |
| Paired t-test and Wilcoxon vs strongest baseline | `har/stats.py: significance` |
| Identical optimiser/epochs/batch size + hyper-parameter table | `har/config.py: TrainCfg` (shown in the app) |
| Frozen-layer sweep 0…all | `--freeze_ablation` (methods `freeze_0 … freeze_G`) |
| Single vs multiple source | `--sources uci_har uci_har+pamap2` |
| Component ablations | `adapt_no_select`, `adapt_no_lrscale`, `adapt_random_select`, `adapt_inverse_select`, `adapt_reinit_head` |
| Negative transfer | `har/stats.py: negative_transfer`, `mean_discrepancy` |
| t-SNE/UMAP before/after, confusion matrices, CKA drift | page 3 of the app, `har/analysis.py`, `har/viz.py` |
| Trainable parameters and training time | recorded per run (`trainable_params`, `train_time`) |

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py                                  # pages 1-3 work immediately with the synthetic demo data
python scripts/run_experiments.py --synthetic --quick # 2 min end-to-end sanity check
```

## Real experiments (GPU recommended, e.g. Colab/Kaggle; the run is resumable)
1. Download the raw datasets (UCI HAR, WISDM v1.1, MotionSense, HHAR, PAMAP2) into `data/`.
2. Preprocess once:
```bash
python scripts/prepare_data.py --uci_har "data/UCI HAR Dataset" --wisdm data/WISDM_ar_v1.1_raw.txt \
  --motionsense data/MotionSense --hhar "data/Activity recognition exp/Phones_accelerometer.csv" --pamap2 data/PAMAP2/Protocol
```
3. Full grid (methods × 5 fractions × pairs × 5 folds × 5 seeds):
```bash
python scripts/run_experiments.py --sources uci_har uci_har+pamap2 --targets wisdm motionsense hhar --freeze_ablation
python scripts/make_report.py        # -> results/report/ (CSV + LaTeX tables, figures)
```
Commit `results/results.csv` (small) and the app's *Paper results* page will display it.

## Deploy as a website (Streamlit Community Cloud, free)
1. `git init && git add . && git commit -m "HAR Transfer Lab" && git remote add origin <your-new-repo-url> && git push -u origin main`
2. Go to share.streamlit.io → *New app* → pick the repo, branch `main`, main file `app.py` → *Deploy*.
3. Raw datasets are too big for the repo: users upload `.npz` / CSV in the app, or commit small cached files into `data/cache/`.
Live training in the app is deliberately small (few folds/seeds/epochs) because the free tier is CPU-only.

## Your own data
Page 1 → *Upload my own CSV*: pick subject/label/x/y/z columns, sampling rate and units, map your labels to the 5 common activities
(unmapped labels are dropped). It is resampled to 50 Hz, windowed, and can be used as source or target. Page 5 trains a model and
classifies a new accelerometer CSV.

## Known limitations (state these in the paper)
- Axis orientation and sensor placement differ across datasets (waist, pocket, chest); only 3-axis total acceleration in g is shared. Consider rotation augmentation.
- WISDM timestamps are ignored (nominal 20 Hz assumed); HHAR is linearly interpolated without an anti-alias filter; PAMAP2 uses the chest IMU.
- "InceptionTime-lite" (depth 3, no residual shortcut) and "DeepConvLSTM" are compact re-implementations, not the original code.
- The paired tests pair (seed, fold) runs; they share data across seeds, so p-values are optimistic – consider also testing on per-seed means.
- Reasons for negative transfer reported by the app are diagnostics (layer-wise MMD, CKA drift), not proofs; verify hypotheses (orientation, label semantics such as stairs) in the paper.
- No real-data results are shipped: the code was verified end-to-end on synthetic data only.
