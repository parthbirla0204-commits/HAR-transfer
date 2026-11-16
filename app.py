"""HAR Transfer-Learning Lab - Streamlit app (deploy on Streamlit Community Cloud)."""
import tempfile, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from har.config import TrainCfg, COMMON_ACTIVITIES, FRACTIONS, WIN
from har.data import HARData, make_synthetic, load_custom_csv, resample_to_fs, window_signal
from har.labels import mapping_table
from har.experiments import run_grid
from har.analysis import case_study, penultimate
from har.stats import summary, significance, negative_transfer, mean_discrepancy, confusion_from_runs
from har.transfer import ALL_METHODS, BASELINES, PROPOSED, predict, get_device
from har.viz import f1_vs_fraction_fig, confusion_fig, embedding_fig, embed2d, drift_fig

st.set_page_config(page_title="HAR Transfer Lab", page_icon="🏃", layout="wide")
S = st.session_state
S.setdefault("datasets", {})

st.sidebar.title("🏃 HAR Transfer Lab")
page = st.sidebar.radio("Page", ["1 · Datasets", "2 · Run experiment", "3 · Analysis", "4 · Paper results", "5 · Predict on my signal"])
st.sidebar.caption("Protocol: 50 Hz · 128-sample windows · 50 % overlap · 3-axis accelerometer (g) · subject-wise splits")


def ds_stats(d):
    return pd.DataFrame({"activity": COMMON_ACTIVITIES, "windows": [int((d.y == c).sum()) for c in range(5)]})


# ------------------------------------------------------------------------------------------------ 1
if page.startswith("1"):
    st.header("Datasets")
    st.write("Load datasets into this session. The same 5 activities are used everywhere (see mapping table).")
    st.dataframe(mapping_table(), hide_index=True, width="stretch")
    t1, t2, t3, t4 = st.tabs(["Synthetic demo", "Upload cached .npz", "Upload my own CSV", "From data/cache"])
    with t1:
        st.caption("Class-conditional synthetic signals with a controllable domain shift - good for trying the app without downloads.")
        if st.button("Create synthetic source + target"):
            S.datasets["syn_source"] = make_synthetic("syn_source", 12, 12, 0, 0.0)
            S.datasets["syn_target"] = make_synthetic("syn_target", 10, 12, 1, 1.5)
    with t2:
        st.caption("Files produced by scripts/prepare_data.py (arrays X, y, subj).")
        for f in st.file_uploader("Cached dataset(s)", type="npz", accept_multiple_files=True) or []:
            d = HARData.load(f); S.datasets[d.name] = d
    with t3:
        up = st.file_uploader("CSV with one row per sample", type="csv", key="csv")
        if up:
            df = pd.read_csv(up); st.dataframe(df.head(), width="stretch")
            c = st.columns(5); cols = list(df.columns)
            sub = c[0].selectbox("subject column", cols, index=0); lab = c[1].selectbox("label column", cols, index=min(1, len(cols) - 1))
            xs = [c[2].selectbox("x", cols, index=min(2, len(cols) - 1)), c[3].selectbox("y", cols, index=min(3, len(cols) - 1)),
                  c[4].selectbox("z", cols, index=min(4, len(cols) - 1))]
            c = st.columns(3); fs = c[0].number_input("sampling rate (Hz)", 1.0, 1000.0, 50.0); units = c[1].selectbox("units", ["g", "m/s2"])
            name = c[2].text_input("dataset name", "my_data")
            labs = sorted(df[lab].astype(str).unique())
            mp = st.data_editor(pd.DataFrame({"your label": labs, "common activity": [l if l in COMMON_ACTIVITIES else "" for l in labs]}),
                                column_config={"common activity": st.column_config.SelectboxColumn(options=[""] + COMMON_ACTIVITIES)},
                                hide_index=True, width="stretch")
            if st.button("Preprocess my dataset"):
                lm = {r["your label"]: r["common activity"] for _, r in mp.iterrows() if r["common activity"]}
                df[lab] = df[lab].astype(str)
                try:
                    S.datasets[name] = load_custom_csv(df, fs, units, lm, sub, lab, tuple(xs), None, name)
                    st.success("Done")
                except Exception as e:
                    st.error(e)
    with t4:
        files = sorted(Path("data/cache").glob("*.npz"))
        pick = st.multiselect("Cached files in this deployment", [f.stem for f in files])
        if st.button("Load selected") and pick:
            for p in pick:
                S.datasets[p] = HARData.load(Path("data/cache") / f"{p}.npz")
    st.subheader("Loaded")
    for n, d in S.datasets.items():
        with st.expander(f"{n}: {len(d)} windows, {len(set(d.subj))} subjects"):
            a, b = st.columns(2); a.dataframe(ds_stats(d), hide_index=True)
            fig, ax = plt.subplots(figsize=(5, 2.6))
            for c in range(5):
                if (d.y == c).any():
                    ax.plot(d.X[np.where(d.y == c)[0][0], 0], label=COMMON_ACTIVITIES[c], lw=.9)
            ax.legend(fontsize=6, ncol=3); ax.set_title("x-axis, one window per class", fontsize=9); b.pyplot(fig)
            if st.button("Remove", key=f"rm{n}"):
                del S.datasets[n]; st.rerun()

# ------------------------------------------------------------------------------------------------ 2
elif page.startswith("2"):
    st.header("Run experiment (live, small scale)")
    if len(S.datasets) < 2:
        st.info("Load at least two datasets (a source and a target) on page 1."); st.stop()
    names = list(S.datasets)
    c = st.columns(2)
    src = c[0].multiselect("Source dataset(s) (multi-select = multiple sources)", names, default=names[:1])
    tgt = c[1].selectbox("Target dataset", [n for n in names if n not in src] or names)
    fr = st.multiselect("Target label fractions", FRACTIONS, default=[0.05, 0.2, 1.0], format_func=lambda x: f"{x:.0%}")
    ms = st.multiselect("Methods", ALL_METHODS, default=["scratch", "source_only", "frozen_head", "full_ft", PROPOSED])
    c = st.columns(3); folds = c[0].slider("CV folds", 2, 5, 3); seeds = c[1].slider("Seeds", 1, 5, 2); ep = c[2].slider("Max epochs", 3, 40, 12)
    st.caption("The paper protocol uses 5 folds × 5 seeds × 40 epochs - run that with scripts/run_experiments.py on a GPU.")
    if st.button("Run", type="primary") and src and fr and ms:
        tmp = Path(tempfile.mkdtemp()) / "live.csv"; bar = st.progress(0.0); msg = st.empty()
        cfg = TrainCfg(max_epochs=ep)
        def prog(n, tot, m): bar.progress(n / tot); msg.text(m)
        S.live = run_grid(S.datasets, [("+".join(src), tgt)], str(tmp), fr, ms, folds, range(seeds), cfg, get_device(), progress=prog)
        S.live_cfg = dict(src="+".join(src), tgt=tgt)
    if "live" in S:
        df = S.live; sm = summary(df); pair = df.pair.iloc[0]
        st.subheader("Mean ± std over seeds")
        st.dataframe(sm.pivot(index="method", columns="frac", values="macro_f1"), width="stretch")
        st.pyplot(f1_vs_fraction_fig(sm, pair))
        sig = significance(df)
        if len(sig):
            st.subheader("Paired tests: proposed vs strongest baseline"); st.dataframe(sig, hide_index=True, width="stretch")
        st.subheader("Trainable parameters & training time"); st.dataframe(df.groupby("method")[["trainable_params", "train_time", "epochs"]].mean().round(2))
        st.download_button("Download raw results CSV", df.to_csv(index=False), "results.csv")

# ------------------------------------------------------------------------------------------------ 3
elif page.startswith("3"):
    st.header("Analysis: embeddings, confusion matrices, representation drift")
    if len(S.datasets) < 2:
        st.info("Load at least two datasets on page 1."); st.stop()
    names = list(S.datasets); c = st.columns(4)
    src = c[0].multiselect("Source(s)", names, default=names[:1]); tgt = c[1].selectbox("Target", [n for n in names if n not in src] or names)
    frac = c[2].select_slider("Target fraction", FRACTIONS, value=0.1, format_func=lambda x: f"{x:.0%}"); ep = c[3].slider("Epochs", 3, 40, 12)
    if st.button("Run case study", type="primary") and src:
        with st.spinner("Training..."):
            S.case = case_study(S.datasets, "+".join(src), tgt, frac, 0, 0, ["scratch", "source_only", "full_ft", PROPOSED], TrainCfg(max_epochs=ep), get_device())
            S.case_meta = (src, tgt, frac)
    if "case" in S:
        cs = S.case; dev = get_device(); res = cs["results"]
        st.subheader("Confusion matrices (target test subjects)")
        cols = st.columns(len(res))
        for col, (m, r) in zip(cols, res.items()):
            cm = np.zeros((5, 5), int); np.add.at(cm, (cs["yte"], r["pred"]), 1)
            col.pyplot(confusion_fig(cm, f"{m}  F1={r['macro_f1']:.3f}"))
        st.subheader("Feature space before vs after adaptation")
        n = min(600, len(cs["yte"])); idx = np.random.RandomState(0).choice(len(cs["yte"]), n, replace=False)
        Xs = cs["Xsrc"][np.random.RandomState(0).choice(len(cs["Xsrc"]), min(300, len(cs["Xsrc"])), replace=False)]
        c1, c2 = st.columns(2)
        for col, (title, model) in zip((c1, c2), (("Before: source model", cs["source_model"]), (f"After: {PROPOSED}", res[PROPOSED]["model"]))):
            Ft = penultimate(model.to(dev), cs["Xte"][idx], dev); Fs = penultimate(model, Xs, dev)
            Z, kind = embed2d(np.concatenate([Ft, Fs]))
            from har.transfer import predict as _p
            ys = _p(cs["source_model"].to(dev), Xs, dev)   # source labels are not stored in the case; colour source points by source-model prediction
            col.pyplot(embedding_fig(Z, np.concatenate([cs["yte"][idx], ys]), f"{title} ({kind})", np.r_[np.zeros(n), np.ones(len(Xs))]))
        st.caption("● target test windows, ✕ source windows (coloured by the source model's prediction). Colour = activity.")
        st.subheader("Representation drift per layer (CKA between pretrained and adapted model)")
        st.pyplot(drift_fig(cs["cka_drift"], cs["layer_names"]))
        st.subheader("Cost per method")
        st.dataframe(pd.DataFrame({m: {"trainable params": r["trainable_params"], "train time (s)": round(r["train_time"], 2), "epochs": r["epochs"]}
                                   for m, r in res.items()}).T)
        if PROPOSED in res and "disc" in res[PROPOSED]:
            d = json.loads(res[PROPOSED]["disc"]); st.write("Layer-wise MMD² (source vs target):", dict(zip(cs["layer_names"], np.round(d, 4))),
                                                          "| layers fine-tuned:", [cs["layer_names"][i] for i in json.loads(res[PROPOSED]["selected"])])

# ------------------------------------------------------------------------------------------------ 4
elif page.startswith("4"):
    st.header("Paper results (full protocol)")
    st.write("Produced by `scripts/run_experiments.py`. Auto-loads `results/results.csv` from the repo, or upload your own.")
    up = st.file_uploader("results.csv", type="csv"); p = Path("results/results.csv")
    df = pd.read_csv(up) if up else (pd.read_csv(p) if p.exists() else None)
    t_h, t_m = st.tabs(["Hyper-parameters", "Label mapping"])
    t_h.dataframe(pd.DataFrame(TrainCfg().table()), hide_index=True); t_m.dataframe(mapping_table(), hide_index=True)
    if df is None:
        st.info("No results yet."); st.stop()
    sm = summary(df); pair = st.selectbox("Source→target pair", sorted(df.pair.unique()))
    metric = st.radio("Metric", ["macro_f1", "acc"], horizontal=True)
    st.dataframe(sm[sm.pair == pair].pivot(index="method", columns="frac", values=metric), width="stretch")
    st.pyplot(f1_vs_fraction_fig(sm, pair, metric="f1" if metric == "macro_f1" else "acc"))
    st.subheader("Paired t-test / Wilcoxon vs strongest baseline"); st.dataframe(significance(df, metric=metric), hide_index=True)
    st.subheader("Negative transfer (transfer method worse than scratch)"); nt = negative_transfer(df, metric=metric)
    st.dataframe(nt[nt.negative], hide_index=True) if len(nt) else None
    st.subheader("Layer-wise discrepancy per pair (higher = larger domain gap)"); st.dataframe(mean_discrepancy(df), hide_index=True)
    st.subheader("Ablations (100 % and 1 % target data)")
    ab = sm[sm.method.str.startswith("adapt") | sm.method.str.startswith("freeze_") | (sm.method == "full_ft")]
    st.dataframe(ab[ab.pair == pair].pivot(index="method", columns="frac", values=metric), width="stretch")
    st.subheader("Parameters & time"); st.dataframe(df.groupby("method")[["trainable_params", "train_time", "epochs"]].mean().round(2))
    m = st.selectbox("Confusion matrix of", sorted(df.method.unique())); fr = st.selectbox("fraction", sorted(df.frac.unique()), index=len(df.frac.unique()) - 1)
    st.pyplot(confusion_fig(confusion_from_runs(df, pair, fr, m), f"{pair} · {m} · {fr:.0%}"))

# ------------------------------------------------------------------------------------------------ 5
else:
    st.header("Predict activities on your own accelerometer signal")
    if len(S.datasets) < 1:
        st.info("Load a dataset on page 1 first (the model is trained on it)."); st.stop()
    names = list(S.datasets); c = st.columns(3)
    src = c[0].multiselect("Pre-train on (optional)", names); tgt = c[1].selectbox("Fine-tune on", names); ep = c[2].slider("Epochs", 3, 40, 12)
    if st.button("Train deployable model") and tgt:
        with st.spinner("Training..."):
            if src and tgt not in src:
                cs = case_study(S.datasets, "+".join(src), tgt, 1.0, 0, 0, [PROPOSED], TrainCfg(max_epochs=ep), get_device())
                S.deploy = (cs["results"][PROPOSED]["model"], cs["norm"], f"{'+'.join(src)} → {tgt} ({PROPOSED})", cs["results"][PROPOSED]["macro_f1"])
            else:
                other = {k: v for k, v in S.datasets.items() if k != tgt}
                cs = case_study({**S.datasets, "_dummy": S.datasets[tgt]}, "_dummy", tgt, 1.0, 0, 0, ["scratch"], TrainCfg(max_epochs=ep), get_device())
                S.deploy = (cs["results"]["scratch"]["model"], cs["norm"], f"scratch on {tgt}", cs["results"]["scratch"]["macro_f1"])
    if "deploy" in S:
        model, norm, desc, f1 = S.deploy; st.success(f"Model: {desc} — held-out-subject macro-F1 {f1:.3f}")
        up = st.file_uploader("CSV with columns x,y,z (optional: time in seconds)", type="csv", key="pred")
        c = st.columns(3); fs = c[0].number_input("sampling rate (Hz)", 1.0, 1000.0, 50.0); units = c[1].selectbox("units", ["g", "m/s2"], key="u2")
        if up:
            df = pd.read_csv(up); xyz = df[["x", "y", "z"]].values.astype(float) * (1 / 9.80665 if units == "m/s2" else 1)
            t = df["time"].values if "time" in df else np.arange(len(df)) / fs
            W = window_signal(resample_to_fs(t, xyz))
            if len(W) == 0:
                st.error(f"Need at least {WIN} samples at 50 Hz (≈2.6 s)."); st.stop()
            p = predict(model, norm(W), get_device()); labs = [COMMON_ACTIVITIES[i] for i in p]
            st.write("**Majority activity:**", pd.Series(labs).mode()[0])
            fig, ax = plt.subplots(figsize=(8, 2)); ax.step(np.arange(len(p)) * 1.28, p, where="post"); ax.set_yticks(range(5)); ax.set_yticklabels(COMMON_ACTIVITIES)
            ax.set_xlabel("time (s)"); st.pyplot(fig); st.dataframe(pd.Series(labs).value_counts().rename("windows"))
