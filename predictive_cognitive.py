import os
import glob
import pickle
import random
import argparse
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Optional Streamlit import for GUI mode only
try:
    import streamlit as st  # type: ignore
except Exception:  # ImportError or envs without GUI libs
    st = None

from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
from lime.lime_tabular import LimeTabularExplainer

# =============================================================================
# 0) CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PREDICT_DATA_DIR", os.path.join(BASE_DIR, "data"))

CSV_DIR = os.path.join(DATA_DIR, "data_csv")
SPLITS = ["train", "valid", "test"]  # your folders are 'train', 'test', 'valid'

# CSV files (PHQ8_Score target lives here)
CSV_TRAIN = os.path.join(CSV_DIR, "train_split_Depression_AVEC2017.csv")
CSV_DEV   = os.path.join(CSV_DIR, "dev_split_Depression_AVEC2017.csv")   # using as "valid"
CSV_TEST  = os.path.join(CSV_DIR, "full_test_split.csv")                 # has PHQ_Score

# Which modalities to use (will load if file exists)
MODALITIES = [
    "audio",
    "fkps",
    "gaze_conf",
    "pose_conf",
    "text",
    # If you later add precomputed image features, add "image" and file pattern below.
]

# File pattern helpers (covers both flat and nested layouts)
def candidate_paths(split: str, pid: int, modality: str) -> List[str]:
    """
    Return possible filepaths for a given split/pid/modality.
    Supports flat and nested layouts and multiple naming conventions for the valid/dev split:
      - <DATA>/<split>/<prefix>_ft_<modality>_<pid>.npy
      - <DATA>/<split>/<pid>/<prefix>_ft_<modality>_<pid>.npy
    Where prefix is one of: train | valid | dev | test (depending on split).
    """
    paths: List[str] = []

    if split == "train":
        base_dirs = [os.path.join(DATA_DIR, "train")]
        prefixes = ["train"]
    elif split == "valid":
        # Support both directory names and file prefixes used in the wild
        base_dirs = [os.path.join(DATA_DIR, "valid"), os.path.join(DATA_DIR, "dev")]
        prefixes = ["dev", "valid"]
    elif split == "test":
        base_dirs = [os.path.join(DATA_DIR, "test")]
        prefixes = ["test"]
    else:
        base_dirs = [os.path.join(DATA_DIR, split)]
        prefixes = [split]

    for base in base_dirs:
        for pref in prefixes:
            flat = os.path.join(base, f"{pref}_ft_{modality}_{pid}.npy")
            nested = os.path.join(base, str(pid), f"{pref}_ft_{modality}_{pid}.npy")
            paths.append(flat)
            paths.append(nested)
    # Remove duplicates while preserving order
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths

def first_existing(path_list: List[str]) -> Optional[str]:
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

# =============================================================================
# 1b) FEATURE REDUCTION HELPERS (to avoid gigantic flattened vectors)
# =============================================================================
MAX_FEATURES_PER_MODALITY = 2048  # hard cap after summarization
ONE_D_SEGMENTS = 64               # number of bins for long 1D sequences

def _downsample_features(vec: np.ndarray, cap: int, base_name: str) -> Tuple[np.ndarray, List[str]]:
    if vec.shape[0] <= cap:
        names = [f"{base_name}_{i}" for i in range(vec.shape[0])]
        return vec.astype(np.float32, copy=False), names
    idx = np.linspace(0, vec.shape[0] - 1, cap, dtype=int)
    vec_ds = vec[idx]
    names = [f"{base_name}_{i}" for i in range(vec_ds.shape[0])]
    return vec_ds.astype(np.float32, copy=False), names

def _summarize_1d(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    arr = np.nan_to_num(arr.astype(np.float32, copy=False))
    stats_vals = [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    ]
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    try:
        q_vals = np.quantile(arr, quantiles).astype(np.float32)
    except Exception:
        q_vals = np.array([float(np.mean(arr))] * len(quantiles), dtype=np.float32)
    # Segment means for long sequences
    seg_feats: List[float] = []
    if arr.shape[0] > ONE_D_SEGMENTS:
        for seg in np.array_split(arr, ONE_D_SEGMENTS):
            seg_feats.append(float(np.mean(seg)))
    feats = np.concatenate([
        np.array(stats_vals, dtype=np.float32),
        q_vals,
        np.array(seg_feats, dtype=np.float32) if len(seg_feats) else np.array([], dtype=np.float32),
    ])
    names = [f"{modality}_mean", f"{modality}_std", f"{modality}_min", f"{modality}_max"]
    names += [f"{modality}_q{int(q*100)}" for q in quantiles]
    if len(seg_feats):
        names += [f"{modality}_segmean_{i}" for i in range(ONE_D_SEGMENTS)]
    # Cap if still too large
    feats, names = _cap_features(feats, names, MAX_FEATURES_PER_MODALITY)
    return feats, names

def _cap_features(vec: np.ndarray, names: List[str], cap: int) -> Tuple[np.ndarray, List[str]]:
    if vec.shape[0] <= cap:
        return vec, names
    idx = np.linspace(0, vec.shape[0] - 1, cap, dtype=int)
    return vec[idx], [names[i] for i in idx.tolist()]

def _summarize_2d_or_more(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    # Reshape to (time, features)
    t = arr.shape[0]
    f = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    mat = np.nan_to_num(arr.reshape(t, f).astype(np.float32, copy=False))
    # Time-aggregated stats per feature
    mean_f = np.mean(mat, axis=0)
    std_f = np.std(mat, axis=0)
    feats = np.concatenate([mean_f, std_f])
    names = (
        [f"{modality}_mean_f{i}" for i in range(mean_f.shape[0])] +
        [f"{modality}_std_f{i}" for i in range(std_f.shape[0])]
    )
    # Downsample features if exceeding cap
    if feats.shape[0] > MAX_FEATURES_PER_MODALITY:
        feats, names = _cap_features(feats, names, MAX_FEATURES_PER_MODALITY)
    return feats.astype(np.float32, copy=False), names

def summarize_modality_features(arr: np.ndarray, modality: str) -> Tuple[np.ndarray, List[str]]:
    if arr.ndim <= 1:
        return _summarize_1d(arr.reshape(-1), modality)
    return _summarize_2d_or_more(arr, modality)

def align_matrix_to_template(X: np.ndarray, current_names: List[str], template_names: List[str]) -> np.ndarray:
    """
    Reorder and pad/truncate columns of X to match template_names.
    Missing columns are filled with zeros; extra columns are dropped.
    """
    if current_names == template_names:
        return X
    name_to_idx = {n: i for i, n in enumerate(current_names)}
    num_samples = X.shape[0]
    aligned = np.zeros((num_samples, len(template_names)), dtype=X.dtype)
    for j, name in enumerate(template_names):
        i = name_to_idx.get(name)
        if i is not None and i < X.shape[1]:
            aligned[:, j] = X[:, i]
    return aligned

# =============================================================================
# 1) LOAD LABEL TABLES
# =============================================================================
def load_label_table(split: str) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      - Participant_ID
      - PHQ8_Score (for train/valid)
      - PHQ_Score (for test; we will rename to PHQ8_Score for consistency if present)
      - optional subscores & Gender if you want as extra features
    """
    if split == "train":
        df = pd.read_csv(CSV_TRAIN)
        # ensure consistent column names
        if "PHQ8_Score" not in df.columns:
            raise ValueError("PHQ8_Score not found in train CSV.")
    elif split == "valid":
        df = pd.read_csv(CSV_DEV)
        if "PHQ8_Score" not in df.columns:
            raise ValueError("PHQ8_Score not found in dev CSV.")
    elif split == "test":
        df = pd.read_csv(CSV_TEST)
        # full_test_split.csv uses PHQ_Score (no '8' in name)
        if "PHQ8_Score" not in df.columns and "PHQ_Score" in df.columns:
            df = df.rename(columns={"PHQ_Score": "PHQ8_Score", "PHQ_Binary": "PHQ8_Binary"})
    else:
        raise ValueError(f"Unknown split: {split}")
    # Normalize participant id column name
    if "Participant_ID" not in df.columns and "participant_ID" in df.columns:
        df = df.rename(columns={"participant_ID": "Participant_ID"})
    return df

# =============================================================================
# 2) FEATURE LOADING
# =============================================================================
def load_features_for_id(split: str, pid: int) -> Tuple[np.ndarray, List[str]]:
    """
    Loads & concatenates all available modality .npy files for a participant id.
    Returns (feature_vector, feature_names).
    If a modality file is missing, it is skipped.
    """
    feats = []
    names = []

    for modality in MODALITIES:
        path = first_existing(candidate_paths(split, pid, modality))
        if path is None:
            # silently skip if missing (common in AVEC variants)
            continue
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        feat_vec, feat_names_mod = summarize_modality_features(arr, modality)
        feats.append(feat_vec.astype(np.float32, copy=False))
        names.extend(feat_names_mod)

    if len(feats) == 0:
        return np.array([]), []
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False), names

def build_matrix(split: str, use_demographics: bool = True, use_subscores: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Returns X, y, ids, feature_names for a given split.
    - Uses CSV to know which participants belong to the split and their targets.
    - Optionally appends Gender and PHQ8 subscores as features (from CSV columns).
    """
    df = load_label_table(split)

    # Identify subscores columns if present
    sub_cols = [c for c in df.columns if c.startswith("PHQ8_") and c not in ("PHQ8_Score", "PHQ8_Binary")]
    demo_cols = ["Gender"] if "Gender" in df.columns else []

    X_rows = []
    y_rows = []
    ids = []
    cached_names = None

    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        fvec, fnames = load_features_for_id(split, pid)

        if fvec.size == 0:
            # skip participant if no modality is present
            continue

        extra = []
        extra_names = []

        if use_demographics and len(demo_cols) > 0:
            for c in demo_cols:
                if c in df.columns and pd.notna(row[c]):
                    extra.append(float(row[c]))
                else:
                    extra.append(0.0)
            extra_names.extend(demo_cols)

        if use_subscores and len(sub_cols) > 0:
            for c in sub_cols:
                # Many rows in full dataset have -1 for missing subscores
                val = float(row[c]) if pd.notna(row[c]) and float(row[c]) >= 0 else 0.0
                extra.append(val)
            extra_names.extend(sub_cols)

        full_vec = np.concatenate([fvec, np.array(extra, dtype=float)]) if len(extra) else fvec
        full_names = fnames + extra_names

        if cached_names is None:
            cached_names = full_names
        else:
            # Ensure consistent dimensionality across participants
            if len(full_vec) != len(cached_names):
                # Align by padding to max length (rare; but can happen if some participant has extra dims in a modality)
                max_len = max(len(cached_names), len(full_vec))
                if len(full_vec) < max_len:
                    full_vec = np.pad(full_vec, (0, max_len - len(full_vec)))
                    full_names = full_names + [f"_pad_{i}" for i in range(max_len - len(full_names))]
                if len(cached_names) < max_len:
                    # previously cached smaller — pad previous rows too
                    pad_needed = max_len - len(cached_names)
                    cached_names = cached_names + [f"_pad_{i}" for i in range(pad_needed)]
                    X_rows = [np.pad(r, (0, pad_needed)) for r in X_rows]

        X_rows.append(full_vec)
        y_rows.append(float(row["PHQ8_Score"]))
        ids.append(pid)

    if len(X_rows) == 0:
        raise RuntimeError(f"No data found for split={split}. Check file paths and naming.")

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=float)
    feature_names = cached_names if cached_names is not None else [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, ids, feature_names

# =============================================================================
# 3) PIPELINE (DEFERRED) + TRAIN / EVAL HELPERS
# =============================================================================

def evaluate_model(model: RandomForestRegressor, split_name: str, Xs: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(Xs)
    mae = mean_absolute_error(y, y_pred)
    try:
        rmse = mean_squared_error(y, y_pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = r2_score(y, y_pred)
    print(f"\n==== {split_name.upper()} METRICS ====")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}

def run_full_pipeline() -> Dict[str, Any]:
    print("Loading TRAIN...")
    X_train, y_train, ids_train, feat_names = build_matrix("train", use_demographics=True, use_subscores=True)
    print("Loading VALID...")
    X_valid, y_valid, ids_valid, feat_names_valid = build_matrix("valid", use_demographics=True, use_subscores=True)
    print("Loading TEST...")
    X_test, y_test, ids_test, feat_names_test = build_matrix("test", use_demographics=True, use_subscores=False)

    # Align to train template
    X_valid_al = align_matrix_to_template(X_valid, feat_names_valid, feat_names)
    X_test_al  = align_matrix_to_template(X_test,  feat_names_test,  feat_names)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid_al)
    X_test_s  = scaler.transform(X_test_al)

    # Train model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    m_train = evaluate_model(model, "train", X_train_s, y_train)
    m_valid = evaluate_model(model, "valid", X_valid_s, y_valid)
    m_test  = evaluate_model(model, "test",  X_test_s,  y_test)

    # Save validation plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_valid, m_valid["y_pred"], alpha=0.6)
    lim_min = min(float(np.min(y_valid)), float(np.min(m_valid["y_pred"])))
    lim_max = max(float(np.max(y_valid)), float(np.max(m_valid["y_pred"])))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
    plt.xlabel("True PHQ8 (Valid)")
    plt.ylabel("Predicted PHQ8")
    plt.title("Predicted vs Actual PHQ8 (Valid)")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/validation_plot.png")
    plt.close()

    # Save model artifact
    with open("artifacts/severity_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": feat_names}, f)

    # Predict on all
    X_all = np.vstack([X_train, X_valid_al, X_test_al])
    X_all_s = scaler.transform(X_all)
    pred_all = model.predict(X_all_s)

    # Explainability
    explainer_shap = shap.TreeExplainer(model)
    explainer_lime = LimeTabularExplainer(
        X_train_s,
        feature_names=feat_names,
        class_names=["PHQ8_Score"],
        verbose=False,
        mode="regression"
    )

    return {
        "X_train": X_train,
        "X_valid": X_valid_al,
        "X_test": X_test_al,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_valid": ids_valid,
        "ids_test": ids_test,
        "feat_names": feat_names,
        "scaler": scaler,
        "model": model,
        "X_train_s": X_train_s,
        "X_valid_s": X_valid_s,
        "X_test_s": X_test_s,
        "X_all_s": X_all_s,
        "pred_all": pred_all,
        "TOTAL_N": len(ids_train) + len(ids_valid) + len(ids_test),
        "m_train": m_train,
        "m_valid": m_valid,
        "m_test": m_test,
        "explainer_shap": explainer_shap,
        "explainer_lime": explainer_lime,
    }

# =============================================================================
# 4) MISINFORMATION SIMULATION
# =============================================================================
def simulate_misinformation(num_nodes, init_infected_frac=0.1, trans_prob=0.2, rec_prob=0.1, steps=20):
    G = nx.barabasi_albert_graph(num_nodes, m=2, seed=42)
    for n in G.nodes():
        G.nodes[n]['state'] = 'S'
    infected = random.sample(list(G.nodes()), max(1, int(init_infected_frac * num_nodes)))
    for n in infected:
        G.nodes[n]['state'] = 'I'
    S_list, I_list, R_list = [], [], []
    for _ in range(steps):
        new_states = {}
        for n in G.nodes():
            state = G.nodes[n]['state']
            if state == 'S':
                for nbr in G.neighbors(n):
                    if G.nodes[nbr]['state'] == 'I' and random.random() < trans_prob:
                        new_states[n] = 'I'
                        break
            elif state == 'I':
                if random.random() < rec_prob:
                    new_states[n] = 'R'
        for n, s in new_states.items():
            G.nodes[n]['state'] = s
        states = [G.nodes[n]['state'] for n in G.nodes()]
        S_list.append(states.count('S'))
        I_list.append(states.count('I'))
        R_list.append(states.count('R'))
    return S_list, I_list, R_list, G

def allocate_resources(severity_scores, capacity=10):
    idx = np.argsort(severity_scores)[::-1]
    return idx[:capacity], idx[capacity:]

# Explainability handled within run_full_pipeline and UI only

# =============================================================================
# 6) STREAMLIT APP
# =============================================================================
def run_app():
    st.title("PHQ8 Prediction (Multimodal) + Resource Prioritisation")

    # Sidebar
    st.sidebar.header("Misinformation Simulation")
    trans_prob = st.sidebar.slider("Transmission Probability", 0.0, 1.0, 0.2, 0.01)
    rec_prob   = st.sidebar.slider("Recovery Probability", 0.0, 1.0, 0.1, 0.01)
    steps      = st.sidebar.slider("Steps", 5, 100, 20, 1)
    capacity   = st.sidebar.number_input("Treatment Capacity", min_value=1, max_value=200, value=10)

    # Run Simulation button
    if st.button("Run Simulation"):
        with st.spinner("Loading data and training model..."):
            arts = run_full_pipeline()
            st.session_state["arts"] = arts
            st.session_state["show_validation_plot"] = True

    if "arts" not in st.session_state:
        st.info("Click 'Run Simulation' to load the dataset, train the model, and view predictions.")
        return

    arts = st.session_state["arts"]

    # Dataset Summary
    st.subheader("Dataset Summary")
    st.write(
        f"Train: {len(arts['ids_train'])} | Valid: {len(arts['ids_valid'])} | Test: {len(arts['ids_test'])} | Features: {len(arts['feat_names'])}"
    )

    # Re-run simulation with user params
    S_list_, I_list_, R_list_, G_net_ = simulate_misinformation(
        num_nodes=arts["TOTAL_N"],
        trans_prob=trans_prob,
        rec_prob=rec_prob,
        steps=steps,
    )
    misinfo_risk_ = I_list_[-1] / arts["TOTAL_N"]

    # Recompute adjusted severity
    adjusted_all_ = arts["pred_all"] * (1 - misinfo_risk_)
    treated, untreated = allocate_resources(adjusted_all_, capacity=capacity)

    st.metric("Overall Misinformation Risk", f"{misinfo_risk_:.3f}")
    st.metric("Treatment Capacity", capacity)

    # Table (show only first 300 rows for UI snappiness)
    df_all = pd.DataFrame({
        "GlobalIndex": list(range(len(adjusted_all_))),
        "RawPredictedPHQ8": np.round(arts["pred_all"], 3),
        "AdjustedPHQ8": np.round(adjusted_all_, 3),
        "Prioritized": ["Yes" if i in treated else "No" for i in range(len(adjusted_all_))]
    })
    st.dataframe(df_all.head(300))

    # Explain a specific example (choose from train/valid/test)
    st.subheader("Patient-level Explanation")
    which_split = st.selectbox("Split", ["train", "valid", "test"])
    if which_split == "train":
        X_s, ids_s, y_s = arts["X_train_s"], arts["ids_train"], arts["y_train"]
    elif which_split == "valid":
        X_s, ids_s, y_s = arts["X_valid_s"], arts["ids_valid"], arts["y_valid"]
    else:
        X_s, ids_s, y_s = arts["X_test_s"], arts["ids_test"], arts["y_test"]

    idx_local = st.number_input(f"Row Index in {which_split}", 0, len(ids_s)-1, 0)
    pred_local = arts["model"].predict(X_s[[idx_local]])[0]
    st.write(f"**Predicted PHQ8:** {pred_local:.3f}")
    st.write(f"**True PHQ8:** {float(y_s[idx_local]):.3f}")

    method = st.radio("Explanation Method", ["SHAP", "LIME"], horizontal=True)
    if method == "SHAP":
        st.subheader("SHAP (local)")
        shap_vals_local = arts["explainer_shap"].shap_values(X_s[[idx_local]])
        shap.force_plot(
            arts["explainer_shap"].expected_value,
            shap_vals_local[0],
            features=X_s[[idx_local]],
            matplotlib=True, show=False
        )
        fig_local = plt.gcf()
        st.pyplot(fig_local, bbox_inches="tight")
        plt.close(fig_local)
    else:
        st.subheader("LIME (local)")
        lime_exp = arts["explainer_lime"].explain_instance(
            X_s[idx_local],
            arts["model"].predict,
            num_features=min(15, len(arts["feat_names"]))
        )
        fig = lime_exp.as_pyplot_figure()
        st.pyplot(fig)

    st.subheader("Risk Heatmap")
    fig, ax = plt.subplots()
    scatter = ax.scatter(range(len(adjusted_all_)), np.zeros(len(adjusted_all_)),
                         c=adjusted_all_, cmap="Reds", s=60)
    plt.colorbar(scatter, label="Adjusted PHQ8")
    ax.set_yticks([])
    ax.set_xlabel("Global Index")
    ax.set_title("Adjusted Risk Heatmap")
    st.pyplot(fig)

    st.subheader("Misinformation Spread Over Time")
    fig_misinfo, ax_misinfo = plt.subplots()
    ax_misinfo.plot(S_list_, label="Susceptible")
    ax_misinfo.plot(I_list_, label="Infected")
    ax_misinfo.plot(R_list_, label="Recovered")
    ax_misinfo.legend()
    ax_misinfo.set_xlabel("Step")
    ax_misinfo.set_ylabel("Nodes")
    st.pyplot(fig_misinfo)
    plt.close(fig_misinfo)

    st.subheader("Network Snapshot")
    fig_net, ax_net = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(G_net_, seed=42)
    c_map = {'S': 'blue', 'I': 'red', 'R': 'green'}
    node_colors = [c_map[G_net_.nodes[n]['state']] for n in G_net_.nodes()]
    nx.draw(G_net_, pos, node_color=node_colors, node_size=20, with_labels=False, ax=ax_net)
    st.pyplot(fig_net)
    plt.close(fig_net)

# Streamlit validation image
st_subtitle_shown = False
def show_validation_image():
    # Only show if a run has occurred in this session
    if st is None or not st.session_state.get("show_validation_plot", False):
        return
    global st_subtitle_shown
    if not st_subtitle_shown:
        st.subheader("Validation: Predicted vs Actual PHQ8 (Valid Set)")
        st_subtitle_shown = True
    img_path = "artifacts/validation_plot.png"
    if os.path.exists(img_path):
        st.image(img_path, caption="Predicted vs Actual (Valid)", use_container_width=True)
    else:
        st.info("Validation plot not found (should be at artifacts/validation_plot.png).")

if __name__ == "__main__":
    # Under Streamlit, never run CLI; always launch app
    if st is not None and (os.environ.get("STREAMLIT_SERVER_PORT") or "streamlit" in os.path.basename(sys.argv[0]).lower()):
        run_app()
        show_validation_image()
    else:
        parser = argparse.ArgumentParser(description="PHQ8 prediction pipeline (CLI or Streamlit)")
        parser.add_argument("--mode", choices=["cli", "app"], default="app", help="Run in CLI mode or Streamlit app mode")
        parser.add_argument("--trans-prob", type=float, default=0.2, dest="trans_prob", help="Transmission probability for misinformation simulation")
        parser.add_argument("--rec-prob", type=float, default=0.1, dest="rec_prob", help="Recovery probability for misinformation simulation")
        parser.add_argument("--steps", type=int, default=20, help="Number of simulation steps")
        parser.add_argument("--capacity", type=int, default=10, help="Treatment capacity for allocation")
        parser.add_argument("--explain-split", choices=["train", "valid", "test"], default=None, help="Which split to explain a sample from")
        parser.add_argument("--explain-index", type=int, default=None, help="Row index within the chosen split to explain")
        parser.add_argument("--explain-method", choices=["SHAP", "LIME"], default="SHAP", help="Explanation method for local explanation image")
        parser.add_argument("--output-dir", default="artifacts", help="Directory to save outputs (plots, models)")
        args = parser.parse_args()

        os.makedirs(args.output_dir, exist_ok=True)

        if args.mode == "app":
            if st is None:
                print("Streamlit is not installed or unavailable in this environment. Install it or run with --mode cli.")
                sys.exit(1)
            run_app()
            show_validation_image()
            sys.exit(0)

        # CLI mode (explicit only)
        print("\nRunning in CLI mode...\n")
        arts = run_full_pipeline()
        TOTAL_N = arts["TOTAL_N"]

        S_list_, I_list_, R_list_, G_net_ = simulate_misinformation(
            num_nodes=TOTAL_N,
            trans_prob=args.trans_prob,
            rec_prob=args.rec_prob,
            steps=args.steps,
        )
        misinfo_risk_ = I_list_[-1] / TOTAL_N
        adjusted_all_ = arts["pred_all"] * (1 - misinfo_risk_)
        treated, untreated = allocate_resources(adjusted_all_, capacity=args.capacity)

        print(f"Misinformation risk: {misinfo_risk_:.3f}")
        print(f"Treatment capacity: {args.capacity}")
        print(f"Top {min(len(treated), args.capacity)} prioritized indices (global): {treated.tolist()}")

        # Save updated risk heatmap
        fig, ax = plt.subplots()
        scatter = ax.scatter(range(len(adjusted_all_)), np.zeros(len(adjusted_all_)), c=adjusted_all_, cmap="Reds", s=40)
        plt.colorbar(scatter, label="Adjusted PHQ8")
        ax.set_yticks([])
        ax.set_xlabel("Global Index")
        ax.set_title("Adjusted Risk Heatmap (CLI)")
        plt.tight_layout()
        heatmap_path = os.path.join(args.output_dir, "risk_heatmap_cli.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved heatmap: {heatmap_path}")

        # Optional local explanation
        if args.explain_split is not None and args.explain_index is not None:
            if args.explain_split == "train":
                X_s, ids_s, y_s = arts["X_train_s"], arts["ids_train"], arts["y_train"]
            elif args.explain_split == "valid":
                X_s, ids_s, y_s = arts["X_valid_s"], arts["ids_valid"], arts["y_valid"]
            else:
                X_s, ids_s, y_s = arts["X_test_s"], arts["ids_test"], arts["y_test"]

            if not (0 <= args.explain_index < len(ids_s)):
                print(f"Invalid --explain-index for split {args.explain_split}; range is [0, {len(ids_s)-1}]")
                sys.exit(2)

            pred_local = arts["model"].predict(X_s[[args.explain_index]])[0]
            print(f"Explaining sample idx={args.explain_index} from {args.explain_split}: Predicted={pred_local:.3f}, True={float(y_s[args.explain_index]):.3f}")

            if args.explain_method.upper() == "SHAP":
                shap_vals_local = arts["explainer_shap"].shap_values(X_s[[args.explain_index]])
                shap.force_plot(
                    arts["explainer_shap"].expected_value,
                    shap_vals_local[0],
                    features=X_s[[args.explain_index]],
                    matplotlib=True,
                    show=False,
                )
                out_path = os.path.join(args.output_dir, f"explain_{args.explain_split}_{args.explain_index}_shap.png")
                plt.savefig(out_path, bbox_inches="tight")
                plt.close()
                print(f"Saved SHAP explanation: {out_path}")
            else:
                lime_exp = arts["explainer_lime"].explain_instance(
                    X_s[args.explain_index],
                    arts["model"].predict,
                    num_features=min(15, len(arts["feat_names"])),
                )
                fig = lime_exp.as_pyplot_figure()
                out_path = os.path.join(args.output_dir, f"explain_{args.explain_split}_{args.explain_index}_lime.png")
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved LIME explanation: {out_path}")

