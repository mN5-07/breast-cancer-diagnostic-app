"""
Breast Cancer Predictor App
A Streamlit demo using Logistic Regression on the Wisconsin Breast Cancer dataset.
Author: Matthew
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import time

# ────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Disclaimer (always front and center)
# ────────────────────────────────────────────────
st.warning(
    "⚠️ **Educational Tool Only** – This is NOT a medical diagnostic tool. "
    "Predictions are for learning purposes. Always consult a qualified "
    "healthcare professional for medical advice."
)

# ────────────────────────────────────────────────
# Paths & Caching
# ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource(show_spinner="Loading model & scaler...")
def load_model_and_scaler():
    model_path  = MODEL_DIR / "best_model_Logistic_Regression.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"

    missing = []
    if not model_path.is_file():
        missing.append(model_path.name)
    if not scaler_path.is_file():
        missing.append(scaler_path.name)

    if missing:
        st.error(
            f"Missing model file(s) in folder **{MODEL_DIR.relative_to(BASE_DIR.parent)}/**:\n"
            f"• {' • '.join(missing)}\n\n"
            "Make sure these files exist and are committed to git (if deploying)."
        )
        st.info(f"Expected full path to model: {model_path}")
        st.stop()

    try:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model or scaler:\n{e.__class__.__name__}: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# ────────────────────────────────────────────────
# Feature Definitions
# ────────────────────────────────────────────────
FEATURE_GROUPS = {
    "Mean": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
    ],
    "Standard Error": [
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se"
    ],
    "Worst": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
}

FEATURES = FEATURE_GROUPS["Mean"] + FEATURE_GROUPS["Standard Error"] + FEATURE_GROUPS["Worst"]

RANGES = {f: (lo * 0.95, hi * 1.05) for f, (lo, hi) in zip(FEATURES, [
    (6.981, 28.110), (9.710, 39.280), (43.790, 188.500), (143.500, 2501.000), (0.053, 0.163),
    (0.019, 0.345), (0.000, 0.427), (0.000, 0.201), (0.106, 0.304), (0.050, 0.097),
    (0.112, 2.873), (0.360, 4.885), (0.757, 21.980), (6.802, 542.200), (0.002, 0.031),
    (0.002, 0.135), (0.000, 0.396), (0.000, 0.053), (0.008, 0.079), (0.001, 0.030),
    (7.930, 36.040), (12.020, 49.540), (50.410, 251.200), (185.200, 4254.000), (0.071, 0.223),
    (0.027, 0.600), (0.000, 1.252), (0.000, 0.291), (0.157, 0.664), (0.055, 0.207)
])}

DEFAULT_VALUES = dict(zip(FEATURES, [
    13.370, 18.840, 86.240, 551.100, 0.096, 0.093, 0.062, 0.034, 0.179, 0.062,
    0.324, 1.108, 2.287, 24.530, 0.006, 0.020, 0.026, 0.011, 0.019, 0.003,
    14.970, 25.410, 97.660, 686.500, 0.131, 0.180, 0.227, 0.100, 0.282, 0.080
]))

IMPORTANT_FEATURES = [
    "concavity_worst", "texture_worst", "symmetry_worst",
    "concave points_worst", "area_worst"
]

FEATURE_HELP = {
    "radius": "Mean distance from center to perimeter points",
    "texture": "Standard deviation of gray-scale values",
    "perimeter": "Perimeter of the cell nucleus",
    "area": "Area of the cell nucleus",
    "smoothness": "Local variation in radius lengths",
    "compactness": "Perimeter² / area - 1.0",
    "concavity": "Severity of concave portions of the contour",
    "concave points": "Number of concave portions of the contour",
    "symmetry": "Symmetry of the cell nucleus",
    "fractal_dimension": "Coastline approximation - 1"
}

# ────────────────────────────────────────────────
# UI – Main Content
# ────────────────────────────────────────────────
st.title("🎗️ Breast Cancer Predictor")

st.markdown("""
This educational app uses a **Logistic Regression** model trained on the  
Wisconsin Breast Cancer dataset to estimate benign vs. malignant tumors.
""")

st.caption("Model accuracy: ~97–98% | Dataset: 569 samples | For learning only")

# ────────────────────────────────────────────────
# Quick Start Presets
# ────────────────────────────────────────────────
st.subheader("🚀 Quick Start Presets")
col1, col2, col3, col4 = st.columns(4)

if col1.button("📊 Median Values", use_container_width=True):
    st.session_state["preset"] = "median"
if col2.button("✅ Typical Benign", use_container_width=True):
    st.session_state["preset"] = "benign"
if col3.button("⚠️ Borderline Case", use_container_width=True):
    st.session_state["preset"] = "borderline"
if col4.button("🔴 Typical Malignant", use_container_width=True):
    st.session_state["preset"] = "malignant"

# Apply preset
if "preset" in st.session_state:
    p = st.session_state["preset"]
    if p == "median":
        values = list(DEFAULT_VALUES.values())
    elif p == "benign":
        values = [v * 0.75 for v in DEFAULT_VALUES.values()]   # placeholder
    elif p == "borderline":
        values = [v * 1.15 for v in DEFAULT_VALUES.values()]
    elif p == "malignant":
        values = [v * 1.4 for v in DEFAULT_VALUES.values()]    # placeholder

    for feat, val in zip(FEATURES, values):
        st.session_state[feat] = val

    del st.session_state["preset"]
    st.rerun()

# ────────────────────────────────────────────────
# Sidebar – Input Controls
# ────────────────────────────────────────────────
st.sidebar.header("📋 Input Features")
st.sidebar.markdown("Adjust values below. Features grouped by type.")

if st.sidebar.button("🔄 Reset to Median", use_container_width=True):
    for feat in FEATURES:
        if feat in st.session_state:
            del st.session_state[feat]
    st.rerun()

st.sidebar.markdown("---")

tab1, tab2, tab3 = st.sidebar.tabs(["📊 Mean", "📈 Std Err", "⚡ Worst"])

def create_slider(feature: str, container):
    minv, maxv = RANGES[feature]
    default = DEFAULT_VALUES[feature]
    base_name = feature.rsplit("_", 1)[0]
    help_text = FEATURE_HELP.get(base_name, "")
    label = feature.replace("_", " ").title().replace("Se ", "SE ").replace("Mean ", "").replace("Worst ", "")

    if maxv < 1:
        step, fmt = 0.001, "%.3f"
    else:
        step, fmt = (0.1 if maxv > 100 else 0.01), ("%.1f" if maxv > 100 else "%.3f")

    return container.slider(
        label=label,
        min_value=float(minv),
        max_value=float(maxv),
        value=float(st.session_state.get(feature, default)),
        step=step,
        format=fmt,
        key=feature,
        help=help_text
    )

inputs = []
with tab1:
    st.markdown("**Average** cell nucleus measurements")
    for f in FEATURE_GROUPS["Mean"]:
        inputs.append(create_slider(f, tab1))

with tab2:
    st.markdown("**Variability** (standard error)")
    for f in FEATURE_GROUPS["Standard Error"]:
        inputs.append(create_slider(f, tab2))

with tab3:
    st.markdown("**Worst / largest** values  ⭐ *most predictive*")
    for f in FEATURE_GROUPS["Worst"]:
        inputs.append(create_slider(f, tab3))

# ────────────────────────────────────────────────
# Live Prediction (auto-updates on slider change)
# ────────────────────────────────────────────────
st.subheader("🔬 Live Prediction")

# Simple debounce: avoid predicting 20× while dragging
if "last_predict_time" not in st.session_state:
    st.session_state.last_predict_time = 0

now = time.time()
if inputs and (now - st.session_state.last_predict_time > 0.35):  # ~350ms debounce
    with st.spinner("Analyzing..."):
        X_raw = np.array([inputs])
        X_scaled = scaler.transform(X_raw)

        pred = model.predict(X_scaled)[0]
        probas = model.predict_proba(X_scaled)[0]
        prob_mal = probas[1] * 100  # to %

    st.session_state.last_predict_time = now

    # ── Result Display ───────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1.2, 1.2])

    with col1:
        if pred == 0:
            st.success("### ✅ **Benign** (non-cancerous)")
        else:
            st.error("### 🔴 **Malignant** (cancerous)")

    with col2:
        st.metric("Confidence", f"{max(probas):.1%}")
    with col3:
        st.metric("Malignancy Risk", f"{prob_mal:.1f}%")

    # Progress bar with dynamic color text
    color_emoji = "🟢" if prob_mal < 30 else "🟡" if prob_mal < 70 else "🔴"
    st.progress(int(prob_mal), text=f"{color_emoji} Malignancy probability: **{prob_mal:.1f}%**")

    # ── Key Features Highlight ───────────────────────────────
    st.markdown("---")
    st.subheader("📊 Most Influential Features")
    cols = st.columns(5)
    for i, feat in enumerate(IMPORTANT_FEATURES):
        idx = FEATURES.index(feat)
        val = inputs[idx]
        name = feat.replace("_", " ").title()
        cols[i].metric(name, f"{val:.3f}")

    # Full inputs table
    with st.expander("📋 All Input Values"):
        df = pd.DataFrame([inputs], columns=FEATURES).T.rename(columns={0: "Value"})
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

    # Extreme value warning
    extremes = [f for f, v in zip(FEATURES, inputs) if v < RANGES[f][0] * 1.05 or v > RANGES[f][1] * 0.95]
    if extremes:
        st.info(f"Note: Some values ({', '.join(extremes[:3])}...) are near/outside typical dataset range. "
                "Prediction reliability may be lower.")

# Placeholder before first interaction
if not inputs or all(v == d for v, d in zip(inputs, DEFAULT_VALUES.values())):
    st.info("Adjust any slider to see a live prediction.")

# ────────────────────────────────────────────────
# About Section
# ────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About the Model & Dataset"):
    st.markdown("""
    **Model**  
    • Logistic Regression (chosen for interpretability + performance)  
    • ~96–97% test accuracy | 5-fold CV ~97.1%  

    **Dataset**  
    • UCI Breast Cancer Wisconsin (Diagnostic)  
    • 569 samples (357 benign, 212 malignant)  
    • 30 features from digitized cell nuclei images  

    **Top Features** (permutation importance)  
    1. concavity_worst  
    2. texture_worst  
    3. symmetry_worst  
    4. concave points_worst  
    5. area_worst  

    **Disclaimer**  
    Educational demo only. Not validated for clinical use.  
    Real diagnosis requires imaging, biopsy, and expert evaluation.
    """)

st.caption("Built with ❤️ by Matthew | Powered by scikit-learn & Streamlit | Updated January 2026")