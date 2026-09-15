"""
app/streamlit_demo.py
──────────────────────
Lightweight research prototype demo for the water quality ML project.

⚠️ DISCLAIMER: This is a research prototype trained on a public dataset
of uncertain provenance. It is NOT a certified water safety system and
must not be used to make decisions about drinking water safety.

Usage:
    streamlit run app/streamlit_demo.py

Requirements:
    pip install streamlit
    # Models must be trained first: python scripts/train_models.py
    # SHAP must be run first: python scripts/run_explainability.py
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_config
from src.preprocessing.pipeline import WaterQualityPreprocessor

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Assessment — Research Demo",
    page_icon="💧",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    cfg = load_config("configs/config.yaml")
    models_dir = Path(cfg["results"]["models"])
    shap_dir = Path(cfg["results"]["shap"])

    resources = {"cfg": cfg, "models": {}, "ranked_features": None}

    # Load models
    for model_path in sorted(models_dir.glob("*.pkl")):
        name = model_path.stem
        if name == "preprocessor":
            continue
        try:
            with open(model_path, "rb") as f:
                resources["models"][name] = pickle.load(f)
        except Exception:
            pass

    # Load preprocessor
    preprocessor_path = models_dir / "preprocessor.pkl"
    if preprocessor_path.exists():
        with open(preprocessor_path, "rb") as f:
            resources["preprocessor"] = pickle.load(f)
    else:
        resources["preprocessor"] = None

    # Load SHAP rankings
    ranking_path = shap_dir / "ranked_features_RandomForest.json"
    if ranking_path.exists():
        with open(ranking_path) as f:
            resources["ranked_features"] = json.load(f)

    return resources


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("💧 Water Quality Assessment — Research Demo")
st.caption("Explainable and Robust Machine Learning for Water Quality Assessment")

# Disclaimer
st.error(
    "⚠️ **Research Prototype — NOT for real-world safety decisions.** "
    "This tool is trained on a public dataset of uncertain provenance. "
    "It is not a certified water testing system and results must not be used "
    "to determine whether water is safe to consume."
)

# Load resources
try:
    res = load_resources()
    if not res["models"]:
        st.warning(
            "No trained models found. Please run `python scripts/train_models.py` first."
        )
        st.stop()
except FileNotFoundError as e:
    st.warning(f"Configuration or model files not found: {e}")
    st.stop()

cfg = res["cfg"]
feature_names = [
    "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# ── Sidebar: model selection ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    available_models = [m for m in res["models"] if m != "Dummy"]
    selected_model = st.selectbox("Select Model", available_models, index=0)

    st.markdown("---")
    st.markdown("**About this demo**")
    st.markdown(
        "Enter physicochemical measurements to get a potability prediction "
        "from the selected ML model, along with a feature importance explanation."
    )
    st.markdown("**Dataset:** Kaggle Water Potability (CC0)")
    st.markdown("**License:** MIT")

# ── Main: parameter input ────────────────────────────────────────────────────
st.subheader("Enter Physicochemical Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                          help="Acidity/alkalinity (WHO range: 6.5–8.5)")
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
    solids = st.number_input("Solids / TDS (ppm)", min_value=0.0, max_value=60000.0, value=20000.0, step=100.0)

with col2:
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=333.0, step=1.0)
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=800.0, value=421.0, step=1.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=14.0, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=130.0, value=66.0, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.99, step=0.01,
                                  help="WHO target: < 1 NTU for effective disinfection")

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("🔍 Predict Potability", type="primary", use_container_width=True):
    input_values = [ph, hardness, solids, chloramines, sulfate,
                    conductivity, organic_carbon, trihalomethanes, turbidity]
    X_input = pd.DataFrame([input_values], columns=feature_names)

    # Preprocess
    if res["preprocessor"] is not None:
        X_proc = res["preprocessor"].transform(X_input)
    else:
        X_proc = X_input

    model = res["models"][selected_model]

    # Predict
    prediction = model.predict(X_proc)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_proc)[0]
        prob_potable = probabilities[1]
        prob_not_potable = probabilities[0]
    else:
        prob_potable = float(prediction)
        prob_not_potable = 1.0 - float(prediction)

    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if prediction == 1:
            st.success(f"🟢 **Predicted: POTABLE**")
            st.metric("Confidence (Potable)", f"{prob_potable:.1%}")
        else:
            st.error(f"🔴 **Predicted: NOT POTABLE**")
            st.metric("Confidence (Not Potable)", f"{prob_not_potable:.1%}")

    with col_r2:
        st.metric("Model", selected_model)
        st.metric("Probability (Potable)", f"{prob_potable:.3f}")
        st.metric("Probability (Not Potable)", f"{prob_not_potable:.3f}")

    # SHAP feature importance (from global rankings)
    st.markdown("---")
    st.subheader("Feature Contributions (Model Context)")

    if res["ranked_features"] is not None:
        ranked = res["ranked_features"]
        st.markdown(
            "The table below shows the global SHAP feature importance ranking "
            f"(from Random Forest, computed on the test set). "
            "This ranking reflects the model's general learned associations, "
            "not a specific explanation for this individual prediction."
        )
        importance_data = pd.DataFrame({
            "Rank": range(1, len(ranked) + 1),
            "Feature": ranked,
            "Your Input": [X_input[f].values[0] if f in X_input.columns else "N/A" for f in ranked]
        })
        st.table(importance_data)
    else:
        st.info("Run `python scripts/run_explainability.py` to generate SHAP rankings.")

    # Warning banner
    st.markdown("---")
    st.warning(
        "**⚠️ Interpretation Warning:** This prediction is produced by a machine learning "
        "model trained on the Kaggle Water Potability dataset (Kadiwal, 2021), which has "
        "uncertain data provenance. The model's output is a statistical association, "
        "not a scientific determination of water safety. Feature importances reflect "
        "model-learned associations and do not imply causation. "
        "Always use certified laboratory testing for actual water safety assessment."
    )

# ── Research context ──────────────────────────────────────────────────────────
with st.expander("📚 Research Context"):
    st.markdown("""
    **Project:** Explainable and Robust Machine Learning for Water Quality Assessment

    **Research Questions:**
    - RQ1: How accurately can ML models classify water potability?
    - RQ2: Which parameters most influence predictions?
    - RQ3: How does reducing the number of measured parameters affect performance?
    - RQ4: How robust are models to measurement noise?

    **Dataset:** Kadiwal, A. (2021). Water Quality and Potability. Kaggle. CC0 Public Domain.

    **Experimental results** are available in `results/tables/` after running the full pipeline.

    **Citation:** See `paper/references.bib` for all references.
    """)
