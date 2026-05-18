from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from two_stage_model import Y_COLS, build_group_keys

st.set_page_config(page_title="Material Intensity Predictor", layout="wide")

ARTIFACT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    return preprocessor, model


TYPOLOGY_MAP = {
    "R-SFH": "Single-Family House",
    "R-MFH": "Multi-Family House",
    "R-AB": "Apartment Block",
    "R-UNK": "Residential (Unknown)",
    "NR-OH": "Office (High)",
    "NR-OL": "Office (Low)",
    "NR-C": "Commercial (Retail/Mall)",
    "NR-E": "Education",
    "NR-I": "Industry",
    "NR-P": "Public/Civic",
    "NR-H": "Hotel/Hospital",
    "NR-UNK": "Non-residential (Unknown)",
}

PRIMARY_CODE_MAP = {
    "B": "Brick",
    "BC": "Brick-Concrete",
    "BW": "Brick-Wood",
    "W": "Wood",
    "C": "Concrete",
    "CW": "Concrete-Wood",
    "S": "Steel",
    "SC": "Steel-Concrete",
    "T": "Traditional material",
}

HYBRID_STRUCTURE_MAP = {
    0: "Single-Material Structure",
    1: "Mixed-Material Structure",
}

st.title("Material Intensity Predictor")
st.write("Estimate material intensity percentiles (5th, 50th, and 95th) for a building.")

try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error(f"Error loading artifacts: {exc}")
    st.stop()

with st.sidebar:
    st.header("Building Inputs")

    construction_period = st.number_input(
        "Construction period", min_value=1900, max_value=2100, value=2015
    )

    typology = st.selectbox(
        "Typology",
        options=list(TYPOLOGY_MAP.keys()),
        format_func=lambda x: TYPOLOGY_MAP.get(x, x),
    )

    primary_code = st.selectbox(
        "Primary Code",
        options=list(PRIMARY_CODE_MAP.keys()),
        format_func=lambda x: PRIMARY_CODE_MAP.get(x, x),
    )

    hybrid_structure = st.selectbox(
        "Hybrid Structure",
        options=list(HYBRID_STRUCTURE_MAP.keys()),
        format_func=lambda x: HYBRID_STRUCTURE_MAP.get(x, x),
    )

    country_options = preprocessor.named_transformers_["cat"].categories_[3]
    country = st.selectbox("Country", options=country_options)

if st.button("Predict Material Intensity", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "Construction period": construction_period,
                "Typology": typology,
                "Primary Code": primary_code,
                "Hybrid Structure": hybrid_structure,
                "Country": country,
            }
        ]
    )

    x_proc = preprocessor.transform(input_df)
    groups = build_group_keys(input_df)
    predictions = model.predict(x_proc, groups, alpha=0.10)

    st.subheader("Predicted Material Intensities (kg/m2)")
    cols = st.columns(len(Y_COLS))

    for col, material in zip(cols, Y_COLS):
        with col:
            p = predictions[material]
            st.markdown(f"### {material}")
            st.metric("5th percentile", f"{float(p['p5'][0]):.2f}")
            st.metric("Median", f"{float(p['p50'][0]):.2f}")
            st.metric("95th percentile", f"{float(p['p95'][0]):.2f}")
            st.metric("Presence probability", f"{float(p['p_presence'][0]):.2f}")
else:
    st.caption("Set inputs in sidebar and click Predict.")
