# Material Intensity Predictor Web App

This web app serves predictions from the current **ThreeStageConditionalModel** pipeline. The App can be accessed here: https://predictmi.streamlit.app/

Model internals:
- Stage 1: classifier-chain `XGBClassifier` per material with Platt calibration (`CalibratedClassifierCV`) → `p_presence`
- Stage 2: per-material `XGBRegressor` quantile regression (`reg:quantileerror`) in log-space → `p5`, `p50`, `p95`
- Joint layer: group-specific multivariate normal on log-residuals (Primary Code groups) — used for residual inspection, not for `predict()` intervals

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Model Used

The app loads:
- `preprocessor.joblib`
- `model.joblib`

Model classes are defined in `two_stage_model.py` and must be present for `joblib.load` to work correctly.

Predictions include, for each material:
- `p5` — 5th percentile (kg/m²)
- `p50` — median / point estimate (kg/m²)
- `p95` — 95th percentile (kg/m²)
- `p_presence` — probability the material appears in the building

Prediction intervals (`p5`, `p95`) are produced by Stage 2 quantile regressors. The joint layer (Primary Code-grouped multivariate normal) is retained in the model object for residual inspection and sampling realism checks, but does not drive app output.

## Input Fields

| Field | Type |
|---|---|
| Construction period | Numeric (year) |
| Typology | Categorical |
| Primary Code | Categorical |
| Hybrid Structure | Categorical |
| Country | Categorical |

## Notes

- The app does not use the legacy PyTorch checkpoint pipeline.
- If artifacts are retrained in the notebook or script, replace `preprocessor.joblib` and `model.joblib` in the project root before restarting the app.

## Troubleshooting

- If port `8501` is occupied: `streamlit run Material_Intensity_Predictor.py --server.port 8502`
- Confirm `preprocessor.joblib`, `model.joblib`, and `two_stage_model.py` all exist in the project root.
