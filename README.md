# Material Intensity Predictor

This repository implements a **TwoStageConditionalModel** for building material-intensity prediction in kg/m². The current pipeline combines calibrated material-occurrence prediction, conditional quantile regression for positive intensities, and a diagnostic joint residual layer used for correlation inspection and sampling realism checks.

The current notebook and script implement three main components:

1. **Stage 1: Material occurrence modeling**  
  A classifier-chain of per-material `XGBClassifier` models predicts material presence probabilities (`p_presence`). Each classifier is wrapped with `CalibratedClassifierCV(method="sigmoid")`, and the sampling workflow applies additional post-hoc isotonic calibration on validation data.

2. **Stage 2: Conditional intensity modeling**  
  For each material, an `XGBRegressor` with `objective="reg:quantileerror"` is trained in log-space at quantiles `[0.05, 0.50, 0.95]`. The model returns `p5`, `p50`, and `p95` in the original kg/m² scale for rows where the material is present.

3. **Stage 3: Joint residual layer (diagnostic and sampling use)**  
  `JointDistributionModel` estimates group-wise residual covariance by **Primary Code**. It is retained for residual diagnostics and for damped covariance perturbations in `sample_query`, but the default `predict()` intervals come from Stage 2 quantile regression rather than from a multivariate normal interval construction.

## Current Artifacts

Required runtime artifacts:
- `preprocessor.joblib`
- `model.joblib`
- `model_info.json`

Legacy artifacts from the previous PyTorch quantile model are obsolete and should not be used.

## Main Files

- `Material_Intensity_Predictor.py` — Streamlit predictor app using `model.joblib`.
- `prediction_model.ipynb` — end-to-end notebook (training, tuning, validation, export).
- `prediction_model.py` — script version of the current notebook workflow, including diagnostics, tuning, validation, and artifact export.
- `two_stage_model.py` — importable module defining the persisted model classes used by `joblib.load`.

## Integrated MI Database Sources

The `Integrated_MI_database_add_Singapore.xlsx` file is harmonized from five source databases. Source labels are stored as R-n, N-n, B-n, G-n, and C-n, where n is the record index from each source.

- **R-n**: Global construction materials database and stock analysis of residential buildings between 1970–2050  
  Link: https://doi.org/10.1016/j.jclepro.2019.119146
- **N-n**: Spatiotemporal Characteristics of Global Building Material Intensity Revealed for Circular and Low-Carbon Construction  
  Link: https://doi.org/10.1021/acs.est.5c05684
- **B-n**: A database seed for a community-driven material intensity research platform  
  Link: https://doi.org/10.1038/s41597-019-0021-x
- **G-n**: Global Buildings Database Seed on Whole Life Carbon Emissions, Energy Performance, and Material Intensity (GBDB CarbEnMats)  
  Link: https://doi.org/10.21203/rs.3.rs-3373442/v1
- **C-n**: CBMICD1.0: China's building material intensity coefficient dataset (1949–2015)  
  Link: https://doi.org/10.1016/j.resconrec.2020.104824

Data integration includes schema alignment (feature names and units), category normalization, and source-ID tracking to preserve provenance of each record.

## Dataset Size and Training Usage

Using the current preprocessing logic in `prediction_model.ipynb` (`MIN_OBSERVED_TARGETS = 2`, `random_state = 42`):

`MIN_OBSERVED_TARGETS = 2` means each row must have at least 2 non-missing material targets (among Concrete, Glass, Steel, Wood, Brick) to be kept.

| Split      | Rows |
|------------|------|
| Raw database | 2,590 |
| After filtering | 2,570 |
| Training (70%) | 1,799 |
| Validation (15%) | 385 |
| Test (15%) | 386 |

So 1,799 data points are directly used to train model weights, and 2,570 data points are used in the overall model-development pipeline (train + validation + test).

Hyperparameter tuning in the current notebook/script uses Optuna and minimises **validation MASE** computed on presence rows only (`y > 0`). The saved notebook output reports a best validation MASE of `0.624003`.

Sampling (`sample_query`) uses the classifier-chain presence model, post-hoc isotonic calibration, probability clipping and dropout, and a damped residual-covariance perturbation by `Primary Code`. The earlier structural-prior term has been removed in the current workflow.

## Model Performance

The following results are taken from the saved outputs in `prediction_model.ipynb`.

### Test-set conditional intensity performance

These metrics are evaluated on **presence rows only** (`y > 0`), which matches the support of the Stage 2 conditional-intensity model.

| Material | 90% coverage | Mean interval width | p50 MAE |
|----------|--------------|---------------------|---------|
| Concrete | 0.8721 | 2461.61 | 477.54 |
| Glass | 0.8705 | 5.07 | 1.14 |
| Steel | 0.8902 | 84.59 | 18.92 |
| Wood | 0.8690 | 50.06 | 9.33 |
| Brick | 0.8773 | 1142.32 | 207.38 |

Across all five materials, empirical coverage is close to the nominal 90% target, with observed test coverage between `0.8705` and `0.8902`.

### Stronger baseline comparison

On test presence rows, Stage 2 outperforms the training-median baseline for every material, and also improves over a matched `RandomForestRegressor` baseline.

| Material | Median MAE | Ridge MAE | RF MAE | Stage 2 MAE | Improvement vs median | Improvement vs RF |
|----------|------------|-----------|--------|-------------|-----------------------|-------------------|
| Concrete | 662.18 | 605.30 | 486.70 | 477.54 | 27.9% | 1.9% |
| Glass | 1.42 | 1.37 | 1.14 | 1.14 | 19.8% | 0.5% |
| Steel | 30.40 | 25.34 | 20.65 | 18.92 | 37.8% | 8.4% |
| Wood | 17.62 | 13.81 | 11.16 | 9.33 | 47.0% | 16.3% |
| Brick | 301.84 | 219.26 | 218.19 | 207.38 | 31.3% | 5.0% |

### Stage 1 probability calibration

The test-set reliability diagnostics show low expected calibration error (ECE) for material-presence probabilities.

| Material | ECE | Brier score | Verdict |
|----------|-----|-------------|---------|
| Concrete | 0.0339 | 0.0586 | well-calibrated |
| Glass | 0.0401 | 0.0685 | well-calibrated |
| Steel | 0.0203 | 0.0407 | well-calibrated |
| Wood | 0.0234 | 0.0520 | well-calibrated |
| Brick | 0.0505 | 0.0757 | acceptable |

### 5-fold cross-validation

The notebook also reports 5-fold cross-validation on the combined train+validation pool (`n = 2184`), leaving the test set untouched.

| Material | MAE mean | MAE std | CRPS mean | CRPS std | Presence freq error |
|----------|----------|---------|-----------|----------|---------------------|
| Concrete | 445.88 | 21.74 | 0.3196 | 0.0245 | 0.110 |
| Glass | 1.19 | 0.07 | 0.2969 | 0.0193 | 0.034 |
| Steel | 22.02 | 1.60 | 0.3805 | 0.0158 | 0.100 |
| Wood | 10.21 | 1.04 | 0.5183 | 0.0297 | 0.098 |
| Brick | 213.21 | 2.92 | 0.4178 | 0.0536 | 0.089 |

### Residual normality diagnostic

The Stage 2 log-residual diagnostic indicates that the Gaussian residual approximation is strongest for **Steel**, **Wood**, and **Brick**, where recovered and empirical log-scale dispersion are relatively close (`σ_recovered / σ_empirical` of `0.966`, `0.981`, and `0.855`). **Concrete** and **Glass** are more heavy-tailed, with ratios near `0.656` and `0.657`, so their intervals should be interpreted as conservative but imperfect summaries of tail uncertainty.

## Run the Web App

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Train and Export (Script)

Run the current script workflow:

```bash
python prediction_model.py
```

The script mirrors the notebook workflow: data preparation, diagnostics, Optuna tuning, validation reporting, and artifact export.

Generated evaluation figures from the current workflow include:
- `paper_reliability_diagrams.pdf`
- `stage2_log_residual_normality.pdf`
- `appendix_qq_plots.pdf`

Artifacts are saved in the output directory (default: current folder):
- `preprocessor.joblib`
- `model.joblib`
- `model_info.json`
