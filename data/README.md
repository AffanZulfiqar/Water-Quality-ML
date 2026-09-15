# Data Directory

## Overview

This directory holds the dataset(s) used in the research project:

> **Explainable and Robust Machine Learning for Water Quality Assessment and Reduced-Parameter Monitoring**

---

## Primary Dataset

### Water Quality and Potability Dataset

| Field | Details |
|-------|---------|
| **Dataset Name** | Water Quality and Potability |
| **Author** | Aditya Kadiwal |
| **Source** | Kaggle |
| **Original URL** | https://www.kaggle.com/datasets/adityakadiwal/water-potability |
| **License** | CC0: Public Domain |

| **Number of Samples** | 3,276 |
| **Number of Features** | 9 physicochemical features + 1 binary target |
| **Target Variable** | `Potability` (0 = not potable, 1 = potable) |
| **Task Type** | Binary Classification |
| **File Format** | CSV |

### Feature Descriptions

| Feature | Unit | Description |
|---------|------|-------------|
| `pH` | dimensionless | Acidity/alkalinity of water (WHO acceptable range: 6.5–8.5) |
| `Hardness` | mg/L | Caused by calcium and magnesium salts |
| `Solids` | mg/L (ppm) | Total dissolved solids (TDS); high TDS degrades taste |
| `Chloramines` | ppm | Disinfectant added to municipal water supplies |
| `Sulfate` | mg/L | Naturally occurring ion; above 250 mg/L may cause laxative effects |
| `Conductivity` | μS/cm | Proxy for total ion concentration |
| `Organic_carbon` | ppm | Total organic carbon; high levels may indicate pollution |
| `Trihalomethanes` | μg/L | Disinfection byproducts formed during chlorination |
| `Turbidity` | NTU | Cloudiness or haziness; WHO target: < 1 NTU |

### Target Variable

- `Potability = 0`: Water is **not safe** for human consumption
- `Potability = 1`: Water is **safe** for human consumption

---

## Known Limitations of This Dataset

> **IMPORTANT:** These limitations must be acknowledged in any paper or analysis using this dataset.

1. **Uncertain Provenance:** The dataset was uploaded to Kaggle by Aditya Kadiwal. The original primary data source is not documented in the Kaggle metadata. It appears to be aggregated or synthetically generated rather than drawn from a single certified monitoring study. Do NOT describe this dataset as coming from a specific regulatory body or geographic source.

2. **Missing Values:** The dataset contains significant missing values:
   - `pH`: approximately 491 missing values (~15%)
   - `Sulfate`: approximately 781 missing values (~24%)
   - `Trihalomethanes`: approximately 162 missing values (~5%)
   - All other features: complete

3. **Class Imbalance:** Approximately 61% of samples are labeled non-potable (0) and 39% are potable (1). Imbalance-aware evaluation metrics (macro F1, PR-AUC, balanced accuracy) should be used.

4. **No Geographic or Temporal Metadata:** There are no timestamps, locations, source types (groundwater, surface water, tap), or treatment histories associated with samples.

5. **Generalizability:** Models trained on this dataset cannot be assumed to generalize to specific real-world water systems without further validation.

---

## Missing Value Handling

Missing values are imputed using the **median** of the training set feature, computed after stratified splitting.

**Critical leakage prevention rule:** Imputation statistics (medians) are computed exclusively on the training split and applied to validation and test splits. The median of the full dataset is never used for imputation.

---

## Train / Validation / Test Strategy

- Stratified random split by the `Potability` label (to maintain class ratio in each split).
- **Test set (15%):** 492 samples — held out completely until final evaluation.
- **Validation set (15%):** 492 samples — used for hyperparameter tuning and early stopping.
- **Training set (70%):** 2,292 samples — used for fitting models and computing feature importance.
- **Random seed:** 42 (set in `configs/config.yaml`)

---

## Preprocessing Summary

1. Remove exact duplicate rows.
2. Impute missing values with training-set median per feature.
3. Cap outliers at IQR × 1.5 fence on training set; apply same caps to validation and test.
4. Apply `StandardScaler` (zero mean, unit variance) fitted on training set.
5. Handle class imbalance via `class_weight="balanced"` in applicable models.

A complete, reproducible preprocessing pipeline is implemented in `src/preprocessing/pipeline.py`.

---

## Dataset Files

```
data/
├── README.md                        ← This file
├── water_potability.csv             ← Raw dataset (download manually from Kaggle)
└── .gitignore                       ← Excludes CSV from git if large
```

> **To download the dataset:**
>
> 1. Install the Kaggle CLI: `pip install kaggle`
> 2. Configure your Kaggle API credentials: place `kaggle.json` in `~/.kaggle/`
> 3. Run: `kaggle datasets download -d adityakadiwal/water-potability -p data/ --unzip`
>
> Alternatively, download manually from:
> https://www.kaggle.com/datasets/adityakadiwal/water-potability
> and place `water_potability.csv` in this `data/` directory.

---

## Data Provenance Declaration

This dataset is used under the CC0 Public Domain license. No redistribution restrictions apply. The dataset is **not** included in this repository because its provenance from a third-party source is documented separately.

This research project uses no proprietary, confidential, or company-specific data. All experiments are reproducible from publicly available data.

---

## Optional: Secondary / External Validation Dataset

If an appropriate secondary public dataset becomes available with compatible features, it should be documented in this file with the same fields as above. At the time of initial project setup, no sufficiently compatible secondary dataset has been identified. The UCI Water Quality Prediction dataset (sensor time-series from Georgia, USA) uses a different feature set and target variable (next-day pH forecast) and is not compatible for direct validation transfer.

---


