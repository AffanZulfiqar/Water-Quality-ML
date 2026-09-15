# Explainable and Robust Machine Learning for Water Quality Assessment and Reduced-Parameter Monitoring

A complete, reproducible research project investigating machine learning for drinking water potability classification, with SHAP-based explainability, feature ablation, and noise-robustness experiments.

> **Motivation:** This research project was motivated by practical exposure to water-filtration challenges during the author's work with Porteco, a water-filtration startup. The experimental study itself uses publicly available datasets and is an independent research project.

> ⚠️ **Research Prototype:** Models in this project are research tools trained on a public dataset of uncertain provenance. They are **not** certified water safety systems and must not be used to make decisions about drinking water safety.

---

## Research Questions

| RQ | Question |
|----|---------|
| **RQ1** | How accurately can different ML models classify water potability from physicochemical parameters? |
| **RQ2** | Which physicochemical parameters most strongly influence potability predictions, and are these consistent across models? |
| **RQ3** | How does reducing the number of measured parameters affect predictive performance? |
| **RQ4** | How robust are the developed models to realistic measurement noise? |

---

## Dataset

**Water Quality and Potability** — Kaggle (Kadiwal, A. 2021)  
License: CC0 Public Domain  

| Property | Value |
|----------|-------|
| Samples | 3,276 |
| Features | 9 physicochemical |
| Target | Binary potability (0/1) |
| Class ratio | ~61% non-potable / ~39% potable |
| Missing values | pH (15%), Sulfate (24%), THMs (5%) |

**Features:** pH, Hardness, Total Dissolved Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity

**Known limitations:** The dataset's primary data source is not documented. It likely represents aggregated or synthetic data rather than measurements from a specific regulatory monitoring program. Results cannot be assumed to generalize to specific real-world water systems.

---

## Models Benchmarked

1. Dummy Classifier (majority-class baseline)
2. Logistic Regression
3. Decision Tree
4. **Random Forest** ← primary model for SHAP analysis
5. SVM (RBF kernel)
6. **XGBoost**
7. **LightGBM**
8. MLP Neural Network (3-layer)

---

## Experiments

### Experiment 1 — Model Benchmark (RQ1)
All 8 models evaluated on 5-fold stratified cross-validation.  
Metrics: Macro F1, ROC-AUC, PR-AUC, Balanced Accuracy, class-wise precision/recall.

### Experiment 2 — SHAP Explainability (RQ2)
- Global feature importance (mean |SHAP|) for all models
- SHAP summary beeswarm plots
- Local waterfall plots for: correct predictions, false positives, false negatives, high-confidence cases
- Cross-model SHAP ranking comparison

### Experiment 3 — Feature Ablation (RQ3)
| Config | Features |
|--------|---------|
| A — Full | All 9 |
| B — Top 7 | Top 7 by training SHAP |
| C — Top 5 | Top 5 by training SHAP |
| D — Top 3 | Top 3 by training SHAP |
| E — Low-cost | pH, Turbidity, Conductivity |

### Experiment 4 — Noise Robustness (RQ4)
Multiplicative Gaussian noise at {0%, 2%, 5%, 10%, 15%} applied to test features.  
Models trained on clean data; noise applied only at evaluation time.

---

## Installation

```bash
git clone <repo-url>
cd water-quality-ml

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Data Download

The dataset is not included in this repository (no redistribution needed; CC0 license allows self-download).

**Option 1 — Kaggle CLI:**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d adityakadiwal/water-potability -p data/ --unzip
```

**Option 2 — Manual:**
1. Visit: https://www.kaggle.com/datasets/adityakadiwal/water-potability
2. Download and extract `water_potability.csv` to `data/`

**Validate download:**
```bash
python scripts/prepare_data.py
```

---

## Reproduction — Full Pipeline

Run experiments in order:

```bash
# Step 1: Validate dataset
python scripts/prepare_data.py

# Step 2: Train all models + cross-validation
python scripts/train_models.py

# Step 3: Hyperparameter optimization (optional, slower)
python scripts/train_models.py --tune

# Step 4: SHAP explainability analysis
python scripts/run_explainability.py

# Step 5: Feature ablation experiment
python scripts/run_ablation.py

# Step 6: Noise robustness experiment
python scripts/run_robustness.py

# Step 7: Generate all tables and EDA figures
python scripts/generate_results.py
```

All results are saved to `results/` (metrics as CSV/JSON, figures as PNG).

---

## Project Structure

```
water-quality-ml/
├── README.md
├── LICENSE                          MIT
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md                    Dataset provenance + download instructions
│
├── configs/
│   └── config.yaml                  All experiment settings (seeds, paths, etc.)
│
├── src/
│   ├── data/
│   │   └── loader.py                Dataset loading + leakage-safe splitting
│   ├── preprocessing/
│   │   └── pipeline.py              Imputation, outlier capping, scaling
│   ├── models/
│   │   └── registry.py              Model factory + hyperparameter search spaces
│   ├── evaluation/
│   │   └── metrics.py               Metrics, cross-validation, serialization
│   ├── explainability/
│   │   └── shap_analysis.py         SHAP global/local analysis
│   ├── experiments/
│   │   ├── ablation.py              Feature ablation experiment
│   │   └── robustness.py            Noise robustness experiment
│   └── visualization/
│       └── plots.py                 All publication-quality figures
│
├── scripts/
│   ├── prepare_data.py              Download + validate dataset
│   ├── train_models.py              Train + evaluate all models
│   ├── run_explainability.py        SHAP analysis
│   ├── run_ablation.py              Feature ablation
│   ├── run_robustness.py            Noise robustness
│   └── generate_results.py          Tables + EDA figures
│
├── results/
│   ├── metrics/                     JSON/CSV per-model metrics
│   ├── models/                      Saved fitted model files (.pkl)
│   ├── figures/                     EDA, model comparison, SHAP, ablation, robustness
│   ├── tables/                      Publication tables (CSV)
│   ├── shap/                        Raw SHAP values, ranking comparisons
│   ├── ablation/                    Per-configuration ablation results
│   └── robustness/                  Per noise-level results
│
├── research/
│   ├── literature_review.md         18 papers with full bibliographic records
│   ├── research_gap.md              Evidence-based gap analysis
│   ├── methodology.md               Detailed experimental methodology
│   └── experiment_plan.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_benchmarking.ipynb
│   ├── 03_explainability.ipynb
│   ├── 04_feature_ablation.ipynb
│   └── 05_robustness_analysis.ipynb
│
└── paper/
    ├── manuscript.md                Conference paper draft
    ├── abstract.md
    ├── references.bib               Verified BibTeX entries
    └── figures/
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Macro F1 as primary metric | Dataset is imbalanced (~61/39); accuracy would inflate scores |
| `class_weight="balanced"` over SMOTE | Avoids SMOTE leakage risk; simpler and well-justified |
| Median imputation over KNN | Faster; sufficient for this dataset size; avoids k-nearest-neighbor global information leakage |
| IQR capping over removal | Preserves sample count; avoids removing valid extreme measurements |
| TreeSHAP for tree models | Exact computation vs. KernelSHAP approximation |
| Multiplicative noise model | More physically realistic than additive for ratio-scale measurements |
| SHAP-based feature ranking | Provides model-grounded, comparable ranking vs. arbitrary heuristic |

---

## Limitations

- **Dataset provenance:** The Kaggle dataset has no documented chain of custody. Results cannot be assumed to generalize to specific geographic water systems.
- **Class imbalance:** Moderate (61/39) imbalance may still cause bias in certain model families.
- **No temporal or spatial data:** Static snapshot dataset; no concept drift analysis.
- **Noise model is simplified:** Multiplicative Gaussian noise does not capture all sensor failure modes (biofouling, drift, saturation).
- **No external validation:** No second independent dataset with compatible features was identified for cross-dataset validation.
- **Dataset size:** n ≈ 3,000 is sufficient for classical ML but limits statistical power.

---

## Publication Assessment

*(Preliminary — to be updated after experiments produce real results)*

| Quality Level | Assessment |
|--------------|-----------|
| **Portfolio quality** | Yes — demonstrates solid ML engineering, research methodology, and academic writing |
| **Undergraduate research quality** | Yes — exceeds typical undergraduate project scope |
| **Workshop/short paper potential** | Possible — if results are strong and experimental framework is well-presented |
| **Full conference paper potential** | Requires: (1) actual experimental results replacing all placeholders, (2) independent DOI verification of all citations, (3) statistical testing for model comparisons, (4) ideally a second validation dataset |

**What would strengthen the paper:**
- An independent validation dataset (even partial feature alignment)
- Formal Friedman+Nemenyi statistical comparison across models
- Ablation analysis on a second, geographically distinct dataset
- Comparison with a reported state-of-the-art result on the Kaggle dataset

---

## Citation

If you use this code or research in your work:

```bibtex
@misc{waterqualityml2026,
  title  = {Explainable and Robust Machine Learning for Water Quality Assessment
            and Reduced-Parameter Monitoring},
  author = {[Author Name]},
  year   = {2026},
  url    = {[repository URL]}
}
```

---

## Acknowledgements

This project uses the Water Potability dataset released by Aditya Kadiwal on Kaggle under the CC0 Public Domain license. The project was independently conducted using publicly available data and tools.

---

*Last updated: September 2026*
