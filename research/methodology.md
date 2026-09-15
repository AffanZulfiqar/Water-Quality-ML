# Methodology

**Project:** Explainable and Robust Machine Learning for Water Quality Assessment and Reduced-Parameter Monitoring

---

## 1. Research Questions

| ID | Question |
|----|---------|
| **RQ1** | How accurately can different ML models classify water potability from physicochemical parameters? |
| **RQ2** | Which physicochemical parameters most strongly influence potability predictions, and are these rankings consistent across model families? |
| **RQ3** | How does reducing the number of measured parameters affect predictive performance? |
| **RQ4** | How robust are the developed models to realistic measurement noise? |

---

## 2. Dataset

**Primary dataset:** Water Quality and Potability (Kadiwal, 2021)  
Source: https://www.kaggle.com/datasets/adityakadiwal/water-potability  
License: CC0 Public Domain

- 3,276 samples
- 9 physicochemical features: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
- Binary target: Potability (0 = not potable, 1 = potable)
- Class imbalance: ~61% class 0, ~39% class 1
- Missing values: pH (~15%), Sulfate (~24%), Trihalomethanes (~5%)

---

## 3. Data Preprocessing

Preprocessing is implemented in `src/preprocessing/pipeline.py` and follows strict leakage-prevention rules:

### Step-by-step
1. **Duplicate removal:** Exact row duplicates are removed before splitting.
2. **Stratified split:** 70% train / 15% validation / 15% held-out test (stratified by target class).
3. **Missing value imputation:** Training-set median per feature. Applied to all splits.
4. **Outlier capping:** IQR × 1.5 fence computed from training data. Values clipped at fence.
5. **Feature scaling:** `StandardScaler` fitted on training data, applied to all splits.
6. **Class imbalance:** Handled via `class_weight="balanced"` for applicable models. Not SMOTE (to avoid information leakage risk with small datasets).

### Leakage prevention rules
- Steps 3–5 use statistics computed from training data ONLY.
- The test set is accessed only once, during final evaluation.
- Feature importance for ablation is computed from training-set SHAP values only.

---

## 4. Benchmark Models

Eight models are implemented and compared:

| # | Model | Rationale |
|---|-------|-----------|
| 1 | Dummy (majority-class) | Lower bound reference |
| 2 | Logistic Regression | Linear baseline |
| 3 | Decision Tree | Interpretable non-linear baseline |
| 4 | Random Forest | Strong ensemble baseline; primary SHAP model |
| 5 | SVM (RBF kernel) | Kernel method comparison |
| 6 | XGBoost | State-of-the-art gradient boosting |
| 7 | LightGBM | Efficient gradient boosting; cross-SHAP comparison |
| 8 | MLP (3-layer) | Shallow neural network comparison |

All models are fitted with fixed random seeds (seed = 42, configurable in `configs/config.yaml`).

---

## 5. Model Evaluation

### Metrics used
For the imbalanced binary classification task:

- **Primary metric:** Macro F1-score (equal weight to both classes)
- **Secondary metrics:** ROC-AUC, PR-AUC, Balanced Accuracy
- **Supplementary:** Class-wise precision, recall, F1 (to detect class-specific failure modes)
- **Avoided as sole metric:** Accuracy (can be inflated by class imbalance)

### Evaluation protocol
- **Cross-validation:** 5-fold Stratified K-Fold on training data (mean ± std reported)
- **Final evaluation:** Single evaluation on held-out test set (15% of data)
- **Hyperparameter tuning:** RandomizedSearchCV on training+validation splits; test set untouched

---

## 6. Explainability (SHAP)

SHAP (SHapley Additive Explanations) analysis is performed on all models using:
- **TreeSHAP** (exact, fast) for Random Forest, XGBoost, LightGBM, Decision Tree
- **KernelSHAP** (approximate) for Logistic Regression, SVM, MLP

### Global explanations
- Mean absolute SHAP value per feature → global feature importance ranking
- SHAP beeswarm summary plot (shows direction of effect)

### Local explanations
- Waterfall plots for:
  - 2 correctly classified samples
  - 2 false positives (predicted potable but not potable)
  - 2 false negatives (predicted not potable but potable)
  - 2 high-confidence predictions

### Cross-model consistency
SHAP rankings are compared across all models to assess whether important features are model-agnostic (more useful for practical reduced-sensor deployment).

**Caution on interpretation:** SHAP values indicate the model's learned association between features and predictions. They do NOT establish causality. Language such as "the model identified X as strongly associated with the prediction" is used, not "X causes unsafe water quality."

---

## 7. Feature Ablation Experiment (RQ3)

Feature subsets are built from training-set SHAP importance rankings:

| Configuration | Features Included |
|--------------|------------------|
| A — Full | All 9 features |
| B — Top 7 | Top 7 by training SHAP |
| C — Top 5 | Top 5 by training SHAP |
| D — Top 3 | Top 3 by training SHAP |
| E — Low-cost | pH, Turbidity, Conductivity (proxy accessible sensors) |

For each configuration:
- Model is re-trained from scratch on the reduced training set.
- Evaluated on the same held-out test set (same target, reduced feature subset).
- No information from the test set is used in feature selection.

Result: A curve of **Number of features vs. Macro F1 / ROC-AUC**.

The goal is not to recommend the smallest set but to quantify the performance–measurement cost trade-off.

---

## 8. Noise Robustness Experiment (RQ4)

### Noise model
Multiplicative Gaussian noise is applied at evaluation time:

```
x_noisy = x * (1 + ε),  ε ~ N(0, noise_level)
```

This simulates proportional sensor uncertainty (common in pH, conductivity, and turbidity sensors), where measurement error scales with the measured value.

### Noise levels tested
0%, 2%, 5%, 10%, 15%

### Protocol
- Models are trained on **clean** training data.
- Noise is applied **only to test set features** — never to training data or labels.
- Each noise level is repeated 5 times with different random seeds; results are averaged.
- Confidence intervals are reported.

### Motivation
This simulates the degraded sensing conditions that would occur in practical IoT-based monitoring (biofouling, sensor drift, electrical interference). The study does NOT claim to represent any specific sensor or hardware system.

---

## 9. Statistical Analysis

- Cross-validation provides mean ± std for inter-fold variance estimation.
- Results from different noise repetitions are averaged with reported std.
- Paired model comparisons are discussed qualitatively; no formal significance tests are used for the primary benchmark due to the small number of folds (5).
- No p-values are reported without proper multiple-test correction.

---

## 10. Hardware and Software Environment

Reported at the end of each experiment run. Document:
- Python version
- Library versions (from `requirements.txt`)
- Hardware (CPU/RAM; GPU not required for this project)
- Operating system

Example:
```
Python: 3.11.x
scikit-learn: 1.3.x
xgboost: 2.0.x
lightgbm: 4.0.x
shap: 0.44.x
numpy: 1.24.x
pandas: 2.0.x
```

---

## 11. Reproducibility Checklist

- [ ] Random seed 42 set globally and in all model constructors
- [ ] Train/val/test split performed with stratification
- [ ] Preprocessing fitted on training data only
- [ ] Cross-validation uses StratifiedKFold
- [ ] SHAP computed on training data for feature ranking
- [ ] Ablation models re-trained from scratch for each configuration
- [ ] Robustness noise applied only at inference time
- [ ] All numerical results loaded from saved CSV/JSON files
- [ ] No manually typed numbers in tables or paper

---

*End of Methodology Document*
