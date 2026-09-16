# Final Research Report
## Explainable and Robust Machine Learning for Water Quality Assessment and Reduced-Parameter Monitoring

---

## A. Dataset Selected

**Dataset:** Water Quality and Potability (Kadiwal, 2021)  
**Source:** Kaggle (Kadiwal, A. *Water Quality and Potability*. Kaggle, 2021)  
**License:** CC0 Public Domain  

### Why this dataset?

| Criterion | Assessment |
|-----------|-----------|
| **Physicochemical features** | Contains 9 physicochemical variables: pH, Hardness, Solids (TDS-related measure), Chloramines, Sulfate, Conductivity, Organic Carbon, THMs, Turbidity |
| **Binary potability target** | Directly aligns with RQ1 |
| **Publicly accessible** | CC0, no redistribution restrictions |
| **Class imbalance** | Mild (~61/39); manageable with `class_weight="balanced"` |
| **Missing values** | pH (15%), Sulfate (24%), THMs (5%); imputable but documented |
| **Data provenance** | Standard Kaggle benchmark dataset; specific geographic source not documented |
| **Sample size** | 3,276 rows — sufficient for classical ML |

---

## B. Research Gap

### What existing work does:
- Benchmarks Random Forest and XGBoost for water quality classification
- Applies SHAP to water quality regression and classification 
- Evaluates noise robustness for air quality and general environmental ML

### What this project investigates:
This study integrates model benchmarking, SHAP analysis, feature ablation, and noise robustness evaluation within a unified experimental framework. It aims to provide a reproducible, leakage-free benchmark for the widely used Kaggle potability dataset, while exploring the consistency of SHAP-derived feature importances and the sensitivity of tree ensembles to multiplicative measurement noise.

---

## C. Research Questions

| ID | Question |
|----|---------|
| **RQ1** | How accurately can different ML models classify water potability from physicochemical parameters? |
| **RQ2** | Which physicochemical parameters most strongly influence potability predictions, and are these consistent across model families? |
| **RQ3** | How does reducing the number of measured parameters affect predictive performance? |
| **RQ4** | How robust are the developed models to realistic measurement noise? |

---

## D. Experimental Methodology

### Data
- 3,276 samples, 9 features, binary target
- Stratified 70/15/15 split (seed=42), leakage-free preprocessing

### Preprocessing pipeline
1. Exact duplicate removal
2. Training-set median imputation
3. IQR×1.5 outlier capping (training-set statistics only)
4. StandardScaler (training-set fit)
5. `class_weight="balanced"` for imbalance handling

### Models
Dummy, Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, MLP.

### Evaluation Protocol
- **Model selection:** 5-fold stratified cross-validation on the training partition.
- **Final evaluation:** One-time evaluation on the held-out test partition.
- **Primary metric:** **Macro F1** (class imbalance-robust)
- **Secondary metrics:** ROC-AUC, PR-AUC, Accuracy (reported with caveats regarding imbalance)

### SHAP Analysis
- KernelSHAP for generalized feature importance; TreeSHAP for ensemble models.
- Global: mean \|SHAP\| per feature used for establishing the baseline feature ranking. Crucially, this ranking was derived strictly from the **training data** to prevent test-set selection bias (data leakage) in subsequent ablation experiments.

### Feature Ablation (RQ3)
Using the feature ranking derived from the training-set SHAP analysis, models were retrained from scratch on 5 configurations (All, Top-7, Top-5, Top-3, 2-Feature). 

### Noise Robustness (RQ4)
Multiplicative Gaussian noise at {0%, 2%, 5%, 10%, 15%} applied to test features. 5 random repetitions per level; results averaged.

---

## E. Best Model (Results)

After evaluating all models on the held-out test set, given the noisy nature of this dataset and uncertain label provenance, absolute performance ceilings are moderate. 

| Model | Accuracy | Macro F1 | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest** | 0.6829 | **0.6444** | 0.6992 |
| **XGBoost** | 0.6565 | 0.6320 | 0.6882 |
| **Decision Tree** | 0.6463 | 0.6277 | 0.6500 |
| **MLP (Neural Net)** | 0.6829 | 0.6231 | 0.6444 |
| **SVM** | 0.6220 | 0.6043 | 0.6534 |
| **Logistic Regression** | 0.5183 | 0.5097 | 0.5074 |

*Note on metrics:* While accuracy ranges up to 68.3%, Macro F1 (64.4%) provides a more honest assessment of the model's predictive capability across both the majority and minority classes. Random Forest achieved the highest Macro F1 and ROC-AUC among the evaluated models. Both Random Forest and XGBoost outperformed the linear baseline, suggesting that nonlinear relationships and feature interactions may be important for this dataset.

---

## F. Explainability Findings

Using SHAP, we examined global feature importance for Logistic Regression, MLP, and tree-based models. 

For the linear baseline (Logistic Regression), the top 5 most important features were:
1. **Solids (TDS-related measure)** 
2. **pH** 
3. **Chloramines**
4. **Organic Carbon**
5. **Conductivity**

For the MLP Neural Network, **Sulfate** and **Solids (TDS-related measure)** were highly dominant. 

**Critical note:** SHAP values measure model-learned associations. They do not establish that any feature *causes* poor water quality; they merely indicate which variables the model relies on most heavily to minimize loss on this specific dataset.

---

## G. Reduced-Feature Findings (Ablation)

We systematically reduced the number of features provided to the Random Forest model and observed the impact on Macro F1:

| Configuration | Features Used | Macro F1 | ROC-AUC | % of Full Macro F1 |
|---------------|---------------|----------|---------|--------------------|
| **A: Full** | 9 | 0.6444 | 0.6992 | 100% |
| **B: Top 7** | 7 | **0.6514** | 0.6881 | 101% |
| **C: Top 5** | 5 | 0.6294 | 0.6818 | 97.6% |
| **D: Top 3** | 3 | 0.5633 | 0.5908 | 87.4% |
| **E: 2-Feature** | 2 (pH, Turbidity) | 0.4452 | 0.4458 | 69.1% |

**Conclusion:** 
The Top-7 configuration achieved a slightly higher Macro F1 than the full feature set (0.6514 vs. 0.6444), although its ROC-AUC was slightly lower (0.6881 vs. 0.6992). The Top-5 configuration achieved a Macro F1 of 0.6294, corresponding to approximately 97.6% of the full-feature Macro F1 while reducing the input variables from nine to five. These results suggest that some feature reduction may be possible without a large loss in predictive performance. However, the experiment does not directly establish sensor-cost savings or field-level monitoring performance.

---

## H. Robustness Findings

We applied simulated Gaussian multiplicative measurement noise (0% to 15%) to the test set features to simulate variance. Results are presented as Mean ± Standard Deviation across 5 random repetitions:

| Noise Level | Random Forest | XGBoost | Logistic Regression | SVM |
|-------------|---------------|---------|----------------------|-----|
| **0% (Clean)** | 0.6444 ± 0.0000 | 0.6320 ± 0.0000 | 0.5097 ± 0.0000 | 0.6043 ± 0.0000 |
| **2% Noise** | 0.6486 ± 0.0071 | 0.6350 ± 0.0076 | 0.5117 ± 0.0012 | 0.6009 ± 0.0019 |
| **5% Noise** | 0.6552 ± 0.0069 | 0.6338 ± 0.0057 | 0.5092 ± 0.0042 | 0.6016 ± 0.0049 |
| **10% Noise** | 0.6576 ± 0.0079 | 0.6352 ± 0.0037 | 0.5056 ± 0.0047 | 0.5996 ± 0.0041 |
| **15% Noise** | 0.6494 ± 0.0069 | 0.6390 ± 0.0056 | 0.5022 ± 0.0049 | 0.5980 ± 0.0063 |

**Conclusion:** 
Random Forest and XGBoost showed relatively stable Macro F1 under the simulated multiplicative-noise conditions. Random Forest increased from 0.6444 at 0% noise to 0.6576 at 10% noise before decreasing to 0.6494 at 15%, while XGBoost remained within a relatively narrow range across the tested noise levels. These results indicate limited sensitivity to the specific simulated noise model used in this experiment; they should not be interpreted as evidence of robustness to real-world sensor errors.

---

## I. Overall Findings

Across the evaluated models, Random Forest achieved the highest Macro F1 (0.6444) and ROC-AUC (0.6992) on the held-out test set. SHAP analysis identified different feature-importance patterns across model families, reinforcing that model interpretation is dependent on the learned predictive structure.

Feature ablation showed that reducing the input space from nine to five variables resulted in a Macro F1 of 0.6294, corresponding to 97.6% of the full-feature Macro F1. Under the specified multiplicative-noise experiment, Random Forest and XGBoost maintained relatively stable performance across noise levels up to 15%.

Taken together, the experiments demonstrate a reproducible framework for evaluating predictive performance, interpretability, feature reduction, and sensitivity to simulated measurement noise on this dataset. The findings should be interpreted as dataset-specific evidence rather than as validation of real-world water-monitoring systems.

---

## J. Research Contribution and Limitations

### Research Contribution
This project provides a reproducible experimental framework combining model benchmarking, SHAP-based interpretability, feature ablation, and simulated measurement-noise evaluation on a publicly available water-potability dataset. The experiments provide quantitative evidence regarding model performance, feature reduction, and sensitivity to the specified noise model.

### Current Limitations for Publication
The main limitations are the uncertain provenance of the dataset, absence of external validation, reliance on a single dataset, simplified noise assumptions, and the lack of temporal or field-sensor data. Further validation on independently collected water-quality datasets would strengthen the generalizability of the findings.

---


