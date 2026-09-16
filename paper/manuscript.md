# Explainable ML for Water Potability Classification and Reduced-Parameter Monitoring

---

## Abstract

*(To be completed after experiments are run. Template below.)*

Safe drinking water access remains a global health priority. Conventional physicochemical assessment relies on laboratory-based testing that is time-consuming, expensive, and unsuitable for real-time monitoring. Machine learning offers the potential for rapid automated water quality assessment, but practical deployment is hindered by (1) limited interpretability of black-box models, (2) the cost of measuring a large number of parameters, and (3) uncertainty about model behavior under realistic sensor noise. This paper presents a systematic experimental study addressing these barriers.

Using the publicly available Water Potability dataset (Kadiwal, 2021; 3,276 samples; 9 physicochemical parameters), we benchmark eight machine learning classifiers and apply SHAP (SHapley Additive Explanations) to quantify global and local feature contributions. We then conduct a controlled feature-ablation study to evaluate performance under progressively reduced parameter sets, and a noise-robustness experiment simulating proportional sensor measurement uncertainty at noise levels of 0–15%.

[**Results will be inserted here after experiments complete.**]

Our findings suggest that ensemble tree-based methods achieve the strongest classification performance on this dataset, that a subset of [N] parameters retains [X]% of full-feature performance, and that tree-based models degrade more gracefully than kernel and linear methods under measurement noise. These findings support the feasibility of cost-effective, explainable ML-assisted water quality monitoring.

---

## 1. Introduction

Access to safe drinking water is recognized as a fundamental human right by the United Nations, yet approximately two billion people worldwide lack safely managed drinking water services [WHO/UNICEF, 2023]. Water quality assessment is the foundation of safe water supply, but conventional methods based on laboratory physicochemical analysis are expensive, labor-intensive, and provide only retrospective — not real-time — information.

The emergence of low-cost sensor technologies and the internet of things (IoT) has created opportunities for continuous, in-situ water monitoring. However, translating raw sensor measurements into actionable quality assessments remains a challenge. Regulatory frameworks typically evaluate 20–50 parameters, many of which require specialized laboratory equipment [WHO, 2022]. Practical monitoring systems must therefore balance measurement thoroughness with operational cost.

Machine learning methods offer promising capabilities for water quality classification and index prediction. Ensemble methods such as Random Forest and gradient-boosted trees have demonstrated strong performance across multiple water quality datasets [Mohammed & Kora, 2023; Zhu et al., 2022]. However, several critical gaps remain in the literature.

**First**, the majority of ML water quality studies report only aggregate classification accuracy, which can be misleading under class imbalance [Singha et al., 2021]. **Second**, while SHAP-based explainability has been applied in individual studies [Bui et al., 2021; Pal et al., 2022], a systematic cross-model comparison of SHAP-derived feature rankings for the binary potability task has not been conducted. **Third**, the question of how much predictive performance is retained when only a minimal subset of parameters is available has been addressed only for regression-based WQI prediction [Ly et al., 2021], not for potability classification. **Fourth**, model robustness to sensor measurement noise has been studied for air quality monitoring [Chen et al., 2023] but not specifically for drinking water potability classification.

This paper addresses all four gaps within a single, coherent experimental framework on a common publicly available dataset. The contributions of this study are:

1. A rigorous, reproducible benchmark of eight ML classifiers for water potability classification using imbalance-aware evaluation metrics and cross-validation.
2. A SHAP-based explainability analysis that quantifies both global feature importance and local prediction explanations, with cross-model consistency assessment.
3. A feature-ablation experiment that quantifies the performance–measurement cost trade-off across five feature configurations (full set to minimal 3-feature set).
4. A controlled noise-robustness evaluation comparing model degradation under multiplicative measurement noise (0–15% noise level).

---

## 2. Related Work

*(Summarized from `research/literature_review.md`. Expand with specific results when writing the final manuscript.)*

**Machine learning for water quality classification.** Early work by Ahmed et al. (2019) reviewed over 100 ML studies for water quality monitoring, finding ANN and ensemble methods to be dominant but noting a lack of standardized evaluation. Mohammed & Kora (2023) applied seven classifiers to the Kaggle Water Potability dataset, confirming that ensemble methods outperform single learners and that imbalance handling is critical. Dawood et al. (2023) found gradient-boosted trees to generalize best across multiple US EPA datasets.

**SHAP-based explainability.** Bui et al. (2021) introduced SHAP analysis for groundwater quality classification in Vietnam, identifying pH and electrical conductivity as dominant features. Zhu et al. (2022) applied XGBoost and LightGBM with SHAP to the Yangtze River, finding dissolved oxygen and permanganate index most influential. Pal et al. (2022) used local SHAP waterfall plots to explain individual potability predictions from West Bengal groundwater data. Kumar et al. (2024) compared SHAP rankings across four model families for Indian river WQI, finding XGBoost and CatBoost largely consistent.

**Feature reduction.** Ly et al. (2021) demonstrated that a 5-feature subset of 14 parameters retained >95% performance for WQI regression in Korean rivers. To our knowledge, no analogous study exists for binary potability classification.

**Robustness to sensor noise.** Leuenberger & Kanevski (2022) showed that tree-based ensembles degrade more gracefully than SVM and MLP under Gaussian noise in environmental datasets. Chen et al. (2023) extended this finding to air quality monitoring with IEEE sensor data. Neither study investigated water potability specifically.

This study synthesizes insights from these work streams and integrates model benchmarking, SHAP analysis, feature ablation, and noise robustness evaluation within a unified experimental framework for drinking water potability classification.

---

## 3. Dataset and Methodology

### 3.1 Dataset

The Water Quality and Potability dataset (Kadiwal, 2021) is an open-access dataset (CC0 Public Domain) available from Kaggle. It contains 3,276 samples with nine physicochemical parameters as features and a binary potability label (0 = not potable, 1 = potable).

**Known limitations.** The primary data source is not documented in the Kaggle metadata. The dataset likely represents aggregated or synthetic data rather than a single certified monitoring study. It lacks geographic, temporal, and water-source type metadata. All claims based on this dataset must be appropriately scoped.

**Feature set:** pH (dimensionless), Hardness (mg/L), Total Dissolved Solids (ppm), Chloramines (ppm), Sulfate (mg/L), Conductivity (μS/cm), Organic Carbon (ppm), Trihalomethanes (μg/L), Turbidity (NTU).

**Class distribution:** 61.2% not potable (class 0), 38.8% potable (class 1) — mild to moderate imbalance.

**Missing values:** pH (15.0%), Sulfate (23.8%), Trihalomethanes (4.9%).

### 3.2 Preprocessing

Preprocessing follows a strict leakage-prevention protocol. All fitting operations (imputation statistics, outlier bounds, scaler parameters) are computed exclusively from training data and applied to validation and test sets without re-fitting.

The preprocessing pipeline:
1. Remove exact duplicate rows (pre-split).
2. Stratified 70/15/15 train/validation/test split (seed = 42).
3. Impute missing values with training-set median per feature.
4. Cap outliers at training-set IQR × 1.5 fence.
5. Apply StandardScaler fitted on training data.
6. Handle class imbalance via `class_weight="balanced"` where supported.

### 3.3 Benchmark Models

Eight classifiers are evaluated:

| Model | Implementation |
|-------|---------------|
| Majority-class Dummy | scikit-learn DummyClassifier |
| Logistic Regression | scikit-learn LogisticRegression |
| Decision Tree | scikit-learn DecisionTreeClassifier |
| Random Forest | scikit-learn RandomForestClassifier |
| SVM (RBF kernel) | scikit-learn SVC |
| XGBoost | xgboost XGBClassifier |
| LightGBM | lightgbm LGBMClassifier |
| MLP (3-layer) | scikit-learn MLPClassifier |

### 3.4 Evaluation Protocol

- **Primary metric:** Macro F1-score (treats both classes equally)
- **Secondary:** ROC-AUC, PR-AUC, Balanced Accuracy
- **Cross-validation:** 5-fold Stratified K-Fold on training data
- **Hyperparameter tuning:** RandomizedSearchCV (50 iterations) for Random Forest, XGBoost, LightGBM

### 3.5 SHAP Explainability

TreeSHAP is applied to tree-based models; KernelSHAP to others. SHAP values are computed on the test set. Global importance is the mean absolute SHAP value per feature. Local waterfall plots are generated for four example categories.

### 3.6 Feature Ablation

Features are ranked by training-set SHAP importance. Five configurations are evaluated (Full, Top-7, Top-5, Top-3, Low-cost proxy). For each configuration, models are re-fitted from scratch on the reduced training set and evaluated on the held-out test set.

### 3.7 Noise Robustness

Multiplicative Gaussian noise is applied to test-set features at levels {0%, 2%, 5%, 10%, 15%}:
$$x_\text{noisy} = x \cdot (1 + \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, \sigma)$$
where σ is the noise level. This is repeated five times per level; results are averaged. Labels are never modified.

---

## 4. Experimental Setup

**Random seed:** 42 (all experiments)  
**Python:** 3.11  
**scikit-learn:** ≥ 1.3.0  
**xgboost:** ≥ 2.0.0  
**lightgbm:** ≥ 4.0.0  
**shap:** ≥ 0.44.0  

Hardware and software environment are recorded at run time in experiment log files.

---

## 5. Results

*(All numbers in this section are placeholders. Run the experimental pipeline and replace with actual values.)*

### 5.1 Model Benchmark (Table 2)

| Model | Acc. | Bal. Acc. | Macro F1 | ROC-AUC | PR-AUC |
|-------|------|-----------|----------|---------|--------|
| Dummy | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| SVM | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| LightGBM | — | — | — | — | — |
| MLP | — | — | — | — | — |

### 5.2 Cross-Validation (Table 3)

*(To be populated from results/tables/table3_cv_results.csv)*

### 5.3 SHAP Feature Importance (Table 4)

*(To be populated from results/tables/table4_shap_importance.csv)*

**Preliminary expected finding (based on literature):** Solids/TDS, Sulfate, and pH are frequently among the most predictive features for potability in this dataset, though the actual SHAP ranking may differ and must be reported from experimental output.

### 5.4 Feature Ablation (Table 5)

*(To be populated from results/tables/table5_ablation_results.csv)*

### 5.5 Noise Robustness (Table 6)

*(To be populated from results/tables/table6_robustness_results.csv)*

---

## 6. Discussion

*(Template — to be written after experiments produce real results)*

### 6.1 Model Performance

Discuss:
- Which model achieved the highest Macro F1 and why this metric is appropriate for the imbalanced setting.
- The gap between best-performing model and the dummy baseline.
- Whether the cross-validation variance is acceptable.
- Any unexpected model behaviors.

### 6.2 Feature Importance and Cross-Model Consistency

Discuss:
- Which features had the highest SHAP importance for the primary model.
- Whether SHAP rankings are consistent across model families.
- Physical interpretation of the top features (pH range, TDS, conductivity) in the context of WHO drinking water guidelines.
- Explicit caution: these are model associations, not causal relationships.

### 6.3 Feature Ablation

Discuss:
- The performance drop from full-feature to reduced configurations.
- Whether a "knee" in the performance curve exists (point of maximum efficiency).
- Practical implications: what level of parameter reduction is acceptable for a monitoring system?
- Limitations: the ablation uses a fixed ranking; other selection methods (e.g., RFE, mutual information) might yield different subsets.

### 6.4 Robustness to Measurement Noise

Discuss:
- Which models degrade most gracefully under noise.
- At what noise level does performance drop significantly.
- Whether the clean-data model ranking holds under noise.
- Limitations: the noise model is simplified and does not represent any specific sensor.

### 6.5 Practical Implications

Discuss the feasibility of:
- Using a small subset of low-cost sensors with ML.
- The importance of knowing which noise level a specific sensing system operates at before selecting a model.
- The role of explainability in building operator trust.

### 6.6 Limitations

- Uncertain dataset provenance; results may not generalize to specific real water systems.
- No geographic, temporal, or source-type information in the dataset.
- Small dataset size (n ≈ 3,000) limits statistical power.
- No real sensor data; noise robustness is based on a simplified multiplicative model.
- No external validation dataset with compatible features.

---

## 7. Conclusion

*(Template — to be written after experiments)*

This study presents a systematic, reproducible experimental investigation of ML methods for water potability classification. The combination of model benchmarking, SHAP explainability, feature ablation, and noise robustness analysis in a single experimental framework provides a more complete view of the practical deployment trade-offs than any individual component would.

Key findings:
1. [Best model] achieved the highest Macro F1 of [X.XXX ± X.XXX] under 5-fold cross-validation.
2. SHAP analysis identified [feature 1], [feature 2], and [feature 3] as the most consistently important predictors.
3. [N] features retained [X]% of full-feature performance, suggesting a viable reduced-parameter monitoring configuration.
4. Tree-based ensemble models showed greater robustness to multiplicative measurement noise than kernel-based and linear models.

---

## Future Work

1. **Real sensor data:** Validate on datasets with documented sensor chain-of-custody and known measurement uncertainty.
2. **Temporal and geographic generalization:** Evaluate models across sites and seasons.
3. **Hardware-aware noise models:** Characterize actual noise distributions for specific sensor types.
4. **Edge deployment:** Evaluate latency and memory constraints for embedded deployment.
5. **Online learning:** Investigate adaptive models that update as water source conditions change.
6. **Multi-class prediction:** Extend beyond binary potability to WHO guideline compliance level prediction.

---

## References

*(BibTeX format — see `paper/references.bib`)*

[1] Kadiwal, A. (2021). Water Quality and Potability. Kaggle. https://www.kaggle.com/datasets/adityakadiwal/water-potability  
[2] Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.  
[3] Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16.  
[4] Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 30.  
[5] Mohammed, A. & Kora, R. (2023). Predicting water quality using machine learning. Journal of Cleaner Production.  
[6] Bui, D.T. et al. (2021). Explainable machine learning for water quality prediction. Science of the Total Environment, 794.  
[7] Zhu, M. et al. (2022). Water quality prediction using gradient boosting with SHAP. npj Clean Water, 5.  
[8] Ly, Q.V. et al. (2021). Reducing monitoring cost through feature selection for ML-based WQI prediction. Journal of Environmental Management, 287.  
[9] Leuenberger, M. & Kanevski, M. (2022). Robustness of ML classifiers to noisy features. Environmental Modelling & Software, 149.  
[10] Chen, Y. et al. (2023). Sensor noise effects on ML models for environmental monitoring. IEEE Trans. Instrumentation and Measurement, 72.  
[11] Singha, S. et al. (2021). Machine learning for drinking water quality assessment. Chemosphere, 283.  
[12] Ahmed, U. et al. (2019). A systematic review of ML for water quality. Sensors, 19(21).  
[13] Pal, S.C. et al. (2022). SHAP-based explainable AI for groundwater quality. Environmental Science and Pollution Research, 29.  
[14] Kumar, R. et al. (2024). Predicting WQI using stacked ensemble with SHAP. Environmental Research, 240.  
[15] Sagan, V. et al. (2020). Deep learning and traditional ML for IoT-based water quality monitoring. Remote Sensing, 12(12).  
[16] WHO/UNICEF. (2023). Progress on Household Drinking Water, Sanitation and Hygiene 2000–2022. WHO Press.  
[17] WHO. (2022). Guidelines for Drinking-water Quality (4th ed.). WHO Press.  

---

*Manuscript status: DRAFT — experiments pending. Do not submit or share until all result placeholders are replaced with actual experimental values.*

