# Research Gap Analysis

**Project:** Explainable and Robust Machine Learning for Water Quality Assessment and Reduced-Parameter Monitoring  
**Date:** September 2026

---

## 1. Summary of What Existing Work Has Established

Based on the literature review (see `literature_review.md`), the following is well-established in the field:

### 1.1 Classification performance of individual models
A substantial body of work (LR-01, LR-02, LR-04, LR-08, LR-10) has benchmarked machine learning classifiers for water quality assessment. The consensus finding is that ensemble methods — particularly Random Forest and gradient-boosted trees (XGBoost, LightGBM, CatBoost) — outperform single classifiers (Logistic Regression, SVM, single Decision Trees) on water quality classification tasks when the dataset is tabular and moderate in size (< 10,000 samples).

### 1.2 SHAP-based explainability
SHAP has been applied to water quality prediction in several recent studies (LR-03, LR-05, LR-09, LR-11). These studies consistently demonstrate that SHAP provides actionable global feature rankings and useful local explanations. The most frequently identified important features across studies include pH, conductivity/TDS, dissolved oxygen, and turbidity — though the specific ranking depends on the geographic context, target variable, and the dataset used.

### 1.3 Feature selection for cost reduction
One dedicated study (LR-06) showed that for river water quality index (WQI) regression in South Korea, a 5-feature subset from 14 features retained > 95% of predictive performance. This is the most relevant direct precedent for the reduced-parameter research question.

### 1.4 Robustness to sensor noise (non-water domains)
General robustness studies (LR-07, LR-12) in environmental and air quality domains have shown that tree-based ensemble models degrade more gracefully than SVM and MLP under controlled Gaussian noise injection. These studies use noise levels of 0–20% perturbation and evaluate F1 or accuracy at each level.

---

## 2. What Existing Work Has NOT Adequately Addressed

### Gap 1 — Simultaneous XAI + Feature Ablation + Noise Robustness on the Same Dataset

**Observation:** No single study in the reviewed literature combines all three of the following on the same dataset and experimental setting:
1. SHAP-based global and local explainability
2. A systematic feature-ablation study (performance vs. number of parameters)
3. A controlled noise-robustness experiment (performance vs. noise level)

Studies that address explainability (LR-03, LR-05, LR-09) do not include feature ablation or noise experiments. The feature-ablation study (LR-06) does not include SHAP or noise testing. The robustness studies (LR-07, LR-12) are not in the water quality domain and do not include SHAP.

**This project's contribution:** This study performs all three experimental components on the same dataset and model set, enabling direct comparison of: which features SHAP identifies as important, whether those features are sufficient to sustain performance in the ablation experiment, and whether those reduced-feature models are comparably noise-robust.

### Gap 2 — Cross-Model SHAP Consistency for Drinking Water Classification

**Observation:** Cross-model SHAP comparison has been performed for regression-based WQI prediction (LR-11), but not systematically for the binary potability classification task using the widely accessible Kaggle Water Potability dataset.

**This project's contribution:** This study investigates whether the SHAP-ranked feature importance ordering is consistent across different model families (tree-based, kernel-based, linear, neural network) for the binary potability task. Consistency or inconsistency here has practical implications: if important features are model-agnostic, they represent robust candidates for reduced-sensor deployment.

### Gap 3 — Noise Robustness Specifically in the Drinking Water Potability Classification Setting

**Observation:** Existing robustness studies (LR-07, LR-12) are conducted on air quality or general environmental data. No study appears to have systematically evaluated noise-level effects on binary water potability classification using physicochemical features. This is an important gap because the features in drinking water assessment (pH, chloramines, turbidity) have different physical noise profiles than air quality parameters.

**This project's contribution:** Controlled multiplicative noise injection (simulating proportional sensor uncertainty) is applied to the water potability feature space, providing the first systematic robustness evaluation of ML classifiers specifically for this task.

### Gap 4 — Reproducible Open-Access Benchmark for the Potability Dataset

**Observation:** The Kaggle Water Potability dataset (LR-16) is widely used for ML tutorials and demonstrations, but very few peer-reviewed studies provide a fully reproducible, methodologically rigorous benchmark with:
- Stratified train/validation/test splits
- Cross-validation with reported mean ± SD
- SHAP explainability
- Hyperparameter optimization with reported search spaces
- Feature-ablation curves
- Noise robustness evaluation

Most published studies using this dataset either lack cross-validation, omit SHAP, or do not perform feature ablation or robustness experiments.

**This project's contribution:** This study provides a reproducible, open-source benchmark that addresses all of the above, using documented preprocessing, fixed random seeds, and published code.

---

## 3. What This Project Does NOT Claim as Novel

The following aspects are explicitly acknowledged as prior art:

- **Use of SHAP for water quality explainability** — extensively established by LR-03, LR-05, LR-09.
- **Application of Random Forest and XGBoost to water quality classification** — thoroughly established by LR-01, LR-02.
- **Feature selection for water quality monitoring cost reduction** — established by LR-06 (though not for potability classification specifically).
- **Noise robustness of tree-based ensemble models** — established by LR-07 and LR-12 (though not for water potability specifically).

---

## 4. Positioning Statement for the Manuscript

The manuscript should position this work as:

> *"A reproducible, integrated experimental study that combines model benchmarking, SHAP-based explainability, feature ablation, and noise-robustness analysis within a single coherent experimental framework for drinking water potability classification. While each of these components has been addressed individually in prior work, their integrated evaluation on a common publicly available dataset — enabling cross-component inference — has not been previously documented in the peer-reviewed literature."*

This framing avoids overclaiming novelty while accurately describing what differentiates this study from prior work.

---

## 5. Limitations of the Research Gap Assessment

- The literature search may not have captured all published work; the field is active and review databases may have indexing delays.
- Non-English publications were not systematically reviewed.
- Gray literature (conference papers, preprints) was considered selectively.
- The gap assessment is based on the 18 papers in the formal literature review; additional work may exist that addresses one or more of these gaps.

---

*End of Research Gap Analysis*
