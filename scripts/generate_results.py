"""
scripts/generate_results.py
─────────────────────────────
Aggregate all experiment results into publication-quality tables
and generate remaining EDA figures.

Produces:
- Table 1: Dataset Characteristics
- Table 2: Model Performance (test set)
- Table 3: Cross-Validation Results
- Table 4: Top SHAP Features
- Table 5: Feature Ablation Results
- Table 6: Noise Robustness Results
- EDA figures (class distribution, correlations, etc.)

Usage:
    python scripts/generate_results.py
    python scripts/generate_results.py --config configs/config.yaml

ALL numbers in tables come from saved experiment files.
No values are manually typed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_config, load_raw_dataset, validate_dataset
from src.visualization.plots import (
    plot_class_distribution, plot_feature_distributions,
    plot_correlation_matrix, plot_missing_values, plot_boxplots_by_class,
    plot_model_comparison_bar, plot_confusion_matrix, plot_roc_curves
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("generate_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate all result tables and figures.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def generate_table1_dataset(df: pd.DataFrame, cfg: dict, tables_dir: Path) -> None:
    """Table 1: Dataset characteristics."""
    target_col = cfg["data"]["target_column"]
    features = [c for c in df.columns if c != target_col]
    class_counts = df[target_col].value_counts()
    stats = {
        "Statistic": [
            "Total samples", "Features",
            "Samples – Not Potable (class 0)", "Samples – Potable (class 1)",
            "Class ratio (0:1)",
            "Features with missing values",
            "Total missing values",
            "Missing value rate (%)"
        ],
        "Value": [
            len(df),
            len(features),
            int(class_counts.get(0, 0)),
            int(class_counts.get(1, 0)),
            f"{class_counts.get(0, 0) / max(1, class_counts.get(1, 0)):.2f}:1",
            int((df[features].isnull().sum() > 0).sum()),
            int(df[features].isnull().sum().sum()),
            f"{100 * df[features].isnull().sum().sum() / (len(df) * len(features)):.2f}"
        ]
    }
    table1 = pd.DataFrame(stats)
    table1.to_csv(tables_dir / "table1_dataset.csv", index=False)
    logger.info("Table 1 saved.")


def generate_table2_performance(metrics_dir: Path, tables_dir: Path) -> None:
    """Table 2: Test set model performance."""
    metrics_file = metrics_dir / "all_test_metrics.csv"
    if not metrics_file.exists():
        logger.warning("Test metrics not found. Run train_models.py first.")
        return
    df = pd.read_csv(metrics_file)
    cols = ["model", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
            "precision_class1", "recall_class1", "roc_auc", "pr_auc"]
    available = [c for c in cols if c in df.columns]
    table2 = df[available].round(4)
    table2 = table2.sort_values("f1_macro", ascending=False)
    table2.to_csv(tables_dir / "table2_model_performance.csv", index=False)
    logger.info("Table 2 saved.")


def generate_table3_cv(metrics_dir: Path, tables_dir: Path) -> None:
    """Table 3: Cross-validation results."""
    cv_file = metrics_dir / "all_cv_results.csv"
    if not cv_file.exists():
        logger.warning("CV results not found. Run train_models.py without --skip-cv.")
        return
    df = pd.read_csv(cv_file)
    # Create formatted mean ± std columns
    table3_rows = []
    for _, row in df.iterrows():
        r = {"model": row["model"]}
        for m in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "average_precision"]:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in row and std_col in row:
                r[m] = f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
        table3_rows.append(r)
    table3 = pd.DataFrame(table3_rows)
    table3 = table3.sort_values("f1_macro", ascending=False) if "f1_macro" in table3.columns else table3
    table3.to_csv(tables_dir / "table3_cv_results.csv", index=False)
    logger.info("Table 3 saved.")


def generate_table4_shap(shap_dir: Path, tables_dir: Path) -> None:
    """Table 4: Global SHAP feature importance."""
    importance_files = list(shap_dir.glob("global_importance_*.csv"))
    if not importance_files:
        logger.warning("SHAP importance files not found. Run run_explainability.py first.")
        return
    all_importance = []
    for f in importance_files:
        model_name = f.stem.replace("global_importance_", "")
        df = pd.read_csv(f)
        df["model"] = model_name
        all_importance.append(df)
    table4 = pd.concat(all_importance, ignore_index=True)
    table4 = table4[["model", "rank", "feature", "mean_abs_shap"]]
    table4.to_csv(tables_dir / "table4_shap_importance.csv", index=False)
    logger.info("Table 4 saved.")


def generate_table5_ablation(ablation_dir: Path, tables_dir: Path) -> None:
    """Table 5: Feature ablation results."""
    ablation_file = ablation_dir / "ablation_results_all.csv"
    if not ablation_file.exists():
        logger.warning("Ablation results not found. Run run_ablation.py first.")
        return
    df = pd.read_csv(ablation_file)
    cols = ["model", "ablation_config", "n_features", "accuracy", "f1_macro", "roc_auc"]
    available = [c for c in cols if c in df.columns]
    table5 = df[available].round(4)
    table5 = table5.sort_values(["model", "n_features"])
    table5.to_csv(tables_dir / "table5_ablation_results.csv", index=False)
    logger.info("Table 5 saved.")


def generate_table6_robustness(robustness_dir: Path, tables_dir: Path) -> None:
    """Table 6: Noise robustness results."""
    robustness_file = robustness_dir / "robustness_results_all.csv"
    if not robustness_file.exists():
        logger.warning("Robustness results not found. Run run_robustness.py first.")
        return
    df = pd.read_csv(robustness_file)
    cols = ["model", "noise_level", "f1_macro_mean", "f1_macro_std", "roc_auc_mean", "roc_auc_std"]
    available = [c for c in cols if c in df.columns]
    df["noise_level_pct"] = (df["noise_level"] * 100).round(0).astype(int).astype(str) + "%"
    table6 = df[available].round(4)
    table6 = table6.sort_values(["model", "noise_level"])
    table6.to_csv(tables_dir / "table6_robustness_results.csv", index=False)
    logger.info("Table 6 saved.")


def generate_eda_figures(df: pd.DataFrame, cfg: dict, figs_dir: Path) -> None:
    """Generate all EDA visualizations."""
    target_col = cfg["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info("Generating EDA figures...")
    plot_class_distribution(y, str(figs_dir / "eda_class_distribution.png"))
    plot_feature_distributions(X, str(figs_dir / "eda_feature_distributions.png"))
    plot_correlation_matrix(X, str(figs_dir / "eda_correlation_matrix.png"))
    plot_missing_values(df, str(figs_dir / "eda_missing_values.png"))
    plot_boxplots_by_class(X, y, str(figs_dir / "eda_boxplots_by_class.png"))
    logger.info("EDA figures saved.")


def generate_model_comparison_figures(metrics_dir: Path, figs_dir: Path) -> None:
    """Generate model comparison bar charts and ROC curves."""
    metrics_file = metrics_dir / "all_test_metrics.csv"
    if not metrics_file.exists():
        return
    df = pd.read_csv(metrics_file)
    for metric in ["f1_macro", "roc_auc", "accuracy"]:
        if metric in df.columns:
            plot_model_comparison_bar(
                df, metric=metric,
                output_path=str(figs_dir / f"model_comparison_{metric}.png")
            )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tables_dir = Path(cfg["results"]["tables"])
    figs_dir = Path(cfg["results"]["figures"])
    metrics_dir = Path(cfg["results"]["metrics"])
    shap_dir = Path(cfg["results"]["shap"])
    ablation_dir = Path(cfg["results"]["ablation"])
    robustness_dir = Path(cfg["results"]["robustness"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load raw dataset for EDA
    try:
        df = load_raw_dataset(cfg["data"]["primary_dataset"])
        validate_dataset(df)
        generate_table1_dataset(df, cfg, tables_dir)
        generate_eda_figures(df, cfg, figs_dir)
    except FileNotFoundError as e:
        logger.warning(f"Dataset not found: {e}\nSkipping EDA and Table 1.")

    generate_table2_performance(metrics_dir, tables_dir)
    generate_table3_cv(metrics_dir, tables_dir)
    generate_table4_shap(shap_dir, tables_dir)
    generate_table5_ablation(ablation_dir, tables_dir)
    generate_table6_robustness(robustness_dir, tables_dir)
    generate_model_comparison_figures(metrics_dir, figs_dir)

    logger.info(f"\n═══ Results Generation Complete ═══")
    logger.info(f"Tables saved to: {tables_dir}")
    logger.info(f"Figures saved to: {figs_dir}")


if __name__ == "__main__":
    main()
