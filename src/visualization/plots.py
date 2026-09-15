"""
src/visualization/plots.py
───────────────────────────
Publication-quality figure generation for all experiment components.

All figures are saved to results/figures/ and results/*/
Every plot is designed to directly answer a research question or
explain the dataset. No decorative visualizations.

Style: seaborn-v0_8-whitegrid with custom color palette.
DPI: 150 (screen-quality); increase to 300 for print submission.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Plots will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


FIGURE_DIR = Path("results/figures/")
SHAP_DIR = Path("results/shap/")
ABLATION_DIR = Path("results/ablation/")
ROBUSTNESS_DIR = Path("results/robustness/")


def _save_figure(fig, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA / Dataset visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series, output_path: str) -> None:
    """Bar chart of potability class distribution."""
    if not MATPLOTLIB_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    colors = ["#E74C3C", "#2ECC71"]
    bars = ax.bar(
        ["Not Potable (0)", "Potable (1)"],
        counts.values,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.5
    )
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / len(y)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=12
        )
    ax.set_title("Target Class Distribution — Water Potability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(fig, output_path)


def plot_feature_distributions(X: pd.DataFrame, output_path: str) -> None:
    """Grid of histograms for each physicochemical feature."""
    if not MATPLOTLIB_AVAILABLE:
        return
    n_features = len(X.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()
    palette = sns.color_palette("viridis", n_features)
    for i, col in enumerate(X.columns):
        ax = axes[i]
        ax.hist(X[col].dropna(), bins=40, color=palette[i], alpha=0.85, edgecolor="white")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions — Water Physicochemical Parameters", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_correlation_matrix(X: pd.DataFrame, output_path: str) -> None:
    """Heatmap of Pearson correlation between all features."""
    if not MATPLOTLIB_AVAILABLE:
        return
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_missing_values(df: pd.DataFrame, output_path: str) -> None:
    """Horizontal bar chart of missing value percentages per feature."""
    if not MATPLOTLIB_AVAILABLE:
        return
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]
    if len(missing_pct) == 0:
        logger.info("No missing values found; skipping missing value plot.")
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(missing_pct) * 0.8)))
    colors = ["#E74C3C" if v > 20 else "#F39C12" if v > 10 else "#3498DB" for v in missing_pct.values]
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="white")
    for bar, pct in zip(bars, missing_pct.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Missing Values (%)", fontsize=12)
    ax.set_title("Missing Values by Feature", fontsize=14, fontweight="bold")
    ax.set_xlim(0, missing_pct.max() * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(fig, output_path)


def plot_boxplots_by_class(X: pd.DataFrame, y: pd.Series, output_path: str) -> None:
    """Box plots showing feature distributions by potability class."""
    if not MATPLOTLIB_AVAILABLE:
        return
    df = X.copy()
    df["Potability"] = y.values
    df_melt = df.melt(id_vars="Potability", var_name="Feature", value_name="Value")
    df_melt["Class"] = df_melt["Potability"].map({0: "Not Potable", 1: "Potable"})
    n_features = len(X.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()
    for i, feature in enumerate(X.columns):
        ax = axes[i]
        feature_data = df_melt[df_melt["Feature"] == feature]
        sns.boxplot(
            data=feature_data, x="Class", y="Value",
            palette={"Not Potable": "#E74C3C", "Potable": "#2ECC71"},
            ax=ax, width=0.5
        )
        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Potability Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model performance visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    metric: str = "f1_macro",
    output_path: str = "results/figures/model_comparison.png",
) -> None:
    """Horizontal bar chart comparing models on a given metric."""
    if not MATPLOTLIB_AVAILABLE:
        return
    df = metrics_df.sort_values(metric, ascending=True)
    palette = sns.color_palette("viridis", len(df))
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.7)))
    bars = ax.barh(df["model"], df[metric], color=palette, edgecolor="white", height=0.6)
    for bar, val in zip(bars, df[metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, min(1.0, df[metric].max() * 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Annotated confusion matrix heatmap."""
    if not MATPLOTLIB_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Not Potable", "Potable"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.7}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str,
) -> None:
    """
    Overlay ROC curves for all models.

    roc_data: dict mapping model_name → (fpr, tpr, auc_value)
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("tab10", len(roc_data))
    for i, (model_name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})",
                color=palette[i], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    model_name: str,
    output_path: str,
    max_display: int = 9,
) -> None:
    """SHAP beeswarm summary plot (global explanation)."""
    if not (MATPLOTLIB_AVAILABLE and SHAP_AVAILABLE):
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary Plot — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_shap_bar(
    importance_df: pd.DataFrame,
    model_name: str,
    output_path: str,
) -> None:
    """Horizontal bar chart of mean absolute SHAP values (global importance)."""
    if not MATPLOTLIB_AVAILABLE:
        return
    df = importance_df.sort_values("mean_abs_shap")
    palette = sns.color_palette("viridis", len(df))
    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.7)))
    bars = ax.barh(df["feature"], df["mean_abs_shap"], color=palette, edgecolor="white")
    for bar, val in zip(bars, df["mean_abs_shap"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Global Feature Importance (SHAP) — {model_name}", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_shap_waterfall(
    explainer: Any,
    X_row: pd.Series,
    model_name: str,
    example_label: str,
    output_path: str,
) -> None:
    """SHAP waterfall plot for a single sample."""
    if not (MATPLOTLIB_AVAILABLE and SHAP_AVAILABLE):
        return
    try:
        X_arr = X_row.values.reshape(1, -1)
        shap_exp = explainer(pd.DataFrame(X_arr, columns=X_row.index))
        if hasattr(shap_exp, "__getitem__"):
            shap_exp = shap_exp[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_exp, show=False)
        plt.title(f"SHAP Waterfall — {model_name} [{example_label}]", fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save_figure(fig, output_path)
    except Exception as e:
        logger.warning(f"Waterfall plot failed for {model_name}/{example_label}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_curves(
    ablation_df: pd.DataFrame,
    metrics: List[str] = ["f1_macro", "roc_auc"],
    output_path: str = "results/figures/ablation_curve.png",
) -> None:
    """
    Line plots: Number of features vs. performance metric.
    One line per model, one subplot per metric.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    models = [m for m in ablation_df["model"].unique() if m != "Dummy"]
    palette = sns.color_palette("tab10", len(models))
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for i, model_name in enumerate(models):
            subset = ablation_df[ablation_df["model"] == model_name].sort_values("n_features")
            ax.plot(
                subset["n_features"], subset[metric],
                marker="o", label=model_name,
                color=palette[i], linewidth=2, markersize=6
            )
        ax.set_xlabel("Number of Features", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Feature Count vs. {metric.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Feature Ablation Experiment", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness_curves(
    robustness_df: pd.DataFrame,
    metrics: List[str] = ["f1_macro_mean", "roc_auc_mean"],
    output_path: str = "results/figures/robustness_curves.png",
) -> None:
    """
    Line plots: Noise level vs. performance metric.
    One line per model, with ± std shading.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    models = [m for m in robustness_df["model"].unique() if m != "Dummy"]
    palette = sns.color_palette("tab10", len(models))
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        std_col = metric.replace("_mean", "_std")
        display_name = metric.replace("_mean", "").replace("_", " ").title()
        for i, model_name in enumerate(models):
            subset = robustness_df[robustness_df["model"] == model_name].sort_values("noise_level")
            ax.plot(
                subset["noise_level"] * 100, subset[metric],
                marker="o", label=model_name,
                color=palette[i], linewidth=2, markersize=6
            )
            if std_col in subset.columns:
                ax.fill_between(
                    subset["noise_level"] * 100,
                    subset[metric] - subset[std_col],
                    subset[metric] + subset[std_col],
                    alpha=0.12, color=palette[i]
                )
        ax.set_xlabel("Noise Level (%)", fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        ax.set_title(f"Noise Level vs. {display_name}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Robustness to Measurement Noise", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, output_path)
