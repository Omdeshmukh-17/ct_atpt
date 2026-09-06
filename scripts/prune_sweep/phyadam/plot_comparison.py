"""Compare AdamW vs PhyAdam across the pruning-percentage sweep.

Reads both results CSVs (produced incrementally by eval_and_save_metrics.py,
once per optimizer) and produces:
  - comparison_all_metrics.png (2x3 grid: accuracy, precision, recall, f1,
    roc_auc, pr_auc — AdamW vs PhyAdam)
  - comparison_roc_auc.png (single-panel ROC-AUC view with per-point deltas)
  - comparison_table.txt (LaTeX table, bolding the better value per pair)
  - a formatted comparison table printed to stdout with a delta column

Usage:
    python scripts/prune_sweep/phyadam/plot_comparison.py \
        --adamw-csv results/all_prune_metrics.csv \
        --phyadam-csv results/all_phyadam_metrics.csv \
        --plot-dir results/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHYADAM_DIR = Path(__file__).resolve().parent
SWEEP_DIR = PHYADAM_DIR.parent
RESULTS_DIR = SWEEP_DIR / "results"

DEFAULT_ADAMW_CSV = RESULTS_DIR / "all_prune_metrics.csv"
DEFAULT_PHYADAM_CSV = RESULTS_DIR / "all_phyadam_metrics.csv"
DEFAULT_PLOT_DIR = RESULTS_DIR / "plots"

GRID_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
GRID_TITLES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}
STDOUT_METRICS = [
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision", "f1", "roc_auc", "pr_auc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare AdamW vs PhyAdam prune-sweep results.")
    p.add_argument("--adamw-csv", type=Path, default=DEFAULT_ADAMW_CSV)
    p.add_argument("--phyadam-csv", type=Path, default=DEFAULT_PHYADAM_CSV)
    p.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return p.parse_args()


def load(csv_path: Path, label: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{label} results CSV not found at {csv_path}. "
            f"Run the {label} sweep to completion first."
        )
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="prune_percent", keep="last")
    df = df.sort_values("prune_percent").reset_index(drop=True)
    # recall is stored as "recall" in the CSV (== sensitivity); make sure both
    # names are available regardless of which script wrote the row.
    if "sensitivity" not in df.columns and "recall" in df.columns:
        df["sensitivity"] = df["recall"]
    if "recall" not in df.columns and "sensitivity" in df.columns:
        df["recall"] = df["sensitivity"]
    return df


def merge(adamw: pd.DataFrame, phyadam: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        adamw, phyadam, on="prune_percent", suffixes=("_adamw", "_phyadam"), how="inner"
    )
    if merged.empty:
        raise ValueError(
            "No overlapping prune_percent values between the AdamW and PhyAdam "
            "CSVs — make sure both sweeps ran the same set of percentages."
        )
    return merged.sort_values("prune_percent").reset_index(drop=True)


def print_comparison_table(merged: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print("ADAMW vs PHYADAM — FULL METRIC COMPARISON (held-out test set)")
    print("=" * 110)
    for _, row in merged.iterrows():
        pct = row["prune_percent"]
        print(f"\n--- Prune {pct:.0f}% ---")
        header = f"{'metric':<20}{'adamw':>12}{'phyadam':>12}{'delta (phy-adam)':>20}"
        print(header)
        print("-" * len(header))
        for metric in STDOUT_METRICS:
            a = row.get(f"{metric}_adamw")
            b = row.get(f"{metric}_phyadam")
            if a is None or b is None:
                continue
            delta = b - a
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
            print(f"{metric:<20}{a:>12.4f}{b:>12.4f}{delta_str:>20}")
    print("\n" + "=" * 110)


def plot_all_metrics(merged: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, GRID_METRICS):
        ax.plot(
            merged["prune_percent"], merged[f"{metric}_adamw"],
            marker="o", linestyle="-", color="tab:blue", linewidth=2, label="AdamW",
        )
        ax.plot(
            merged["prune_percent"], merged[f"{metric}_phyadam"],
            marker="s", linestyle="--", color="tab:red", linewidth=2, label="PhyAdam",
        )
        ax.set_title(GRID_TITLES[metric])
        ax.set_xlabel("Pruning Percentage (%)")
        ax.set_ylabel(GRID_TITLES[metric])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("CT-ATPT: AdamW vs PhyAdam across Token Pruning Levels", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "comparison_all_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_auc(merged: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        merged["prune_percent"], merged["roc_auc_adamw"],
        marker="o", linestyle="-", color="tab:blue", linewidth=2, label="AdamW",
    )
    ax.plot(
        merged["prune_percent"], merged["roc_auc_phyadam"],
        marker="s", linestyle="--", color="tab:red", linewidth=2, label="PhyAdam",
    )

    for _, row in merged.iterrows():
        delta = row["roc_auc_phyadam"] - row["roc_auc_adamw"]
        y_top = max(row["roc_auc_adamw"], row["roc_auc_phyadam"])
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(row["prune_percent"], y_top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:green" if delta >= 0 else "tab:red",
        )

    ax.set_xlabel("Pruning Percentage (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC: AdamW vs PhyAdam (annotations = PhyAdam - AdamW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "comparison_roc_auc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_latex_table(merged: pd.DataFrame, out_dir: Path) -> Path:
    def bold_better(a: float, b: float) -> tuple[str, str]:
        a_str, b_str = f"{a:.3f}", f"{b:.3f}"
        if a > b:
            return f"\\textbf{{{a_str}}}", b_str
        if b > a:
            return a_str, f"\\textbf{{{b_str}}}"
        return a_str, b_str

    lines = []
    lines.append("% Auto-generated by scripts/prune_sweep/phyadam/plot_comparison.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{AdamW vs. PhyAdam across pruning percentages (held-out test set).}")
    lines.append("\\label{tab:phyadam_comparison}")
    lines.append("\\begin{tabular}{c cc cc cc}")
    lines.append("\\toprule")
    lines.append(
        "Prune \\% & AdamW Acc. & PhyAdam Acc. & AdamW F1 & PhyAdam F1 & AdamW AUC & PhyAdam AUC \\\\"
    )
    lines.append("\\midrule")
    for _, row in merged.iterrows():
        acc_a, acc_p = bold_better(row["accuracy_adamw"], row["accuracy_phyadam"])
        f1_a, f1_p = bold_better(row["f1_adamw"], row["f1_phyadam"])
        auc_a, auc_p = bold_better(row["roc_auc_adamw"], row["roc_auc_phyadam"])
        cells = [f"{row['prune_percent']:.0f}", acc_a, acc_p, f1_a, f1_p, auc_a, auc_p]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    path = out_dir / "comparison_table.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    adamw = load(args.adamw_csv, "AdamW")
    phyadam = load(args.phyadam_csv, "PhyAdam")
    merged = merge(adamw, phyadam)

    print_comparison_table(merged)

    all_metrics_path = plot_all_metrics(merged, args.plot_dir)
    roc_path = plot_roc_auc(merged, args.plot_dir)
    table_path = write_latex_table(merged, args.plot_dir)

    print(f"\nWrote {all_metrics_path}")
    print(f"Wrote {roc_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
