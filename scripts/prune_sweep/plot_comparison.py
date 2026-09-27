"""Compare ALL optimizers across the pruning-percentage sweep (N-generic).

Unlike scripts/prune_sweep/phyadam/plot_comparison.py (which is hardcoded to
the AdamW-vs-PhyAdam pair and is left untouched so its existing outputs never
change), this script auto-discovers every optimizer that has logged results
and overlays all of them on the same plots. Adding a future optimizer to the
comparison requires no code change here — it only needs to write its metrics
to results/all_<name>_metrics.csv via scripts/prune_sweep/eval_and_save_metrics.py
(--optimizer-name <name> --csv-path results/all_<name>_metrics.csv), the same
convention every existing sweep (adamw, phyadam, lion, sam_adamw, ...) already
follows.

Produces:
  - comparison_all_metrics.png (2x3 grid: accuracy, precision, recall, f1,
    roc_auc, pr_auc — one line per optimizer)
  - comparison_roc_auc.png (single-panel ROC-AUC view, one line per optimizer)
  - comparison_table.txt (LaTeX table, bolding the best value per pruning row)
  - a formatted comparison table printed to stdout

Usage:
    python scripts/prune_sweep/plot_comparison.py
    # or name specific CSVs explicitly:
    python scripts/prune_sweep/plot_comparison.py \
        --csv results/all_prune_metrics.csv=AdamW \
        --csv results/all_phyadam_metrics.csv=PhyAdam \
        --csv results/all_lion_metrics.csv=Lion \
        --csv results/all_sam_adamw_metrics.csv=SAM+AdamW
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SWEEP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SWEEP_DIR / "results"
DEFAULT_PLOT_DIR = RESULTS_DIR / "plots"

# Known CSV -> display-name mapping for the optimizers this repo ships sweep
# infrastructure for; any other results/all_*_metrics.csv is picked up too
# (labeled from its filename) so a new optimizer needs no edit here.
KNOWN_CSVS = {
    RESULTS_DIR / "all_prune_metrics.csv": "AdamW",
    RESULTS_DIR / "all_phyadam_metrics.csv": "PhyAdam",
    RESULTS_DIR / "all_lion_metrics.csv": "Lion",
    RESULTS_DIR / "all_sam_adamw_metrics.csv": "SAM+AdamW",
}

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
# Cycled by index so any number of optimizers gets a distinct marker/style.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown", "tab:pink", "tab:gray"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all optimizers' prune-sweep results.")
    p.add_argument(
        "--csv", action="append", default=None, metavar="PATH=LABEL",
        help="Explicit CSV=Label pair to include (repeatable). If omitted, "
             "auto-discovers every results/all_*_metrics.csv.",
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return p.parse_args()


def discover_csvs(results_dir: Path) -> dict[Path, str]:
    found: dict[Path, str] = {}
    for path in sorted(results_dir.glob("all_*_metrics.csv")):
        label = KNOWN_CSVS.get(path)
        if label is None:
            # Derive a display label from "all_<name>_metrics.csv".
            stem = path.stem  # all_<name>_metrics
            name = stem[len("all_"):-len("_metrics")] if stem.startswith("all_") and stem.endswith("_metrics") else stem
            label = name.replace("_", " ").title()
        found[path] = label
    return found


def load(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="prune_percent", keep="last")
    df = df.sort_values("prune_percent").reset_index(drop=True)
    if "sensitivity" not in df.columns and "recall" in df.columns:
        df["sensitivity"] = df["recall"]
    if "recall" not in df.columns and "sensitivity" in df.columns:
        df["recall"] = df["sensitivity"]
    df["_optimizer_label"] = label
    return df


def print_comparison_table(frames: dict[str, pd.DataFrame]) -> None:
    all_percents = sorted(set().union(*[set(df["prune_percent"]) for df in frames.values()]))
    print("\n" + "=" * 120)
    print("OPTIMIZER COMPARISON — FULL METRIC TABLE (held-out test set)")
    print("=" * 120)
    for pct in all_percents:
        print(f"\n--- Prune {pct:.0f}% ---")
        header = f"{'metric':<20}" + "".join(f"{label:>16}" for label in frames)
        print(header)
        print("-" * len(header))
        for metric in STDOUT_METRICS:
            cells = []
            any_value = False
            for label, df in frames.items():
                row = df[df["prune_percent"] == pct]
                if row.empty or metric not in row.columns:
                    cells.append("—")
                else:
                    cells.append(f"{row.iloc[0][metric]:.4f}")
                    any_value = True
            if any_value:
                print(f"{metric:<20}" + "".join(f"{c:>16}" for c in cells))
    print("\n" + "=" * 120)


def plot_all_metrics(frames: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, GRID_METRICS):
        for i, (label, df) in enumerate(frames.items()):
            if metric not in df.columns:
                continue
            ax.plot(
                df["prune_percent"], df[metric],
                marker=MARKERS[i % len(MARKERS)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                color=COLORS[i % len(COLORS)], linewidth=2, label=label,
            )
        ax.set_title(GRID_TITLES[metric])
        ax.set_xlabel("Pruning Percentage (%)")
        ax.set_ylabel(GRID_TITLES[metric])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("CT-ATPT: Optimizer Comparison across Token Pruning Levels", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "comparison_all_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_auc(frames: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, df) in enumerate(frames.items()):
        ax.plot(
            df["prune_percent"], df["roc_auc"],
            marker=MARKERS[i % len(MARKERS)], linestyle=LINESTYLES[i % len(LINESTYLES)],
            color=COLORS[i % len(COLORS)], linewidth=2, label=label,
        )
    ax.set_xlabel("Pruning Percentage (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC vs. Pruning Percentage — All Optimizers")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "comparison_roc_auc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_latex_table(frames: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    all_percents = sorted(set().union(*[set(df["prune_percent"]) for df in frames.values()]))
    labels = list(frames.keys())

    lines = []
    lines.append("% Auto-generated by scripts/prune_sweep/plot_comparison.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Optimizer comparison across pruning percentages (held-out test set). Best ROC-AUC per row in bold.}")
    lines.append("\\label{tab:optimizer_comparison}")
    ncols = 1 + len(labels)
    lines.append("\\begin{tabular}{" + "c" * ncols + "}")
    lines.append("\\toprule")
    lines.append("Prune \\% & " + " & ".join(labels) + " \\\\")
    lines.append("\\midrule")
    for pct in all_percents:
        values: dict[str, float | None] = {}
        for label, df in frames.items():
            row = df[df["prune_percent"] == pct]
            values[label] = float(row.iloc[0]["roc_auc"]) if not row.empty else None
        best = max((v for v in values.values() if v is not None), default=None)
        cells = [f"{pct:.0f}"]
        for label in labels:
            v = values[label]
            if v is None:
                cells.append("—")
            elif best is not None and v == best:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
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

    frames: dict[str, pd.DataFrame] = {}
    if args.csv:
        for spec in args.csv:
            if "=" not in spec:
                raise ValueError(f"--csv expects PATH=LABEL, got {spec!r}")
            path_str, label = spec.split("=", 1)
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"{label} results CSV not found at {path}.")
            frames[label] = load(path, label)
    else:
        discovered = discover_csvs(args.results_dir)
        if not discovered:
            raise FileNotFoundError(
                f"No results/all_*_metrics.csv files found under {args.results_dir}. "
                "Run at least one sweep to completion first (e.g. "
                "scripts/prune_sweep/run_full_sweep.py for AdamW)."
            )
        for path, label in discovered.items():
            frames[label] = load(path, label)

    print(f"Comparing {len(frames)} optimizer(s): {', '.join(frames)}")

    print_comparison_table(frames)

    all_metrics_path = plot_all_metrics(frames, args.plot_dir)
    roc_path = plot_roc_auc(frames, args.plot_dir)
    table_path = write_latex_table(frames, args.plot_dir)

    print(f"\nWrote {all_metrics_path}")
    print(f"Wrote {roc_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
