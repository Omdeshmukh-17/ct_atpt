"""Visualize the Bayesian per-stage pruning search.

Produces:
  1. Optuna's own diagnostic plots, saved as PNGs (requires the `kaleido`
     package to rasterize Plotly figures — pip install kaleido):
       - optimization_history.png (accuracy over trials)
       - param_importances.png (which stage matters most)
       - contour.png (2D contour of stage1 vs stage2, colored by accuracy)
       - slice.png (each stage's keep fraction vs accuracy)
  2. A bar chart comparing the best uniform-pruning accuracy (from the
     existing results/all_prune_metrics.csv sweep) against the best
     non-uniform (Bayesian search) accuracy.

All saved to results/plots/bayesian/.

Usage:
    python scripts/plot_bayesian_results.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = PROJECT_ROOT / "scripts" / "prune_sweep"
DEFAULT_STUDY_DB = PROJECT_ROOT / "results" / "optuna_prune_search.db"
DEFAULT_UNIFORM_CSV = SWEEP_DIR / "results" / "all_prune_metrics.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "plots" / "bayesian"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the Bayesian per-stage pruning search.")
    p.add_argument("--study-db", type=Path, default=DEFAULT_STUDY_DB)
    p.add_argument("--uniform-csv", type=Path, default=DEFAULT_UNIFORM_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--metric", default="accuracy", help="Must match the metric bayesian_prune_search.py optimized.")
    return p.parse_args()


def best_uniform_accuracy(uniform_csv: Path, metric: str) -> tuple[float | None, float | None]:
    """Returns (best_accuracy, prune_percent_at_best) from the uniform sweep, or (None, None)."""
    if not uniform_csv.exists():
        return None, None
    best_value, best_pct = None, None
    with open(uniform_csv) as f:
        for row in csv.DictReader(f):
            if row.get("optimizer") not in (None, "", "adamw"):
                continue
            if metric not in row or row[metric] in (None, ""):
                continue
            v = float(row[metric])
            if best_value is None or v > best_value:
                best_value, best_pct = v, float(row["prune_percent"])
    return best_value, best_pct


def save_optuna_plots(study: optuna.Study, out_dir: Path) -> list[Path]:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_slice,
    )

    written: list[Path] = []
    figs = {
        "optimization_history.png": plot_optimization_history(study),
        "param_importances.png": plot_param_importances(study),
        "contour.png": plot_contour(study, params=["stage1_keep", "stage2_keep"]),
        "slice.png": plot_slice(study),
    }
    try:
        for name, fig in figs.items():
            path = out_dir / name
            fig.write_image(str(path))
            written.append(path)
            print(f"Wrote {path}")
    except ValueError as e:
        # write_image needs the `kaleido` package; fail loudly with the fix
        # rather than silently skipping these plots.
        raise RuntimeError(
            "Could not rasterize Optuna's Plotly figures — install kaleido "
            "(`pip install kaleido`) and re-run this script."
        ) from e
    return written


def plot_uniform_vs_bayesian(best_uniform: float | None, best_bayesian: float | None, metric: str, out_dir: Path) -> Path | None:
    if best_uniform is None or best_bayesian is None:
        print("Skipping uniform-vs-Bayesian bar chart: missing one of the two comparison points "
              f"(uniform={best_uniform}, bayesian={best_bayesian}).")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(
        ["Best uniform\npruning", "Best non-uniform\n(Bayesian search)"],
        [best_uniform, best_bayesian],
        color=["tab:gray", "tab:blue"],
    )
    for bar, value in zip(bars, [best_uniform, best_bayesian]):
        ax.annotate(f"{value:.4f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    delta = best_bayesian - best_uniform
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Uniform vs. Non-uniform Per-Stage Pruning\n(delta = {delta:+.4f})")
    ax.set_ylim(0, max(best_uniform, best_bayesian) * 1.15)
    fig.tight_layout()
    path = out_dir / "uniform_vs_bayesian.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.study_db.exists():
        raise FileNotFoundError(f"No study database at {args.study_db} — run scripts/bayesian_prune_search.py first.")
    study = optuna.load_study(study_name="ct_atpt_prune_search", storage=f"sqlite:///{args.study_db}")
    print(f"Loaded study with {len(study.trials)} trial(s). Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    save_optuna_plots(study, args.out_dir)

    best_uniform, best_uniform_pct = best_uniform_accuracy(args.uniform_csv, args.metric)
    if best_uniform is not None:
        print(f"Best uniform-sweep {args.metric}: {best_uniform:.4f} (at {best_uniform_pct:.0f}% pruned)")
    plot_uniform_vs_bayesian(best_uniform, study.best_value, args.metric, args.out_dir)

    print(f"\nAll plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
