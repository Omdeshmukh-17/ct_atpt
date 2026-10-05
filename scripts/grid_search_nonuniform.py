"""Manual grid search over a handful of hand-picked NON-uniform per-stage pruning configs.

Separate from the Bayesian search. For each config it runs the SAME training recipe as
the uniform pruning sweep (scripts/prune_sweep/run_full_sweep.py: same epochs, batch size,
AdamW lr, label smoothing, drop-path, augmentation flag, seed, split, pruning warmup/ramp)
through scripts/train_ct_atpt_ddp.py, then scores the best checkpoint on test.csv with the
same scripts/prune_sweep/eval_and_save_metrics.py. The ONLY thing that changes between runs
is the per-stage keep fractions (--prune-keep-stage1/2/3).

The keep fractions are CUMULATIVE fractions of the original 432 tokens that survive each
stage (after the 4th, 7th and 10th blocks). So (0.7, 0.7, 0.7) prunes 30% at stage 1 and
nothing at stages 2-3, and it is NOT the same run as the uniform 20% sweep point, which
compounds to roughly 0.93 / 0.86 / 0.80. Pass --include-uniform-baseline to also run that
original uniform 20% setup (via --prune-target-keep 0.8) under identical conditions.
The model's keep floors (at least 25% of tokens per stage, at most 50% pruned per stage)
still apply, so (0.9, 0.5, 0.1) ends near 25% kept, not 10%.

Outputs (under results/):
  grid_search_nonuniform_metrics.csv           one row per config (all eval columns + stage keeps)
  grid_search_nonuniform/<config>/eval_row.csv raw eval row for that config (used to skip finished ones)
  grid_search_nonuniform/plots/<config>/prune<N>.png   ROC curve (same style as prune10.png)
  grid_search_nonuniform/scores/<config>/prune<N>.npz  raw labels/scores
  plots/grid_search/accuracy_comparison.png, roc_auc_comparison.png

Usage:
    python scripts/grid_search_nonuniform.py                  # all 4 configs, one after another
    python scripts/grid_search_nonuniform.py --configs 1 3    # only some (resume across sessions)
    python scripts/grid_search_nonuniform.py --dry-run        # print the commands only
    python scripts/grid_search_nonuniform.py --plots-only     # rebuild CSV + plots from finished configs
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "scripts" / "prune_sweep" / "splits_tvt"
RESULTS_ROOT = PROJECT_ROOT / "results"
GRID_DIR = RESULTS_ROOT / "grid_search_nonuniform"
METRICS_CSV = RESULTS_ROOT / "grid_search_nonuniform_metrics.csv"
PLOTS_DIR = RESULTS_ROOT / "plots" / "grid_search"
RUNS_DIR = PROJECT_ROOT / "runs" / "grid_search_nonuniform"

TRAIN_SCRIPT = "scripts/train_ct_atpt_ddp.py"
EVAL_SCRIPT = "scripts/prune_sweep/eval_and_save_metrics.py"

# (name, (stage1, stage2, stage3), description)
CONFIGS = [
    ("cfg1_gradual", (0.9, 0.7, 0.5), "gradual pruning"),
    ("cfg2_sharp_dropoff", (0.9, 0.5, 0.1), "sharp drop-off"),
    ("cfg3_uniform_moderate", (0.7, 0.7, 0.7), "uniform moderate"),
    ("cfg4_uniform_20pct", (0.8, 0.8, 0.8), "uniform 20% prune (per-stage form)"),
]
# Optional reference: the ORIGINAL uniform 20% sweep setup (single --prune-target-keep, compounded over stages).
BASELINE = ("ref_uniform20_compounded", None, "original uniform 20% sweep setup")
BASELINE_TARGET_KEEP = 0.8

EXTRA_FIELDS = ["config_name", "stage1_keep", "stage2_keep", "stage3_keep"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual grid search over non-uniform per-stage pruning configs.")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                   help="Directory with train.csv / val.csv / test.csv (same split as the uniform sweep).")
    p.add_argument("--configs", type=int, nargs="*", default=None,
                   help="Which configs to run, by number 1-4 (default: all).")
    p.add_argument("--include-uniform-baseline", action="store_true",
                   help="Also run the original uniform 20%% sweep setup as a reference.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands and exit.")
    p.add_argument("--plots-only", action="store_true", help="Skip training; rebuild the CSV and plots.")
    p.add_argument("--force", action="store_true", help="Re-run configs that already have results.")
    p.add_argument("--backup-dir", type=Path, default=None,
                   help="Optional persistent dir (e.g. on Drive); results are copied here after each config.")
    p.add_argument("--python", type=str, default=sys.executable)
    # Everything below mirrors run_full_sweep.py's defaults EXACTLY. Do not change them.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--pruning-warmup-epochs", type=int, default=10)
    p.add_argument("--prune-ramp-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--depth", type=int, default=96)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--patch-z", type=int, default=8)
    p.add_argument("--patch-y", type=int, default=16)
    p.add_argument("--patch-x", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--transformer-depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    return p.parse_args()


# ── command construction (flag order and values copied from run_full_sweep.py) ──────────
def train_flags(args: argparse.Namespace, splits_rel: str, run_dir: str,
                stages: tuple[float, float, float] | None) -> list[str]:
    if stages is None:
        keep_flags = ["--prune-target-keep", str(BASELINE_TARGET_KEEP)]
    else:
        keep_flags = ["--prune-keep-stage1", str(stages[0]),
                      "--prune-keep-stage2", str(stages[1]),
                      "--prune-keep-stage3", str(stages[2])]
    return [
        "--train-manifest", f"{splits_rel}/train.csv",
        "--val-manifest", f"{splits_rel}/val.csv",
        "--output-dir", run_dir,
        "--pretrained",
        "--optimizer", "adamw",
        "--embed-dim", str(args.embed_dim),
        "--transformer-depth", str(args.transformer_depth),
        "--heads", str(args.heads),
        "--depth", str(args.depth),
        "--height", str(args.height),
        "--width", str(args.width),
        "--patch-z", str(args.patch_z),
        "--patch-y", str(args.patch_y),
        "--patch-x", str(args.patch_x),
        "--pruning-mode", "adaptive",
        *keep_flags,
        "--pruning-warmup-epochs", str(args.pruning_warmup_epochs),
        "--prune-ramp-epochs", str(args.prune_ramp_epochs),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--cls-loss", "ce",
        "--label-smoothing", str(args.label_smoothing),
        "--drop-path", str(args.drop_path),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--det-weight", "0",
        "--amp", args.amp,
        "--tta",
        "--augment",
        "--best-metric", "roc_auc",
    ]


def effective_prune_percent(stages: tuple[float, float, float] | None) -> int:
    keep = BASELINE_TARGET_KEEP if stages is None else stages[2]
    return int(round((1.0 - keep) * 100))


def eval_flags(splits_rel: str, run_dir: str, name: str, stages, cfg_dir: Path) -> list[str]:
    return [
        "--checkpoint", f"{run_dir}/best_checkpoint.pt",
        "--val-manifest", f"{splits_rel}/test.csv",
        "--prune-percent", str(effective_prune_percent(stages)),
        "--optimizer-name", name,
        "--csv-path", str(cfg_dir / "eval_row.csv"),
        "--plots-dir", str(GRID_DIR / "plots"),
        "--scores-dir", str(GRID_DIR / "scores"),
        "--amp", "bf16",
        "--tta",
    ]


# ── results aggregation ────────────────────────────────────────────────────────────────
def load_rows(selected: list[tuple]) -> list[dict]:
    rows = []
    for name, stages, _desc in selected:
        path = GRID_DIR / name / "eval_row.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            eval_rows = list(csv.DictReader(f))
        if not eval_rows:
            continue
        row = eval_rows[-1]
        s = stages if stages is not None else (None, None, None)
        rows.append({"config_name": name, "stage1_keep": s[0], "stage2_keep": s[1], "stage3_keep": s[2], **row})
    return rows


def write_metrics_csv(rows: list[dict]) -> None:
    if not rows:
        return
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = EXTRA_FIELDS + [k for k in rows[0] if k not in EXTRA_FIELDS]
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    head = f"{'config':<26}{'stages':<16}" + "".join(f"{c:>10}" for c in cols)
    print("\n" + "=" * len(head) + "\nGRID SEARCH SUMMARY (test set)\n" + "=" * len(head))
    print(head + "\n" + "-" * len(head))
    for r in rows:
        st = "/".join("-" if r[k] in (None, "") else f"{float(r[k]):g}" for k in ("stage1_keep", "stage2_keep", "stage3_keep"))
        print(f"{r['config_name']:<26}{st:<16}" + "".join(f"{float(r[c]):>10.4f}" for c in cols))
    print("=" * len(head))


def make_plots(rows: list[dict]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    for r in rows:
        st = "/".join("-" if r[k] in (None, "") else f"{float(r[k]):g}" for k in ("stage1_keep", "stage2_keep", "stage3_keep"))
        labels.append(f"{r['config_name']}\n({st})")
    written = []
    for metric, title, fname in (("accuracy", "Accuracy", "accuracy_comparison.png"),
                                 ("roc_auc", "ROC-AUC", "roc_auc_comparison.png")):
        values = [float(r[metric]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(rows)), 5.5))
        bars = ax.bar(range(len(rows)), values, color=[f"C{i}" for i in range(len(rows))])
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.4f}", xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"Non-uniform per-stage pruning: {title} (stage keeps = cumulative fraction kept)")
        ax.set_ylim(0, min(1.05, max(values) * 1.15 + 0.02))
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = PLOTS_DIR / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def finalize(selected: list[tuple]) -> None:
    rows = load_rows(selected)
    if not rows:
        print("No finished configs yet - nothing to summarize.")
        return
    write_metrics_csv(rows)
    print_summary(rows)
    for path in make_plots(rows):
        print(f"Wrote {path}")
    print(f"Wrote {METRICS_CSV}")


def backup(args: argparse.Namespace) -> None:
    if args.backup_dir is None:
        return
    try:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        if GRID_DIR.exists():
            shutil.copytree(GRID_DIR, args.backup_dir / GRID_DIR.name, dirs_exist_ok=True)
        if PLOTS_DIR.exists():
            shutil.copytree(PLOTS_DIR, args.backup_dir / "plots_grid_search", dirs_exist_ok=True)
        if METRICS_CSV.exists():
            shutil.copy2(METRICS_CSV, args.backup_dir / METRICS_CSV.name)
    except OSError as e:
        print(f"  WARNING: backup to {args.backup_dir} failed: {e}")


def main() -> None:
    args = parse_args()

    selected = list(CONFIGS)
    if args.configs:
        bad = [i for i in args.configs if not 1 <= i <= len(CONFIGS)]
        if bad:
            raise SystemExit(f"--configs values must be between 1 and {len(CONFIGS)} (got {bad})")
        selected = [CONFIGS[i - 1] for i in args.configs]
    if args.include_uniform_baseline:
        selected.append(BASELINE)

    if args.plots_only:
        finalize(selected)
        return

    try:
        splits_rel = str(args.splits_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        splits_rel = str(args.splits_dir)
    if not args.dry_run:
        missing = [f for f in ("train.csv", "val.csv", "test.csv") if not (args.splits_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} under {args.splits_dir} - build the split first "
                                    "(scripts/prune_sweep/run_full_sweep.py).")

    print("Per-stage keeps are CUMULATIVE fractions of the 432 tokens kept after stages at blocks 4/7/10.")
    print("Model floors still apply: >=25% of tokens per stage, <=50% pruned per stage.\n")

    for name, stages, desc in selected:
        cfg_dir = GRID_DIR / name
        run_dir = str((RUNS_DIR / name).resolve())
        train_cmd = [args.python, TRAIN_SCRIPT] + train_flags(args, splits_rel, run_dir, stages)
        eval_cmd = [args.python, EVAL_SCRIPT] + eval_flags(splits_rel, run_dir, name, stages, cfg_dir)

        print("=" * 70)
        print(f"{name}: {desc}  stages={stages if stages else f'uniform --prune-target-keep {BASELINE_TARGET_KEEP}'}")
        print("=" * 70)
        if args.dry_run:
            print("TRAIN:", " ".join(train_cmd))
            print("EVAL: ", " ".join(eval_cmd), "\n")
            continue
        if (cfg_dir / "eval_row.csv").exists() and not args.force:
            print("Already finished - skipping (use --force to redo).\n")
            continue

        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "eval_row.csv").unlink(missing_ok=True)
        for label, cmd in (("Training", train_cmd), ("Evaluating", eval_cmd)):
            print(f"--- {label} {name} ---")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                raise SystemExit(f"{label} failed for {name} (exit {result.returncode}). "
                                 "Finished configs are kept; re-run to continue from this one.")
        backup(args)
        finalize([c for c in selected if (GRID_DIR / c[0] / "eval_row.csv").exists()])

    if not args.dry_run:
        finalize(selected)
        backup(args)


if __name__ == "__main__":
    main()
