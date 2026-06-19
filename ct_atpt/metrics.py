from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    roc_auc: float
    pr_auc: float
    best_balanced_accuracy: float
    best_threshold: float
    score_min: float
    score_mean: float
    score_max: float
    tp: int
    tn: int
    fp: int
    fn: int


def _safe_roc_auc(labels: list[int], positive_scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, positive_scores))


def _safe_pr_auc(labels: list[int], positive_scores: list[float]) -> float:
    if sum(labels) == 0:
        return 0.0
    return float(average_precision_score(labels, positive_scores))


def compute_binary_metrics(
    labels: list[int],
    positive_scores: list[float],
    threshold: float = 0.5,
) -> BinaryMetrics:
    if len(labels) != len(positive_scores):
        raise ValueError("labels and positive_scores must have the same length")
    if not labels:
        raise ValueError("Cannot compute metrics for an empty prediction list")

    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(positive_scores, dtype=np.float64)
    preds = [1 if score >= threshold else 0 for score in positive_scores]
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    sensitivity = float(recall_score(labels, preds, pos_label=1, zero_division=0))
    specificity = float(recall_score(labels, preds, pos_label=0, zero_division=0))
    precision = float(precision_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    accuracy = float(accuracy_score(labels, preds))
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    pos_mask = labels_np == 1
    neg_mask = labels_np == 0
    best_balanced_accuracy = 0.0
    best_threshold = float(threshold)
    for candidate in np.unique(np.concatenate([scores_np, np.asarray([threshold], dtype=np.float64)])):
        pred_np = scores_np >= candidate
        cand_sens = float(pred_np[pos_mask].mean()) if pos_mask.any() else 0.0
        cand_spec = float((~pred_np[neg_mask]).mean()) if neg_mask.any() else 0.0
        cand_bal = 0.5 * (cand_sens + cand_spec)
        if cand_bal > best_balanced_accuracy:
            best_balanced_accuracy = cand_bal
            best_threshold = float(candidate)

    return BinaryMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        roc_auc=_safe_roc_auc(labels, positive_scores),
        pr_auc=_safe_pr_auc(labels, positive_scores),
        best_balanced_accuracy=best_balanced_accuracy,
        best_threshold=best_threshold,
        score_min=float(scores_np.min()),
        score_mean=float(scores_np.mean()),
        score_max=float(scores_np.max()),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )


def format_binary_metrics(metrics: BinaryMetrics, prefix: str = "val") -> str:
    return (
        f"{prefix}_acc={metrics.accuracy:.4f} "
        f"{prefix}_bal_acc={metrics.balanced_accuracy:.4f} "
        f"{prefix}_auc={metrics.roc_auc:.4f} "
        f"{prefix}_pr_auc={metrics.pr_auc:.4f} "
        f"{prefix}_sens={metrics.sensitivity:.4f} "
        f"{prefix}_spec={metrics.specificity:.4f} "
        f"{prefix}_prec={metrics.precision:.4f} "
        f"{prefix}_f1={metrics.f1:.4f} "
        f"{prefix}_best_bal_acc={metrics.best_balanced_accuracy:.4f}@{metrics.best_threshold:.3f} "
        f"{prefix}_score=[{metrics.score_min:.3f},{metrics.score_mean:.3f},{metrics.score_max:.3f}] "
        f"{prefix}_cm=[tn={metrics.tn},fp={metrics.fp},fn={metrics.fn},tp={metrics.tp}]"
    )
