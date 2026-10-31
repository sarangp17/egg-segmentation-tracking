"""Metrics for binary masks (True = egg)."""
import numpy as np


def compute_metrics(pred, ref):
    """IoU, Dice, precision and recall for ONE image.

    Empty-mask rules (same everywhere in the project):
      - IoU / Dice: if there is nothing in either mask, the score is 1.0.
      - precision: if nothing was predicted, 1.0 when nothing was expected, else 0.0.
      - recall: if nothing was expected, 1.0.
    """
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if fn == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "dice": dice,
            "precision": precision, "recall": recall}


def summarise(metric_rows):
    """Macro = mean of per-image scores (every image counts equally).
    Micro = pixels pooled over all images first (big eggs count more)."""
    out = {}
    for k in ["iou", "dice", "precision", "recall"]:
        values = [r[k] for r in metric_rows]
        out["macro_" + k] = float(np.mean(values))
        out["median_" + k] = float(np.median(values))
        out["std_" + k] = float(np.std(values))
    TP = sum(r["tp"] for r in metric_rows)
    FP = sum(r["fp"] for r in metric_rows)
    FN = sum(r["fn"] for r in metric_rows)
    out["micro_iou"] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0
    out["micro_dice"] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 1.0
    out["micro_precision"] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    out["micro_recall"] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return out
