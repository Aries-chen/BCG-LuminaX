"""Mask-aware event decoding, matching and validation-only threshold search."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from wang2017_cv_data import spans

DEFAULT = {"smooth": 1, "threshold": 0.50, "gap": 2, "minimum": 10}


@dataclass
class RecordScores:
    subject_id: str
    record_id: str
    probabilities: np.ndarray
    valid: np.ndarray
    labels: np.ndarray
    references: np.ndarray


def smooth_valid(scores: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    if size == 1:
        return scores
    smooth = np.zeros_like(scores)
    radius = size // 2
    for start, end in spans(valid):
        segment = scores[start:end]
        ticks = np.arange(len(segment))
        left, right = np.maximum(ticks - radius, 0), np.minimum(ticks + radius + 1, len(segment))
        total = np.r_[0., np.cumsum(segment, dtype=np.float64)]
        smooth[start:end] = (total[right] - total[left]) / (right - left)
    return smooth


def decode(scores: np.ndarray, valid: np.ndarray, threshold: float, gap: int,
           minimum: int) -> np.ndarray:
    intervals = []
    positive = valid & (scores >= threshold)
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    for start, end in spans(positive).tolist():
        if intervals:
            prev_end = intervals[-1][1]
            if start - prev_end <= gap and invalid_prefix[start] == invalid_prefix[prev_end]:
                intervals[-1][1] = end
                continue
        intervals.append([start, end])
    return np.asarray([v for v in intervals if v[1] - v[0] >= minimum], dtype=np.int64).reshape(-1, 2)


def event_counts(reference: np.ndarray, predictions: np.ndarray, iou: float) -> tuple[int, int, int]:
    # Both streams are sorted and non-overlapping. Add edges only for overlapping events.
    rows, columns = [], []
    i = j = 0
    while i < len(reference) and j < len(predictions):
        a, b = reference[i]
        c, d = predictions[j]
        overlap = min(b, d) - max(a, c)
        if overlap > 0 and overlap / (max(b, d) - min(a, c)) >= iou:
            rows.append(i)
            columns.append(j)
        if b <= d:
            i += 1
        if d <= b:
            j += 1
    tp = 0
    if rows:
        edges = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)),
                           shape=(len(reference), len(predictions)))
        tp = int(np.count_nonzero(maximum_bipartite_matching(edges, perm_type="column") >= 0))
    return tp, len(predictions) - tp, len(reference) - tp


def divide(num, den):
    return float(num / den) if den else None


def event_metrics(counts) -> dict:
    tp, fp, fn = map(int, counts)
    return dict(tp=tp, fp=fp, fn=fn, precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn))


def predict_events(record: RecordScores, config: dict) -> np.ndarray:
    scores = smooth_valid(record.probabilities, record.valid, config["smooth"])
    return decode(scores, record.valid, config["threshold"], config["gap"], config["minimum"])


def patient_rows(records: list[RecordScores], events: list[np.ndarray], *, min_reference=0) -> list[dict]:
    people = {}
    for record, prediction in zip(records, events, strict=True):
        ref = record.references
        if min_reference:
            ref = ref[np.diff(ref, axis=1).ravel() >= min_reference]
        row = people.setdefault(record.subject_id, dict(subject_id=record.subject_id, seconds=0,
                                                       gt=0, pred=0, counts={0.3: [0, 0, 0], 0.5: [0, 0, 0]}))
        row["seconds"] += int(record.valid.sum())
        row["gt"] += len(ref)
        row["pred"] += len(prediction)
        for iou in row["counts"]:
            row["counts"][iou] = [a + b for a, b in zip(row["counts"][iou], event_counts(ref, prediction, iou))]
    for row in people.values():
        if row["seconds"] == 0:
            raise ValueError(f"{row['subject_id']}: no valid evaluation time")
        row["reference_rei"] = 3600 * row["gt"] / row["seconds"]
        row["predicted_rei"] = 3600 * row["pred"] / row["seconds"]
    return list(people.values())


def mae_f1(records: list[RecordScores], predictions: list[np.ndarray]) -> tuple[float, float]:
    counts = np.zeros(3, dtype=np.int64)
    totals = {}
    for record, predicted in zip(records, predictions, strict=True):
        x = totals.setdefault(record.subject_id, [0, 0])
        x[0] += len(predicted) - len(record.references)
        x[1] += int(record.valid.sum())
        counts += event_counts(record.references, predicted, 0.3)
    return float(np.mean([3600 * abs(delta) / time for delta, time in totals.values()])), event_metrics(counts)["f1"] or 0.0


def tune_validation(records: list[RecordScores]) -> tuple[dict, list[dict]]:
    candidates = []
    for size in (1, 3, 5, 7):
        smoothed = [smooth_valid(r.probabilities, r.valid, size) for r in records]
        for threshold, gap in product((i / 100 for i in range(20, 81, 5)), (2, 4, 6)):
            predictions = [decode(p, r.valid, threshold, gap, 10) for p, r in zip(smoothed, records)]
            mae, f1 = mae_f1(records, predictions)
            candidates.append(dict(smooth=size, threshold=threshold, gap=gap, minimum=10,
                                   rei_mae=mae, event_f1=f1))
    required = 0.9 * max(row["event_f1"] for row in candidates)
    best = min((r for r in candidates if r["event_f1"] >= required),
               key=lambda r: (r["rei_mae"], -r["event_f1"], r["smooth"], r["threshold"], r["gap"]))
    return best, candidates
