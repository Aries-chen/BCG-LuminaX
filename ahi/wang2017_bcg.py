"""Wang 2017 BCG records, one-second event labels, and fixed-window datasets."""

from collections import Counter
import csv
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import signal
from scipy.io import loadmat

from sleepnetzero_event import AHI_TOKENS

FS = 100
WINDOW_SECONDS = 120
SMOKE_SUBJECTS = ("1001", "1013")
SPLIT_SIZES = {"train": 24, "val": 4, "test": 4}
BANDS = {"heartbreath": (0.1, 20.0), "snore": (30.0, 45.0)}
SEED = 42


def read_cohort_and_split(cohort_path: Path, split_path: Path) -> tuple[dict, dict]:
    with cohort_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("the cohort repeats a subject")
    cohort = {row["subject_id"]: row for row in rows if row["status"] == "included"}

    with split_path.open(newline="") as stream:
        assignments = list(csv.DictReader(stream))
    ids = [row["subject_id"] for row in assignments]
    if len(set(ids)) != len(ids):
        raise ValueError("subject overlap or repeat in split assignments")
    if not set(ids) <= cohort.keys():
        raise ValueError("the pilot split contains subjects outside the eligible cohort")
    if Counter(row["split"] for row in assignments) != SPLIT_SIZES:
        raise ValueError(f"expected patient-wise split sizes {SPLIT_SIZES}")
    splits = {name: [] for name in SPLIT_SIZES}
    for row in assignments:
        splits[row["split"]].append(row["subject_id"])
    if not set(SMOKE_SUBJECTS) <= set(splits["train"]):
        raise ValueError("both smoke subjects must belong to the training split")
    return cohort, splits


def _load_table(archive: ZipFile, filename: str) -> np.ndarray:
    variables = loadmat(BytesIO(archive.read(filename)), simplify_cells=True)
    arrays = [value for key, value in variables.items() if not key.startswith("__")]
    if len(arrays) != 1:
        raise ValueError(f"{filename} must contain exactly one data variable")
    return np.asarray(arrays[0])


def _event_labels(table: np.ndarray, event_columns: int, samples: int) -> np.ndarray:
    table = np.atleast_2d(table)
    if event_columns not in (5, 6) or table.shape[1] != event_columns:
        raise ValueError(f"unexpected event table dimensions {table.shape}")
    # The optional event-type column sits before Pos and Len.
    intervals = table[:, -2:].astype(np.float64)
    duration = table[:, 2].astype(np.float64)
    if (
        not np.isfinite(intervals).all()
        or np.any(intervals != np.floor(intervals))
        or np.any(intervals[:, 0] < 0)
        or np.any(intervals[:, 1] <= 0)
        or not np.array_equal(intervals[:, 1], duration)
        or np.any(intervals.sum(axis=1) * FS > samples)
    ):
        raise ValueError("invalid Pos/Len interval for waveform duration")
    labels = np.zeros(samples // FS, dtype=np.uint8)
    for start, length in intervals.astype(np.int64):
        labels[start:start + length] = 1
    return labels


def _selected_windows(labels: np.ndarray, mode: str, subject: str) -> np.ndarray:
    n = len(labels) // WINDOW_SECONDS
    positives = labels[:n * WINDOW_SECONDS].reshape(n, WINDOW_SECONDS).sum(axis=1)
    mixed = np.flatnonzero((positives > 0) & (positives < WINDOW_SECONDS))
    if mode == "smoke":
        if not mixed.size:
            raise ValueError("no complete window contains both event classes")
        return mixed[:1]
    if mode == "train":
        negative = np.flatnonzero(positives == 0)
        if min(len(mixed), len(negative)) < 8:
            raise ValueError("expected at least 8 mixed and 8 all-negative windows")
        rng = np.random.default_rng(SEED + int(subject))
        return np.sort(np.concatenate((rng.choice(mixed, 8, replace=False), rng.choice(negative, 8, replace=False))))
    if n < 48:
        raise ValueError("expected at least 48 full evaluation windows")
    return np.linspace(0, n - 1, num=48, dtype=np.int64)


def _subject_windows(archive: ZipFile, row: dict, mode: str) -> dict[str, np.ndarray]:
    subject = row["subject_id"]
    raw = _load_table(archive, row["bcg_file"])
    expected = Decimal(row["recording_seconds"]) * FS
    if raw.ndim != 1 or raw.size != expected or not np.isfinite(raw).all():
        raise ValueError("invalid or misaligned BCG waveform")
    labels = _event_labels(_load_table(archive, row["event_file"]), int(row["event_columns"]), raw.size)
    indices = _selected_windows(labels, mode, subject)
    n_windows = labels.size // WINDOW_SECONDS
    n_seconds = n_windows * WINDOW_SECONDS
    result = {
        "subject_id": np.full(len(indices), subject),
        "start_second": indices * WINDOW_SECONDS,
        "labels": labels[:n_seconds].reshape(n_windows, WINDOW_SECONDS)[indices],
    }

    for name, band in BANDS.items():
        sos = signal.butter(3, band, btype="bandpass", fs=FS, output="sos")
        filtered = signal.sosfiltfilt(sos, raw[:labels.size * FS])
        scale = filtered.std()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid {name} frequency band")
        target_rate = AHI_TOKENS[name]
        normalized = ((filtered - filtered.mean()) / scale).astype(np.float32)
        resampled = signal.resample_poly(normalized, target_rate, FS)
        if len(resampled) != labels.size * target_rate or not np.isfinite(resampled).all():
            raise ValueError(f"invalid {name} sample count or values")
        result[name] = resampled[:n_seconds * target_rate].reshape(n_windows, WINDOW_SECONDS * target_rate)[indices]
    return result


def collect_windows(archive_path: Path, cohort: dict, subjects: tuple | list, mode: str) -> dict[str, np.ndarray]:
    batches = []
    with ZipFile(archive_path) as archive:
        for subject in subjects:
            try:
                batches.append(_subject_windows(archive, cohort[subject], mode))
            except (KeyError, OSError, ValueError) as exc:
                raise ValueError(f"{subject}: {exc}") from exc
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
