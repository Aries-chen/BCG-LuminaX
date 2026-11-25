"""Cohort locking and fold-scoped, continuous Wang 2017 BCG caches."""

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import numpy as np
from scipy import signal
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from wang2017_bcg import AHI_TOKENS, BANDS, FS, SEED, WINDOW_SECONDS, _event_labels, _load_table

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "docs/wang2017-bcg-cohort.csv"
SPLIT = ROOT / "splits/wang2017_5fold_seed42.json"
ARCHIVE = Path("/scratch/s5415436/data/BCG/wang2017_bcg_v3_partial.zip")
CACHE_VERSION = 1
QUALITY = {"rail_values": [0, 4095], "rail_samples_per_second": 25,
           "suspected_off_bed_ptp": 2, "suspected_off_bed_seconds": 30}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def spans(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def quality_mask(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    ticks = raw.reshape(-1, FS)
    missing = ~np.isfinite(ticks).all(axis=1)
    safe = np.where(np.isfinite(ticks), ticks, 0)
    peak_to_peak = np.ptp(safe, axis=1)
    flat = peak_to_peak == 0
    low, high = QUALITY["rail_values"]
    saturated = np.count_nonzero((safe == low) | (safe == high), axis=1) >= QUALITY["rail_samples_per_second"]
    off_bed = np.zeros(len(ticks), dtype=bool)
    for start, end in spans((peak_to_peak <= QUALITY["suspected_off_bed_ptp"]) & ~missing):
        if end - start >= QUALITY["suspected_off_bed_seconds"]:
            off_bed[start:end] = True
    valid = ~(missing | flat | saturated | off_bed)
    counts = {key: int(mask.sum()) for key, mask in (
        ("missing", missing), ("flat", flat), ("saturated", saturated), ("suspected_off_bed", off_bed),
    )}
    return valid, counts


def read_record(archive: ZipFile, row: dict):
    raw = _load_table(archive, row["bcg_file"])
    if raw.ndim != 1 or len(raw) != round(float(row["recording_seconds"]) * FS):
        raise ValueError(f"{row['record_id']}: unexpected waveform length")
    table = _load_table(archive, row["event_file"])
    labels = _event_labels(table, int(row["event_columns"]), raw.size)
    raw = raw[:labels.size * FS]
    valid, qc = quality_mask(raw)
    if not valid.any():
        raise ValueError(f"{row['record_id']}: no valid recording time")
    coords = np.atleast_2d(table)[:, -2:].astype(np.int64)
    intervals = [[int(start), int(start + duration)] for start, duration in coords
                 if valid[start:start + duration].all()]
    return raw, labels, valid, intervals, qc


def _folds(group_ids: np.ndarray, y: np.ndarray):
    if len(group_ids) == len(set(group_ids)):
        assert len(group_ids) == len(np.unique(group_ids))
        return StratifiedKFold(5, shuffle=True, random_state=SEED).split(group_ids, y)
    return StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(group_ids, y, group_ids)


def build_split(archive_path: Path, cohort_path: Path) -> dict:
    with cohort_path.open(newline="") as stream:
        records = sorted((r for r in csv.DictReader(stream) if r["status"] == "included"),
                         key=lambda r: (r["subject_id"], r["bcg_file"]))
    for row in records:
        row["record_id"] = Path(row["bcg_file"]).stem
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("repeated record ID in cohort")
    grouped = {}
    with ZipFile(archive_path) as archive:
        for row in records:
            raw, labels, valid, events, qc = read_record(archive, row)
            row.update(seconds=len(labels), valid_seconds=int(valid.sum()), valid_events=len(events),
                       event_seconds=int(labels[valid].sum()), qc=qc)
            totals = grouped.setdefault(row["subject_id"], [0, 0])
            totals[0] += len(events)
            totals[1] += row["valid_seconds"]
    ids = np.array([r["subject_id"] for r in records])
    subject_rei = {key: count * 3600 / seconds for key, (count, seconds) in grouped.items()}
    clinical = {key: int(np.digitize(rei, [5, 15, 30])) for key, rei in subject_rei.items()}
    clinical_counts = np.bincount(list(clinical.values()), minlength=4)
    strata = clinical
    stratification = "reference_rei_bins"
    if np.min(clinical_counts) < 5:
        import pandas as pd

        strata = dict(zip(subject_rei, map(int, pd.qcut(list(subject_rei.values()), 4, labels=False))))
        stratification = "rei_quartiles_because_clinical_bin_has_fewer_than_five"
    y = np.array([strata[sid] for sid in ids])
    by_record = {row["record_id"]: row for row in records}
    by_subject = {sid: {"rei": rei, "clinical_bin": clinical[sid], "stratum": strata[sid]}
                  for sid, rei in subject_rei.items()}

    def partition(indices):
        subjects = sorted(set(ids[indices].tolist()))
        return {
            "subjects": subjects,
            "records": [records[i]["record_id"] for i in indices],
            "valid_seconds": sum(records[i]["valid_seconds"] for i in indices),
            "rei_mean": float(np.mean([subject_rei[sid] for sid in subjects])),
            "clinical_bins": np.bincount([clinical[sid] for sid in subjects], minlength=4).tolist(),
            "strata": np.bincount([strata[sid] for sid in subjects], minlength=4).tolist(),
        }

    folds = []
    for fold, (development, outer_test) in enumerate(_folds(ids, y)):
        i_train, i_val = next(_folds(ids[development], y[development]))
        folds.append({"fold": fold, "train": partition(development[i_train]),
                      "val": partition(development[i_val]), "test": partition(outer_test)})
    meta = dict(archive_sha256=file_hash(archive_path), cohort_sha256=file_hash(cohort_path),
                seed=SEED, quality=QUALITY, stratification=stratification,
                outer_splitter="StratifiedKFold" if len(ids) == len(grouped) else "StratifiedGroupKFold",
                clinical_bins=clinical_counts.tolist(), records=by_record, subjects=by_subject, folds=folds)
    validate_split(meta)
    return meta


def validate_split(meta: dict) -> None:
    records = meta["records"]
    subjects = set(meta["subjects"])
    tests = set()
    if len(meta["folds"]) != 5 or not subjects:
        raise ValueError("five nonempty folds required")
    if meta["outer_splitter"] == "StratifiedKFold" and len(records) != len(subjects):
        raise ValueError("StratifiedKFold requires exactly one record per subject")
    for i, fold in enumerate(meta["folds"]):
        roles = [set(fold[role]["subjects"]) for role in ("train", "val", "test")]
        if fold["fold"] != i or set.union(*roles) != subjects or sum(map(len, roles)) != len(subjects):
            raise ValueError(f"fold {i}: subject overlap or missing subject")
        for role, expected in zip(("train", "val", "test"), roles):
            assigned = fold[role]["records"]
            correct = {rid for rid, row in records.items() if row["subject_id"] in expected}
            if set(assigned) != correct or len(assigned) != len(correct):
                raise ValueError(f"fold {i}: wrong {role} record membership")
        if tests & roles[2]:
            raise ValueError(f"fold {i}: repeated outer test subject")
        tests.update(roles[2])
        if meta["outer_splitter"] == "StratifiedKFold" and meta["stratification"] == "reference_rei_bins":
            if {meta["subjects"][sid]["clinical_bin"] for sid in roles[2]} != set(range(4)):
                raise ValueError(f"fold {i}: outer test is missing a clinical bin")
    if tests != subjects:
        raise ValueError("outer test coverage differs from cohort")


def feature_arrays(raw: np.ndarray, valid: np.ndarray) -> dict:
    pieces = spans(valid)
    output = {}
    for name, band in BANDS.items():
        rate = AHI_TOKENS[name]
        sos = signal.butter(3, band, btype="bandpass", fs=FS, output="sos")
        data = np.zeros((len(valid), rate), dtype=np.float32)
        for a, b in pieces:
            filtered = signal.sosfiltfilt(sos, raw[a * FS:b * FS]).astype(np.float32)
            data[a:b] = signal.resample_poly(filtered, rate, FS).reshape(b - a, rate)
        if not np.isfinite(data).all():
            raise ValueError(f"{name}: non-finite filtered feature")
        output[name] = data
    return output


def get_records(meta: dict, fold: int, role: str, *, allow_test: bool = False) -> list[dict]:
    if role == "test" and not allow_test:
        raise ValueError("outer test access requires a frozen evaluation")
    assignments = meta["folds"][fold]
    ids = assignments[role]["records"]
    if role != "test" and set(ids) & set(assignments["test"]["records"]):
        raise ValueError("outer-test record in development loader")
    return [meta["records"][rid] for rid in ids]


def _assert_locked(row: dict, valid: np.ndarray, intervals: list) -> None:
    if int(valid.sum()) != row["valid_seconds"] or len(intervals) != row["valid_events"]:
        raise ValueError(f"{row['record_id']}: quality mask differs from split lock")


def prepare_cache(path: Path, archive_path: Path, records: list, fingerprint: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = json.loads((path / "meta.json").read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"cache fingerprint mismatch: {path}")
        return path
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        total = sum(row["seconds"] for row in records)
        arrays = {name: np.lib.format.open_memmap(temporary / f"{name}.npy", "w+", np.float32, (total, rate))
                  for name, rate in AHI_TOKENS.items()}
        labels = np.lib.format.open_memmap(temporary / "labels.npy", "w+", np.uint8, (total,))
        valid = np.lib.format.open_memmap(temporary / "valid.npy", "w+", bool, (total,))
        entries, offset = [], 0
        with ZipFile(archive_path) as archive:
            for row in records:
                raw, y, mask, references, _ = read_record(archive, row)
                _assert_locked(row, mask, references)
                end = offset + len(y)
                for name, data in feature_arrays(raw, mask).items():
                    arrays[name][offset:end] = data
                labels[offset:end], valid[offset:end] = y, mask
                entries.append(dict(subject_id=row["subject_id"], record_id=row["record_id"],
                                    start=offset, stop=end, events=references))
                offset = end
        for array in (*arrays.values(), labels, valid):
            array.flush()
        write_json(temporary / "meta.json", {"fingerprint": fingerprint, "entries": entries})
        try:
            temporary.rename(path)
        except OSError:
            if not path.exists() or json.loads((path / "meta.json").read_text())["fingerprint"] != fingerprint:
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return path


def fit_scaler(train_path: Path) -> tuple[dict, float]:
    valid = np.load(train_path / "valid.npy", mmap_mode="r")
    labels = np.load(train_path / "labels.npy", mmap_mode="r")
    indices = np.flatnonzero(valid)
    positive = int(labels[valid].sum())
    if not 0 < positive < len(indices):
        raise ValueError("training mask needs both classes")
    rng = np.random.default_rng(SEED)
    params = {}
    for name, rate in AHI_TOKENS.items():
        source = np.load(train_path / f"{name}.npy", mmap_mode="r")
        n = min(1_000_000, len(indices) * rate)
        sample = rng.choice(len(indices) * rate, size=n, replace=False)
        values = source[indices[sample // rate], sample % rate]
        lo, q1, median, q3, hi = map(float, np.percentile(values, [1, 25, 50, 75, 99]))
        if not math.isfinite(q3 - q1) or q3 <= q1:
            raise ValueError(f"{name}: degenerate train-only scaling")
        params[name] = {"lo": lo, "hi": hi, "median": median, "iqr": q3 - q1, "samples": n}
    return params, float(np.clip((len(indices) - positive) / positive, 1, 15))
