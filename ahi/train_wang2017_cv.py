"""Run leakage-controlled Wang 2017 BCG five-fold training and frozen inference."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "wuji-dl"))

from sleepnetzero_event import AHI_TOKENS, ARCHITECTURE, build_event_model
from wang2017_cv_data import (ARCHIVE, BANDS, CACHE_VERSION, COHORT, SEED, SPLIT, WINDOW_SECONDS,
                             _assert_locked, feature_arrays, file_hash, fit_scaler, get_records,
                             prepare_cache, read_record, validate_split, write_json)
from wang2017_cv_events import DEFAULT, RecordScores, mae_f1, predict_events, tune_validation

BATCH = 64
MAX_EPOCHS = 80
STRIDE = 60
BASE_LR = 3e-4
FALLBACK_LR = 1e-4
CONFIG = dict(batch_size=BATCH, epochs=MAX_EPOCHS, window_seconds=WINDOW_SECONDS, eval_stride_seconds=STRIDE,
              learning_rate=BASE_LR, fallback_learning_rate=FALLBACK_LR, min_lr=1e-6, weight_decay=1e-2,
              warmup_epochs=5, min_stop_epoch=15, patience=12, gradient_clip=1.0, seed=SEED,
              architecture=ARCHITECTURE, bands_hz=BANDS, rates_hz=AHI_TOKENS, default_postprocessing=DEFAULT)


class Packed:
    def __init__(self, features, labels, valid, entries, scaler, device):
        self.device = device
        self.valid = torch.as_tensor(valid, dtype=torch.bool, device=device)
        self.labels = torch.as_tensor(labels, dtype=torch.uint8, device=device)
        self.entries = entries
        self.stops = torch.tensor([x["stop"] for x in entries], device=device)
        self.offsets = torch.arange(WINDOW_SECONDS, device=device)
        self.features = {}
        for name, source in features.items():
            f = torch.as_tensor(source, dtype=torch.float32, device=device)
            if device.type == "cpu":
                f = f.clone()
            p = scaler[name]
            f.clamp_(p["lo"], p["hi"]).sub_(p["median"]).div_(p["iqr"])
            f.masked_fill_(~self.valid[:, None], 0)
            self.features[name] = f

    @classmethod
    def cached(cls, path, scaler, device):
        meta = json.loads((path / "meta.json").read_text())
        arrays = {name: np.load(path / f"{name}.npy", mmap_mode="c") for name in AHI_TOKENS}
        return cls(arrays, np.load(path / "labels.npy", mmap_mode="c"), np.load(path / "valid.npy", mmap_mode="c"),
                   meta["entries"], scaler, device)

    def batch(self, starts, *, generator=None):
        end = self.stops[torch.searchsorted(self.stops, starts, right=True)]
        duration = torch.clamp(end - starts, max=WINDOW_SECONDS)
        index = torch.minimum(starts[:, None] + self.offsets, (end - 1)[:, None])
        valid = (self.offsets < duration[:, None]) & self.valid[index]
        inputs = {name: source[index] for name, source in self.features.items()}
        if generator is not None:
            scale = 0.85 + 0.30 * torch.rand((len(starts), 1, 1), device=self.device, generator=generator)
            sigma = 0.01 + 0.01 * torch.rand((len(starts), 1, 1), device=self.device, generator=generator)
            for data in inputs.values():
                data.mul_(scale).add_(torch.randn(data.shape, device=self.device, generator=generator) * sigma)
        for name, data in inputs.items():
            inputs[name] = data.masked_fill_(~valid[:, :, None], 0).flatten(1)
        return inputs, self.labels[index].float(), valid, duration, index

    def evaluation_starts(self):
        starts = [entry["start"] + local for entry in self.entries
                  for local in range(0, max(1, entry["stop"] - entry["start"] - WINDOW_SECONDS + STRIDE), STRIDE)]
        return torch.tensor(starts, device=self.device)


def segmentation_loss(logits, target, mask, pos_weight):
    logits = logits.float()
    weights = mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    bce = (bce * weights).sum() / weights.sum()
    probabilities = logits.sigmoid() * weights
    actual = target * weights
    dice = (2 * (probabilities * actual).sum() + 1) / (probabilities.sum() + actual.sum() + 1)
    return 0.5 * bce + 0.5 * (1 - dice)


def forward(model, dataset, starts, generator=None):
    inputs, target, valid, length, index = dataset.batch(starts, generator=generator)
    with torch.autocast(device_type=dataset.device.type, dtype=torch.bfloat16, enabled=dataset.device.type == "cuda"):
        logits = model(length=length, valid_mask=valid, **inputs)["ahi_event"]
    return logits.float(), target, valid, index


@torch.inference_mode()
def whole_record(model, dataset: Packed) -> tuple[list[RecordScores], torch.Tensor]:
    model.eval()
    summed = torch.zeros(len(dataset.valid), device=dataset.device)
    denominator = torch.zeros_like(summed)
    weights = torch.minimum(dataset.offsets + 1, WINDOW_SECONDS - dataset.offsets).float()
    for starts in dataset.evaluation_starts().split(BATCH):
        logits, _, valid, index = forward(model, dataset, starts)
        w = weights * valid
        summed.scatter_add_(0, index.flatten(), (logits.sigmoid() * w).flatten())
        denominator.scatter_add_(0, index.flatten(), w.flatten())
    if not torch.equal(denominator > 0, dataset.valid):
        raise ValueError("stitched timeline differs from valid coverage")
    probabilities = summed / denominator.clamp_min(1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("non-finite stitched predictions")
    p = probabilities.cpu().numpy()
    valid, labels = dataset.valid.cpu().numpy(), dataset.labels.cpu().numpy()
    records = [RecordScores(entry["subject_id"], entry["record_id"], p[entry["start"]:entry["stop"]],
                            valid[entry["start"]:entry["stop"]], labels[entry["start"]:entry["stop"]],
                            np.asarray(entry["events"], dtype=np.int64).reshape(-1, 2))
               for entry in dataset.entries]
    return records, probabilities


def validate_model(model, data, weight):
    records, p = whole_record(model, data)
    logits = torch.logit(p.clamp(1e-6, 1 - 1e-6))
    loss = segmentation_loss(logits, data.labels.float(), data.valid, weight)
    targets = [r.labels[r.valid] for r in records]
    scores = [r.probabilities[r.valid] for r in records]
    by_subject = {}
    for record, label, score in zip(records, targets, scores):
        group = by_subject.setdefault(record.subject_id, ([], []))
        group[0].append(label)
        group[1].append(score)
    ap = []
    for labels, values in by_subject.values():
        y = np.concatenate(labels)
        if y.any():
            ap.append(average_precision_score(y, np.concatenate(values)))
    mae, f1 = mae_f1(records, [predict_events(r, DEFAULT) for r in records])
    result = dict(val_loss=float(loss), val_micro_auprc=float(average_precision_score(np.concatenate(targets), np.concatenate(scores))),
                  val_macro_auprc=float(np.mean(ap)), val_macro_n=len(ap),
                  val_default_rei_mae=mae, val_default_event_f1=f1)
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("non-finite validation metric")
    return result


def _save_pt(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_pt(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def _emit_history(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def caches(root, archive, meta, split_hash, fold):
    paths = []
    for role in ("train", "val"):
        rows = get_records(meta, fold, role)
        fingerprint = dict(cache_version=CACHE_VERSION, split=split_hash, fold=fold,
                           archive=meta["archive_sha256"], role=role, records=[r["record_id"] for r in rows])
        path = prepare_cache(root / "caches" / f"fold-{fold}" / role, archive, rows, fingerprint)
        print(f"Ready {role} cache for fold {fold} with {len(rows)} records", flush=True)
        paths.append(path)
    return paths


class TrainingDiverged(RuntimeError):
    pass


def _fit(out, train, val, weight, scaler, run_id, epochs, learning_rate, resume, smoke):
    device = train.device
    valid_starts = torch.nonzero(train.valid, as_tuple=True)[0]
    windows = sum((int(train.valid[e["start"]:e["stop"]].sum()) + WINDOW_SECONDS - 1) // WINDOW_SECONDS
                  for e in train.entries)
    steps = (windows + BATCH - 1) // BATCH
    warmup, final = CONFIG["warmup_epochs"] * steps, MAX_EPOCHS * steps
    floor = CONFIG["min_lr"] / learning_rate

    def lr_multiplier(tick):
        if tick <= warmup:
            return tick / warmup
        fraction = min(1, (tick - warmup) / (final - warmup))
        return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2

    def new_state():
        model = build_event_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=CONFIG["weight_decay"])
        return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

    generator = torch.Generator(device=device).manual_seed(SEED)
    torch.manual_seed(SEED)
    model, optimizer, scheduler = new_state()
    start, history, best = 1, [], None

    def restore():
        state = _load_pt(out / "last.pt")
        if state["run_id"] != run_id or state["learning_rate"] != learning_rate:
            raise ValueError("resume settings do not match last.pt")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        return state

    if resume:
        state = restore()
        start, history, best = state["epoch"] + 1, state["history"], state["best"]
        if best:
            _save_pt(out / "best.pt", state["best_checkpoint"])
        del state
    for epoch in range(start, epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss, total_grad, n = 0., 0., 0
        selected = valid_starts[torch.randint(len(valid_starts), (windows,), device=device, generator=generator)]
        for starts in selected.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits, target, valid, _ = forward(model, train, starts, generator)
            loss = segmentation_loss(logits, target, valid, weight)
            if not torch.isfinite(loss):
                raise TrainingDiverged(f"epoch {epoch}: non-finite loss")
            loss.backward()
            try:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                raise TrainingDiverged(f"epoch {epoch}: non-finite gradient") from exc
            optimizer.step()
            scheduler.step()
            seconds = int(valid.sum())
            total_loss += float(loss.detach()) * seconds
            total_grad += float(norm)
            n += seconds
        values = validate_model(model, val, weight)
        row = dict(epoch=epoch, train_loss=total_loss / n, mean_gradient_norm=total_grad / steps,
                   lr=scheduler.get_last_lr()[0], seconds=time.monotonic() - started, **values)
        history.append(row)
        if best is None or (row["val_macro_auprc"], -row["val_default_rei_mae"]) > (best["val_macro_auprc"], -best["val_default_rei_mae"]):
            best = row
            _save_pt(out / "best.pt", dict(model=model.state_dict(), epoch=epoch, run_id=run_id,
                                           scaler=scaler, pos_weight=float(weight)))
        print(json.dumps(row, allow_nan=False), flush=True)
        _save_pt(out / "last.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       scheduler=scheduler.state_dict(), epoch=epoch, best=best, history=history,
                                       generator=generator.get_state(), cpu_rng=torch.get_rng_state(),
                                       cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                                       run_id=run_id, learning_rate=learning_rate,
                                       best_checkpoint=_load_pt(out / "best.pt")))
        _emit_history(out / "metrics.csv", history)
        if smoke and epoch == 1:
            model, optimizer, scheduler = new_state()
            if restore()["epoch"] != epoch:
                raise ValueError("smoke resume recovered the wrong epoch")
        if 8 <= epoch <= 10 and all(history[j]["train_loss"] >= 1.10 * history[j - 1]["train_loss"] for j in range(len(history) - 3, len(history))):
            raise TrainingDiverged(f"epoch {epoch}: three consecutive loss rebounds of at least 10%")
        if epoch >= CONFIG["min_stop_epoch"] and epoch - best["epoch"] >= CONFIG["patience"]:
            break
    checkpoint = _load_pt(out / "best.pt")
    model.load_state_dict(checkpoint["model"])
    if checkpoint["epoch"] != best["epoch"]:
        raise ValueError("best checkpoint epoch disagrees with metrics")
    return dict(best=best, epochs=len(history), sampled_windows_per_epoch=windows,
                slower_epoch_seconds=max(row["seconds"] for row in history), resume_checked=smoke,
                cuda_peak_gib=torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None)


def training(args, root, meta, split_hash, device):
    smoke = args.stage == "smoke"
    if smoke and args.fold != 0:
        raise ValueError("smoke is defined only for fold 0")
    out = root / ("smoke-fold-0" if smoke else f"fold-{args.fold}")
    out.mkdir(parents=True, exist_ok=args.resume)
    start = time.monotonic()
    train_path, val_path = caches(root, args.archive, meta, split_hash, args.fold)
    scaler, pos_weight = fit_scaler(train_path)
    train, val = (Packed.cached(p, scaler, device) for p in (train_path, val_path))
    prep_seconds = time.monotonic() - start
    weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    run_id = dict(fold=args.fold, split=split_hash, config=json.dumps(CONFIG, sort_keys=True), stage=args.stage)
    learning_rate = BASE_LR
    fallback_reason = None
    if args.resume:
        learning_rate = _load_pt(out / "last.pt")["learning_rate"]
    try:
        result = _fit(out, train, val, weight, scaler, run_id, 2 if smoke else MAX_EPOCHS,
                      learning_rate, args.resume, smoke)
    except TrainingDiverged as exc:
        epoch = int(str(exc).split(":", 1)[0].split()[-1])
        if smoke or learning_rate != BASE_LR or epoch > 10:
            raise
        fallback_reason = str(exc)
        old = out / "base_lr_attempt"
        old.mkdir(exist_ok=False)
        for name in ("best.pt", "last.pt", "metrics.csv"):
            p = out / name
            if p.exists():
                p.rename(old / name)
        learning_rate = FALLBACK_LR
        write_json(out / "fallback.json", {"reason": fallback_reason})
        result = _fit(out, train, val, weight, scaler, run_id, MAX_EPOCHS, learning_rate, False, False)
    fallback_info = out / "fallback.json"
    if fallback_info.exists():
        fallback_reason = json.loads(fallback_info.read_text())["reason"]
    if smoke:
        result["estimated_80_epoch_seconds"] = prep_seconds + 80 * result["slower_epoch_seconds"]
        if result["estimated_80_epoch_seconds"] > 0.8 * 4 * 3600:
            raise ValueError("80-epoch estimate exceeds the configured 20% PBS margin")
    write_json(out / "summary.json", dict(run_id=run_id, fit_seconds=time.monotonic() - start,
                                          prep_seconds=prep_seconds, learning_rate=learning_rate, fallback_reason=fallback_reason,
                                          device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", **result))


def fold_context(root, fold, split_hash, device):
    out = root / f"fold-{fold}"
    summary = json.loads((out / "summary.json").read_text())
    checkpoint = _load_pt(out / "best.pt")
    if checkpoint["run_id"] != summary["run_id"] or checkpoint["run_id"]["split"] != split_hash or checkpoint["epoch"] != summary["best"]["epoch"]:
        raise ValueError(f"fold {fold}: inconsistent best checkpoint")
    model = build_event_model().to(device)
    model.load_state_dict(checkpoint["model"])
    return out, model, checkpoint


def tuning(args, root, meta, split_hash, device):
    out, model, checkpoint = fold_context(root, args.fold, split_hash, device)
    if (out / "frozen.json").exists():
        raise ValueError("postprocessing already frozen for this fold")
    validation = get_records(meta, args.fold, "val")
    cache = root / "caches" / f"fold-{args.fold}" / "val"
    entries = json.loads((cache / "meta.json").read_text())["entries"]
    if [e["record_id"] for e in entries] != [r["record_id"] for r in validation]:
        raise ValueError("validation cache role mismatch")
    dataset = Packed.cached(cache, checkpoint["scaler"], device)
    records, _ = whole_record(model, dataset)
    chosen, candidates = tune_validation(records)
    _emit_history(out / "postprocessing_grid.csv", candidates)
    write_json(out / "frozen.json", dict(fold=args.fold, split=split_hash, checkpoint_sha256=file_hash(out / "best.pt"),
                                         val_subjects=meta["folds"][args.fold]["val"]["subjects"], postprocessing=chosen))
    print(f"Fold {args.fold} frozen postprocessing {chosen}", flush=True)


def frozen_folds(root, meta, split_hash):
    frozen = []
    for i in range(5):
        path = root / f"fold-{i}"
        item = json.loads((path / "frozen.json").read_text())
        if (item["fold"] != i or item["split"] != split_hash or item["checkpoint_sha256"] != file_hash(path / "best.pt")
                or item["val_subjects"] != meta["folds"][i]["val"]["subjects"]):
            raise ValueError(f"fold {i}: freeze metadata mismatch")
        frozen.append(item)
    return frozen


def testing(args, root, meta, split_hash, device):
    frozen = frozen_folds(root, meta, split_hash)[args.fold]
    out, model, checkpoint = fold_context(root, args.fold, split_hash, device)
    estimates = out / "outer_test"
    estimates.mkdir(exist_ok=True)
    verified = []
    with ZipFile(args.archive) as source:
        for row in get_records(meta, args.fold, "test", allow_test=True):
            rid, sid = row["record_id"], row["subject_id"]
            output = estimates / f"{rid}.npz"
            if output.exists():
                with np.load(output) as previous:
                    if (previous["checkpoint_sha256"].item() != frozen["checkpoint_sha256"]
                            or previous["subject_id"].item() != sid or previous["record_id"].item() != rid):
                        raise ValueError(f"{rid}: existing outer-test predictions do not match the lock")
            else:
                raw, labels, valid, reference, _ = read_record(source, row)
                _assert_locked(row, valid, reference)
                features = feature_arrays(raw, valid)
                entry = dict(record_id=rid, subject_id=sid, start=0, stop=len(labels), events=reference)
                packed = Packed(features, labels, valid, [entry], checkpoint["scaler"], device)
                records, _ = whole_record(model, packed)
                scored = records[0]
                events = predict_events(scored, frozen["postprocessing"])
                envelope = np.sqrt(np.mean(features["heartbreath"] ** 2, axis=1))
                temporary = output.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, subject_id=sid, record_id=rid, probabilities=scored.probabilities,
                                    labels=labels, valid=valid, reference=scored.references, predicted=events,
                                    envelope=envelope, checkpoint_sha256=frozen["checkpoint_sha256"])
                temporary.replace(output)
                print(f"Scored outer test fold {args.fold}, record {rid}", flush=True)
            verified.append(rid)
    write_json(out / "test_complete.json", dict(records=verified, checkpoint_sha256=frozen["checkpoint_sha256"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "train", "tune", "test"))
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        p.error("a CUDA device with BF16 support is required")
    torch.set_num_threads(2)
    meta = json.loads(args.split.read_text())
    validate_split(meta)
    if file_hash(args.archive) != meta["archive_sha256"] or file_hash(COHORT) != meta["cohort_sha256"]:
        raise ValueError("input files do not match the locked split")
    operations = {"smoke": training, "train": training, "tune": tuning, "test": testing}
    operations[args.stage](args, args.root, meta, file_hash(args.split), torch.device(args.device))


if __name__ == "__main__":
    main()
