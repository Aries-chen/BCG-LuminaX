"""Run the two-record BCG smoke test or the patient-wise pilot."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "wuji-dl"))

from losses.event_cls import EventDetectionLoss
from sleepnetzero_event import AHI_TOKENS, ARCHITECTURE, build_event_model
from wang2017_bcg import BANDS, SEED, SMOKE_SUBJECTS, WINDOW_SECONDS, collect_windows, read_cohort_and_split

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/scratch/s5415436/data/BCG/wang2017_bcg_v3_partial.zip")
RUNS = Path("/scratch/s5415436/BCG/runs")
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 8


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _loaders_data(windows: dict, *, shuffle: bool = False) -> DataLoader:
    tensors = [torch.from_numpy(windows[key]) for key in (*AHI_TOKENS, "labels")]
    return DataLoader(
        TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED) if shuffle else None,
    )


def _forward(model, batch, device):
    features = {key: tensor.to(device) for key, tensor in zip(AHI_TOKENS, batch[:-1])}
    targets = batch[-1].to(device, dtype=torch.float32)
    length = torch.full((targets.shape[0],), WINDOW_SECONDS, dtype=torch.long, device=device)
    logits = model(length=length, **features)["ahi_event"]
    if logits.shape != targets.shape:
        raise ValueError(f"logits {tuple(logits.shape)} disagree with targets {tuple(targets.shape)}")
    return logits, targets


def _optimizer_step(model, criterion, optimizer, batch, device, *, check_gradients=False):
    optimizer.zero_grad(set_to_none=True)
    logits, targets = _forward(model, batch, device)
    loss = criterion(logits, {"cls_labels": targets})["loss"]
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    loss.backward()
    if check_gradients:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all():
            raise ValueError("absent or non-finite gradients")
    optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def _evaluate(model, batches, device):
    model.eval()
    outputs = []
    labels = []
    total_bce = 0.0
    for batch in batches:
        logits, targets = _forward(model, batch, device)
        total_bce += float(binary_cross_entropy_with_logits(logits, targets, reduction="sum"))
        outputs.append(logits.cpu().numpy())
        labels.append(targets.cpu().numpy())
    scores = np.concatenate(outputs)
    targets = np.concatenate(labels)
    if not np.isfinite(scores).all() or not 0 < targets.mean() < 1:
        raise ValueError("evaluation requires finite predictions and both target classes")
    return {
        "loss": total_bce / targets.size,
        "auprc": float(average_precision_score(targets.ravel(), scores.ravel())),
        "auroc": float(roc_auc_score(targets.ravel(), scores.ravel())),
        "positive_fraction": float(targets.mean(dtype=np.float64)),
        "windows": len(targets),
    }, scores


def _make_windows(args, out_dir, cohort, subjects, mode):
    windows = collect_windows(args.archive, cohort, subjects, mode)
    np.savez(out_dir / f"windows_{mode}.npz", **windows)
    print(f"Prepared {mode}: {len(subjects)} patients, {len(windows['labels'])} windows", flush=True)
    return windows


def _smoke(args, out_dir, cohort, model, criterion, optimizer, device):
    windows = _make_windows(args, out_dir, cohort, SMOKE_SUBJECTS, "smoke")
    batch = next(iter(_loaders_data(windows)))
    head_weight = model.heads[0].fc.weight
    before = head_weight.detach().clone()
    loss = _optimizer_step(model, criterion, optimizer, batch, device, check_gradients=True)
    if torch.equal(before, head_weight) or not torch.isfinite(head_weight).all():
        raise ValueError("optimizer did not update the event head to finite weights")

    checkpoint = out_dir / "smoke.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = build_event_model().to(device).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        expected, _ = _forward(model, batch, device)
        reloaded, _ = _forward(restored, batch, device)
        if expected.shape != (2, WINDOW_SECONDS) or not torch.isfinite(expected).all():
            raise ValueError("invalid smoke outputs")
        torch.testing.assert_close(reloaded, expected)
    np.savez(out_dir / "smoke_predictions.npz", logits=reloaded.cpu().numpy())
    return {
        "passed": True,
        "subjects": list(SMOKE_SUBJECTS),
        "window_start_seconds": windows["start_second"].tolist(),
        "positive_seconds": windows["labels"].sum(axis=1).tolist(),
        "logit_shape": list(reloaded.shape),
        "finite_gradients": True,
        "parameter_update": True,
        "checkpoint_roundtrip": True,
        "loss": loss,
    }


def _pilot(args, out_dir, cohort, splits, model, criterion, optimizer, device):
    train = _make_windows(args, out_dir, cohort, splits["train"], "train")
    val = _make_windows(args, out_dir, cohort, splits["val"], "val")
    train_loader = _loaders_data(train, shuffle=True)
    val_loader = _loaders_data(val)
    initial, _ = _evaluate(model, val_loader, device)
    checkpoint = out_dir / "best.pt"
    best = None
    with (out_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "val_loss", "val_auprc", "val_positive_fraction"))
        writer.writerow((0, "", initial["loss"], initial["auprc"], initial["positive_fraction"]))
        stream.flush()
        print(f"Initial val AUPRC={initial['auprc']:.5f}, positive fraction={initial['positive_fraction']:.5f}", flush=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            weighted_loss = sum(
                len(batch[-1]) * _optimizer_step(model, criterion, optimizer, batch, device)
                for batch in train_loader
            )
            train_loss = weighted_loss / len(train_loader.dataset)
            val_metrics, _ = _evaluate(model, val_loader, device)
            writer.writerow((epoch, train_loss, val_metrics["loss"], val_metrics["auprc"], val_metrics["positive_fraction"]))
            stream.flush()
            print(f"Epoch {epoch:02d} train loss={train_loss:.5f} val loss={val_metrics['loss']:.5f}"
                  f" AUPRC={val_metrics['auprc']:.5f}", flush=True)
            if epoch == 1:
                first_train_loss = train_loss
            if best is None or val_metrics["auprc"] > best["val"]["auprc"]:
                best = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
                torch.save(model.state_dict(), checkpoint)

    loss_cutoff = 0.90 * first_train_loss
    ap_cutoff = max(initial["auprc"], initial["positive_fraction"]) + 0.02
    gate = best["train_loss"] <= loss_cutoff and best["val"]["auprc"] >= ap_cutoff
    # Test waveforms and event targets are loaded after validation selects the model.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = _make_windows(args, out_dir, cohort, splits["test"], "test")
    test_metrics, test_logits = _evaluate(model, _loaders_data(test), device)
    np.savez(out_dir / "test_predictions.npz", logits=test_logits, labels=test["labels"],
             subject_id=test["subject_id"], start_second=test["start_second"])
    return {
        "convergence_gate_passed": gate,
        "initial_val": initial,
        "epoch_one_train_loss": first_train_loss,
        "best": best,
        "thresholds": {"train_loss_at_most": loss_cutoff, "val_auprc_at_least": ap_cutoff},
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "pilot"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--cohort", type=Path, default=REPO_ROOT / "docs/wang2017-bcg-cohort.csv")
    parser.add_argument("--split", type=Path, default=REPO_ROOT / "docs/wang2017-bcg-pilot-split.csv")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run inside a GPU allocation or use --device cpu.")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    cohort, splits = read_cohort_and_split(args.cohort, args.split)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = args.out_dir or RUNS / f"wang2017-bcg-{args.mode}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.split, out_dir / "split.csv")
    info = {
        "mode": args.mode,
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python": sys.version.split()[0],
        "versions": {p: version(p) for p in ("numpy", "scipy", "torch", "transformers", "scikit-learn")},
        "archive_sha256": _sha256(args.archive),
        "cohort_sha256": _sha256(args.cohort),
        "split_sha256": _sha256(args.split),
        "seed": SEED,
        "window_seconds": WINDOW_SECONDS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": ARCHITECTURE,
        "bands_hz": BANDS,
        "branch_sample_rates_hz": AHI_TOKENS,
    }
    _write_json(out_dir / "run.json", info)
    model = build_event_model().to(device)
    criterion = EventDetectionLoss({"head_loss": 1.0, "iou_loss": 0.0})
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if args.mode == "smoke":
        summary = _smoke(args, out_dir, cohort, model, criterion, optimizer, device)
    else:
        summary = _pilot(args, out_dir, cohort, splits, model, criterion, optimizer, device)
    _write_json(out_dir / "summary.json", dict(info, **summary))
    print(f"Summary written to {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
