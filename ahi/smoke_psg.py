"""Exercise both SleepNetZero tasks on two annotated PhysioNet PSGs."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

# This source snapshot includes local changes to WujiDL's 1-D ResNet blocks.
sys.path.insert(0, str(Path(__file__).resolve().parent / "wuji-dl"))

from losses.event_cls import EventDetectionLoss
from models.base import base_model
from models.sleep_staging import SleepStagingNet
from physionet_psg import EPOCH_SECONDS, ahi_features, read_record, staging_features
from wuji_dl.ops.cross_entropy import CrossEntropyLoss

SCRATCH_ROOT = Path("/scratch/s5415436/BCG")
RECORD_IDS = ("tr03-0005", "tr03-0029")
ARCHITECTURE = {
    "embedding_size": 64,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
}
STAGING_TOKENS = {"heartbeat": 120, "breath": 120, "body_movement": 120}
AHI_TOKENS = {"heartbreath": 50, "snore": 250}
WINDOW_SECONDS = 300
SEED = 42


def _window(labels: np.ndarray, length: int, require_binary_classes: bool) -> int:
    for start in range(0, labels.size - length + 1):
        chunk = labels[start:start + length]
        if np.any(chunk < 0):
            continue
        if not require_binary_classes or (np.any(chunk == 0) and np.any(chunk == 1)):
            return start
    raise ValueError(f"no valid contiguous window of {length} tokens")


def _step(
    task: str, features: dict[str, np.ndarray], labels: np.ndarray, device: torch.device
) -> tuple[dict, np.ndarray]:
    staging = task == "staging"
    token_sizes = STAGING_TOKENS if staging else AHI_TOKENS
    token_seconds = EPOCH_SECONDS if staging else 1
    n_tokens = WINDOW_SECONDS // token_seconds
    start = _window(labels, n_tokens, require_binary_classes=not staging)
    end = start + n_tokens
    specs = [dict(head=name, token_len=size, input_channels=1) for name, size in token_sizes.items()]
    inputs = {"length": torch.tensor([n_tokens], device=device)}
    for name, size in token_sizes.items():
        signal = features[name]
        if signal.size != labels.size * size or not np.isfinite(signal).all():
            raise ValueError(f"{task}: misaligned or non-finite {name} input")
        inputs[name] = torch.from_numpy(signal[start * size:end * size]).unsqueeze(0).to(device)

    torch.manual_seed(SEED)
    model_options = dict(ARCHITECTURE, features=specs)
    if staging:
        model = SleepStagingNet(num_classes=5, **model_options).to(device)
        criterion = CrossEntropyLoss(ignore_index=-1)
        targets = torch.tensor(labels[start:end], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(**inputs)
        expected_shape = (1, n_tokens, 5)
        loss = criterion(logits, targets)
    else:
        backbone = dict(model_options, _name="models.sleep_net_zero.SleepNetZeroBackbone")
        head = dict(
            _name="models.heads.seqProj", hidden_size=ARCHITECTURE["hidden_size"],
            output_name="ahi_event", num_classes=1,
        )
        model = base_model(backbone, [head]).to(device)
        criterion = EventDetectionLoss(loss_weights={"head_loss": 1.0, "iou_loss": 0.0})
        targets = torch.tensor(labels[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(**inputs)["ahi_event"]
        expected_shape = (1, n_tokens)
        loss = criterion(logits, {"cls_labels": targets})["loss"]

    if logits.shape != expected_shape or not torch.isfinite(logits).all():
        raise ValueError(f"{task}: unexpected or non-finite logits with shape {tuple(logits.shape)}")
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError(f"{task}: expected a finite scalar loss")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        raise ValueError(f"{task}: absent or non-finite gradients")
    output_layer = model.fc if staging else model.heads[0].fc
    before = output_layer.weight.detach().clone()
    optimizer.step()
    if not torch.isfinite(output_layer.weight).all() or torch.equal(before, output_layer.weight):
        raise ValueError(f"{task}: optimizer did not produce a finite parameter update")

    classes, counts = np.unique(labels[start:end], return_counts=True)
    return {
        "window_start_second": start * token_seconds,
        "window_length_seconds": WINDOW_SECONDS,
        "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
        "logit_shape": list(logits.shape),
        "label_counts": {str(int(k)): int(v) for k, v in zip(classes, counts)},
        "loss": loss.item(),
        "finite_gradients": True,
        "optimizer_update": True,
    }, logits.detach().cpu().numpy()


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRATCH_ROOT / "test_data")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--labels-python", type=Path, help="Python interpreter with h5py, when it is separate from PyTorch"
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use an allocated GPU or pass --device cpu.")
    device = torch.device(args.device)
    torch.set_num_threads(2)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = args.out_dir or SCRATCH_ROOT / "runs" / f"psg-smoke-{stamp}"
    versions = {p: _version(p) for p in ("h5py", "numpy", "scipy", "torch", "transformers", "wfdb")}
    if args.labels_python is not None:
        versions["h5py"] = subprocess.check_output(
            [str(args.labels_python), "-c", "import h5py; print(h5py.__version__)"], text=True
        ).strip()
    elif versions["h5py"] is None:
        parser.error("h5py is required to read the labels. Install it or pass --labels-python.")
    summary = {
        "mode": "random_initialization_smoke",
        "seed": SEED,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
        "architecture": ARCHITECTURE,
        "records": {},
    }
    outputs = {}
    for record_id in RECORD_IDS:
        try:
            record = read_record(args.data_dir / record_id / record_id, args.labels_python)
            reference = record.reference()
            staging, outputs[f"{record_id}_staging"] = _step(
                "staging", staging_features(record), record.stage_epochs, device
            )
            ahi, outputs[f"{record_id}_ahi_event"] = _step(
                "ahi", ahi_features(record), record.event_seconds, device
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"{record_id}: {exc}") from exc
        summary["records"][record_id] = dict(reference, staging=staging, ahi=ahi)
        print(f"{record_id}  AHI={reference['annotation_ahi_per_hour']:.2f}/h"
              f"  staging_loss={staging['loss']:.5f}  ahi_loss={ahi['loss']:.5f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out_dir / "smoke_logits.npz", **outputs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
