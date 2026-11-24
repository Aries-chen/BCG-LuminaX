"""Copy both sample directories to scratch, verify SHA-256, then remove the source copies."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

RECORD_IDS = ("tr03-0005", "tr03-0029")
SUFFIXES = (".hea", ".mat", ".arousal", "-arousal.mat")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _manifest(directory: Path, expected: set[str], *, allow_partial: bool = False) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"expected a regular record directory at {directory}")
    entries = sorted(directory.iterdir())
    names = {entry.name for entry in entries}
    if not names <= expected or (not allow_partial and names != expected):
        raise ValueError(f"unexpected record files in {directory}")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"expected only regular files in {directory}")
    return {entry.name: _digest(entry) for entry in entries}


def move_samples(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("source and destination must be different directories")
    target_dir.mkdir(parents=True, exist_ok=True)
    verified_sources = []
    for record_id in RECORD_IDS:
        expected = {record_id + suffix for suffix in SUFFIXES}
        source = source_dir / record_id
        target = target_dir / record_id
        if target.exists() or target.is_symlink():
            target_manifest = _manifest(target, expected)
        else:
            source_manifest = _manifest(source, expected)
            with tempfile.TemporaryDirectory(prefix=f".{record_id}-", dir=target_dir) as temp_dir:
                staging = Path(temp_dir) / record_id
                shutil.copytree(source, staging)
                target_manifest = _manifest(staging, expected)
                if target_manifest != source_manifest:
                    raise ValueError(f"SHA-256 mismatch during copy of {record_id}")
                staging.rename(target)

        if source.exists() or source.is_symlink():
            # A completed destination also supports resuming an interrupted source deletion.
            manifest = _manifest(source, expected, allow_partial=True)
            if any(target_manifest.get(name) != sha for name, sha in manifest.items()):
                raise ValueError(f"different contents at {source} and {target}; both copies retained")
            verified_sources.append((source, manifest))

    for source, manifest in verified_sources:
        if _manifest(source, set(manifest)) != manifest:
            raise ValueError(f"source changed after verification at {source}")
        for name in manifest:
            (source / name).unlink()
        source.rmdir()
    print(f"Both sample directories are verified at {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "fm_samples")
    parser.add_argument("--target-dir", type=Path, default=Path("/scratch/s5415436/BCG/test_data"))
    args = parser.parse_args()
    try:
        move_samples(args.source_dir, args.target_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Sample move failed: {exc}\n")


if __name__ == "__main__":
    main()
