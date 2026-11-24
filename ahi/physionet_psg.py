"""Read and align the PhysioNet 2018 training records used for the smoke run."""

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess

import numpy as np
from scipy import io, ndimage, signal

FS = 200
EPOCH_SECONDS = 30
STAGE_DATASETS = ("wake", "nonrem1", "nonrem2", "nonrem3", "rem", "undefined")
RESPIRATORY_EVENTS = frozenset({
    "resp_centralapnea", "resp_obstructiveapnea", "resp_mixedapnea", "resp_hypopnea",
})
CHANNELS = ("ECG", "ABD", "AIRFLOW", "O1-M2")


@dataclass(frozen=True)
class Recording:
    signals: dict[str, np.ndarray]
    sample_stages: np.ndarray
    stage_epochs: np.ndarray
    event_seconds: np.ndarray
    respiratory_intervals: list[tuple[int, int]]

    def reference(self) -> dict:
        sleep = self.sample_stages > 0
        sleep_hours = float(sleep.sum()) / FS / 3600
        if sleep_hours == 0:
            raise ValueError("the record has no scored sleep for the AHI denominator")
        sleep_events = sum(bool(sleep[start]) for start, _ in self.respiratory_intervals)
        return {
            "signal_samples": int(sleep.size),
            "complete_stage_epochs": int(self.stage_epochs.size),
            "scored_stage_epochs": int(np.count_nonzero(self.stage_epochs >= 0)),
            "respiratory_events": len(self.respiratory_intervals),
            "respiratory_events_starting_in_sleep": sleep_events,
            "sleep_hours": sleep_hours,
            "annotation_ahi_per_hour": sleep_events / sleep_hours,
        }


def _read_mat_signals(stem: Path) -> dict[str, np.ndarray]:
    """Read the single-segment, format 16+24 MATLAB V4 layout used by these records."""
    lines = stem.with_suffix(".hea").read_text().splitlines()
    name, channels, fs, samples = lines[0].split()
    n_channels = int(channels)
    if name != stem.name or int(fs) != FS:
        raise ValueError(f"{stem}: unexpected record name or sample rate")
    digital = io.loadmat(stem.with_suffix(".mat"), variable_names=["val"])["val"]
    if digital.shape != (n_channels, int(samples)) or len(lines) < n_channels + 1:
        raise ValueError(f"{stem}: signal dimensions disagree with the header")
    result = {}
    for row, line in enumerate(lines[1:n_channels + 1]):
        filename, layout, calibration, _, adc_zero, *_, channel = line.split()
        if channel not in CHANNELS:
            continue
        if filename != stem.with_suffix(".mat").name or layout != "16+24":
            raise ValueError(f"{stem}: unsupported signal layout for {channel}")
        gain_text, _, baseline_text = calibration.partition("/")[0].partition("(")
        gain = float(gain_text)
        baseline = int(baseline_text.rstrip(")") or adc_zero)
        if gain <= 0:
            raise ValueError(f"{stem}: invalid signal gain for {channel}")
        result[channel] = (digital[row].astype(np.float32) - baseline) / gain
    return result


def _read_mit_notes(path: Path) -> list[tuple[int, str]]:
    """Decode WFDB MIT auxiliary notes for the local PhysioNet .arousal files."""
    data = path.read_bytes()
    cursor = sample = 0
    notes = []
    while cursor + 2 <= len(data):
        word, = struct.unpack_from("<H", data, cursor)
        cursor += 2
        if word == 0:
            return notes
        code, value = word >> 10, word & 1023
        if code == 59:  # SKIP stores the high 16 bits before the low 16 bits.
            if cursor + 4 > len(data):
                break
            high, low = struct.unpack_from("<hH", data, cursor)
            cursor += 4
            sample += (high << 16) + low
        elif code == 63:  # Auxiliary text is padded to an even byte count.
            padded_length = value + value % 2
            if cursor + padded_length > len(data):
                break
            notes.append((sample, data[cursor:cursor + value].decode("ascii").strip("\0 ")))
            cursor += padded_length
        elif code < 59:  # NUM, SUB and CHN are metadata and do not advance time.
            sample += value
    raise ValueError(f"{path}: truncated WFDB annotation stream")


def _read_wfdb(stem: Path) -> tuple[dict[str, np.ndarray], list[tuple[int, str]]]:
    try:
        import wfdb
    except ModuleNotFoundError as exc:
        if exc.name != "wfdb":
            raise
        return _read_mat_signals(stem), _read_mit_notes(stem.with_suffix(".arousal"))

    record = wfdb.rdrecord(str(stem), channel_names=list(CHANNELS), return_res=32)
    annotations = wfdb.rdann(str(stem), "arousal")
    if record.fs != FS or annotations.fs != FS:
        raise ValueError(f"{stem}: expected {FS} Hz for signals and annotations")
    return dict(zip(record.sig_name, record.p_signal.T)), list(
        zip(annotations.sample, annotations.aux_note)
    )


def read_sample_stages(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path) as data:
        group = data["data/sleep_stages"]
        first = group[STAGE_DATASETS[0]]
        if first.ndim != 2 or first.shape[0] != 1:
            raise ValueError(f"{path}: expected single-channel sleep stage vectors")
        stages = np.full(first.shape[1], -2, dtype=np.int8)
        for index, name in enumerate(STAGE_DATASETS):
            flags = group[name][0]
            if flags.shape != stages.shape or np.any((flags != 0) & (flags != 1)):
                raise ValueError(f"{path}: invalid sleep stage vector {name}")
            active = flags == 1
            if np.any(stages[active] != -2):
                raise ValueError(f"{path}: overlapping sleep stage masks")
            stages[active] = -1 if name == "undefined" else index
    if np.any(stages == -2):
        raise ValueError(f"{path}: samples missing a sleep stage mask")
    return stages


def _respiratory_intervals(notes: list[tuple[int, str]], n_samples: int) -> list[tuple[int, int]]:
    open_events = defaultdict(deque)
    intervals = []
    for sample, raw_note in notes:
        note = raw_note.strip()
        if note.startswith("(") and note[1:] in RESPIRATORY_EVENTS:
            open_events[note[1:]].append(int(sample))
        elif note.endswith(")") and note[:-1] in RESPIRATORY_EVENTS:
            starts = open_events[note[:-1]]
            if not starts:
                raise ValueError(f"unmatched respiratory event end {note} at sample {sample}")
            start = starts.popleft()
            if not 0 <= start < sample <= n_samples:
                raise ValueError(f"invalid respiratory interval {start}..{sample}")
            intervals.append((start, int(sample)))
    if any(open_events.values()):
        raise ValueError("unmatched respiratory event starts")
    return sorted(intervals)


def read_record(stem: Path, labels_python: Path | None = None) -> Recording:
    signals, notes = _read_wfdb(stem)
    label_path = stem.with_name(f"{stem.name}-arousal.mat")
    if labels_python is None:
        stages = read_sample_stages(label_path)
    else:
        raw = subprocess.check_output([str(labels_python), str(Path(__file__).resolve()), str(label_path)])
        stages = np.frombuffer(raw, dtype=np.int8)

    if any(name not in signals for name in CHANNELS):
        raise ValueError(f"{stem}: expected signal channels {CHANNELS}")
    if any(values.shape != stages.shape or not np.isfinite(values).all() for values in signals.values()):
        raise ValueError(f"{stem}: missing, misaligned, or non-finite PSG signal samples")
    epoch_samples = EPOCH_SECONDS * FS
    complete_epochs = stages.size // epoch_samples
    epoch_end = complete_epochs * epoch_samples
    stage_epochs = stages[epoch_samples // 2:epoch_end:epoch_samples]
    # PhysioNet's HDF5 export can move the transition sample by one position.
    disagreements = stages[:epoch_end].reshape(complete_epochs, epoch_samples) != stage_epochs[:, None]
    if np.any(disagreements.sum(axis=1) > 1):
        raise ValueError(f"{stem}: sleep stages do not align with 30-second epochs")

    respiratory_intervals = _respiratory_intervals(notes, stages.size)
    n_seconds = stages.size // FS
    event_seconds = np.zeros(n_seconds, dtype=np.float32)
    for start, stop in respiratory_intervals:
        first_second = start // FS
        end_second = min(n_seconds, (stop + FS - 1) // FS)
        event_seconds[first_second:end_second] = 1
    event_seconds[stages[FS // 2:n_seconds * FS:FS] < 0] = -1
    return Recording(signals, stages, stage_epochs, event_seconds, respiratory_intervals)


def resample(values: np.ndarray, target_hz: int) -> np.ndarray:
    return signal.resample_poly(values, target_hz, FS).astype(np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot normalize a constant or non-finite signal")
    return (values - values.mean()) / scale


def staging_features(record: Recording) -> dict[str, np.ndarray]:
    n_samples = record.stage_epochs.size * EPOCH_SECONDS * FS
    ecg = record.signals["ECG"][:n_samples]
    ecg_sos = signal.butter(3, (5, 20), btype="bandpass", fs=FS, output="sos")
    qrs = signal.sosfiltfilt(ecg_sos, ecg)
    energy = ndimage.uniform_filter1d(qrs ** 2, size=round(0.08 * FS)).clip(min=0)
    peaks, _ = signal.find_peaks(
        energy, distance=round(0.28 * FS), prominence=float(np.quantile(energy, 0.95)) * 0.2
    )
    intervals = np.diff(peaks) / FS
    plausible = (intervals >= 0.3) & (intervals <= 2.0)
    if np.count_nonzero(plausible) < 2:
        raise ValueError("too few plausible ECG RR intervals")
    ticks = np.arange(n_samples // (FS // 4)) / 4
    heartbeat = np.interp(ticks, peaks[1:][plausible] / FS, intervals[plausible])

    breath_sos = signal.bessel(3, (0.1, 1 / 3), btype="bandpass", fs=FS, output="sos")
    breath = signal.sosfiltfilt(breath_sos, record.signals["ABD"][:n_samples])
    amplitudes = np.ptp(record.signals["O1-M2"][:n_samples].reshape(-1, 2 * FS), axis=1)
    motion = np.zeros(amplitudes.shape, dtype=np.float32)
    histories = np.lib.stride_tricks.sliding_window_view(amplitudes, 15)[:-1]
    motion[15:] = amplitudes[15:] > histories.mean(axis=1) + 5 * histories.std(axis=1)
    return {
        "heartbeat": heartbeat.astype(np.float32),
        "breath": normalize(resample(breath, 4)),
        "body_movement": np.repeat(motion, 8),
    }


def ahi_features(record: Recording) -> dict[str, np.ndarray]:
    n_samples = record.event_seconds.size * FS
    return {
        "heartbreath": normalize(resample(record.signals["ABD"][:n_samples], 50)),
        "snore": normalize(resample(record.signals["AIRFLOW"][:n_samples], 250)),
    }


if __name__ == "__main__":
    # Permit an installed h5py interpreter to supply labels to the PyTorch environment.
    import sys

    sys.stdout.buffer.write(read_sample_stages(Path(sys.argv[1])).tobytes())
