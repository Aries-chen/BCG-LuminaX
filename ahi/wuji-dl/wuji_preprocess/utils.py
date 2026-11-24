from pathlib import Path
import typing as t


def match_first(state: dict, pattern: str, key: t.Any = None, order: int = 0) -> Path:
    file_list = sorted(t.cast(Path, state["base_dir"]).glob(pattern), key=key)
    if not file_list:
        raise FileNotFoundError(f"No file found for {pattern}")
    if order >= len(file_list):
        return file_list[-1]
    return file_list[order]


def update_duration(state: dict, duration: float) -> None:
    if "duration" not in state or duration < state["duration"]:
        state["duration"] = duration


def match_all(state: dict, pattern: str):
    file_list = sorted(t.cast(Path, state["base_dir"]).glob(pattern))
    if not file_list:
        raise FileNotFoundError(f"No file found for {pattern}")
    return file_list
