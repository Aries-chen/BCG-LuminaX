from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import typing as t

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


Criterion = t.Callable[[NDArray], NDArray[np.bool_]]


def self_defined_eeg_order(state: dict) -> Path:
    priority_patterns = [
        "*/psg_quality_eeg_O*.npz",
        "*/psg_quality_eeg_C*.npz",
        "*/psg_quality_eeg_F*.npz",
    ]
    for p in priority_patterns:
        file_list = sorted(t.cast(Path, state["base_dir"]).glob(p))
        if file_list:
            return file_list[0]
    raise FileNotFoundError(f"No file found for all eeg patterns")


@BaseWorker.register("sleep_net_zero.body_movement")
class BodyMovementWorker(BaseWorker):
    CRITERION_REGISTER: dict[str, Criterion] = {}

    @staticmethod
    def register_criterion(name: str) -> t.Callable[[Criterion], Criterion]:
        def wrapper(func: Criterion) -> Criterion:
            BodyMovementWorker.CRITERION_REGISTER[name] = func
            return func

        return wrapper

    @staticmethod
    def get_criterion(name: str) -> Criterion:
        return BodyMovementWorker.CRITERION_REGISTER[name]

    def __init__(
        self, path: str, fs: float, criterion: str, use_self_defined_eeg: bool = False
    ) -> None:
        self.path = path
        self.fs = fs
        self.use_self_defined_eeg = use_self_defined_eeg
        self.criterion = self.get_criterion(criterion)

    def __call__(self, state: dict) -> NDArray:
        def parse(quality: NDArray, fs: NDArray) -> NDArray[np.bool_]:
            in_fs = float(fs)
            mask = self.criterion(quality)
            duration = len(mask) / in_fs
            utils.update_duration(state, duration)
            x = np.arange(duration * self.fs) / self.fs * in_fs
            return mask[x.round().astype(int).clip(0, len(mask) - 1)]

        if self.use_self_defined_eeg:
            fp = self_defined_eeg_order(state)
        else:
            fp = utils.match_first(state, self.path)
        with np.load(fp) as data:
            return parse(**data)


@BodyMovementWorker.register_criterion("<sigmoid(-0.5)")
def default_criterion(quality: NDArray) -> NDArray[np.bool_]:
    return quality < 1 / (1 + np.exp(0.5))


@BodyMovementWorker.register_criterion(">5")
def straightforward_criterion(quality: NDArray) -> NDArray[np.bool_]:
    return quality > 5


@BodyMovementWorker.register_criterion(">8")
def gt8_criterion(quality: NDArray) -> NDArray[np.bool_]:
    return quality > 8


@BodyMovementWorker.register_criterion(">0")
def gt0_criterion(quality: NDArray) -> NDArray[np.bool_]:
    return quality > 0
