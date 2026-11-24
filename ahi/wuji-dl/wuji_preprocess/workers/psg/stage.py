import csv
import numpy as np
from numpy.typing import NDArray

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("psg.stage")
class StageWorker(BaseWorker):
    SEC_PER_STAGE: int = 30
    DEFAULT_MAPPING: dict[str, int] = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    DEFAULT_INVALID: int = -1

    def __init__(
        self,
        path: str,
        mapping: dict[str, int] = DEFAULT_MAPPING,
        invalid: int = DEFAULT_INVALID,
    ) -> None:
        self.path = path
        self.mapping = mapping
        self.invalid = invalid

    def __call__(self, state: dict) -> NDArray:
        with open(utils.match_first(state, self.path)) as f:
            n = int(state["duration"] // self.SEC_PER_STAGE)
            stages = np.full(n, self.invalid, dtype=np.int64)
            last = 0.0
            for line in csv.DictReader(f):
                stage = line["Type"]
                start = float(line["Start"])
                dur = float(line["Duration"])
                if start < last:
                    raise ValueError(f"start={start} < {last}")
                last = start + dur
                if dur < self.SEC_PER_STAGE:
                    continue
                if dur > self.SEC_PER_STAGE:
                    raise ValueError(f"dur={dur} > {self.SEC_PER_STAGE}")
                if start != int(start) or int(start) % self.SEC_PER_STAGE != 0:
                    raise ValueError(
                        f"start={start} is not multiple of {self.SEC_PER_STAGE}"
                    )
                pos = int(start) // self.SEC_PER_STAGE
                if pos >= n:
                    continue
                stages[pos] = self.mapping.get(stage, self.invalid)
            return stages
