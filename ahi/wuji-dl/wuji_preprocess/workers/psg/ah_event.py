import numpy as np
from numpy.typing import NDArray
import pandas as pd

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("psg.ah_event")
class AHEventWorker(BaseWorker):
    DEFAULT_MAPPING = {"Apnea": 0, "Hypopnea": 1}

    def __init__(self, path: str, mapping: dict[str, int] = DEFAULT_MAPPING) -> None:
        self.path = path
        self.mapping = mapping

    def __call__(self, state: dict) -> NDArray:
        with open(utils.match_first(state, self.path)) as f:
            df = pd.read_csv(f)
            df["Type"] = df["Type"].map(self.mapping)
            if len(df) == 0:
                return np.empty((0, 3), dtype=np.float32)
            else:
                return df[["Type", "Start", "Duration"]].to_numpy()
