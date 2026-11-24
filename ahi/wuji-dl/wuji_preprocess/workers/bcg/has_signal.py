import numpy as np

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("bcg.has_signal")
class HasSignalWorker(BaseWorker):
    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self, state: dict) -> bool:
        with np.load(utils.match_first(state, self.path)) as data:
            return data["has_signal"].all()
