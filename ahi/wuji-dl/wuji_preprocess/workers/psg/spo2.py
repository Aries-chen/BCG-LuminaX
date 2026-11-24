import numpy as np
from numpy.typing import NDArray
import typing as t
from wuji.Preprocessor.signal import filter_spo2
from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("psg.spo2")
class SpO2Worker(BaseWorker):
    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self, state: dict) -> NDArray:
        def parse(
            fs: NDArray, spo2: t.Union[NDArray, None] = None, raw: t.Union[NDArray, None] = None
        ) -> NDArray[np.float32]:
            if spo2 is None:
                assert raw is not None
                spo2 = raw
            in_fs = float(fs)
            spo2 = t.cast(NDArray, filter_spo2(spo2, in_fs)).astype(np.float32)
            utils.update_duration(state, len(spo2))
            nan_ratio = np.isnan(spo2).mean()
            assert nan_ratio < 0.4, f"spo2 contains high ratio NaN: {nan_ratio}"
            return spo2

        with np.load(utils.match_first(state, self.path)) as data:
            return parse(**data)
