import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray
import typing as t

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("sleep_net_zero.breath")
class BreathWorker(BaseWorker):
    def __init__(self, path: str, fs: float, do_filter: bool) -> None:
        self.path = path
        self.fs = fs
        self.do_filter = do_filter

    def __call__(self, state: dict) -> NDArray:
        def parse(respiration: NDArray, fs: NDArray) -> NDArray[np.float32]:
            in_fs = int(fs)
            if self.do_filter:
                respiration = t.cast(
                    NDArray,
                    nk.signal_filter(
                        respiration,
                        lowcut=0.1,
                        highcut=0.35,
                        method="bessel",
                        sampling_rate=in_fs,
                        order=3,
                    ),
                )
            utils.update_duration(state, len(respiration) / in_fs)
            respiration = t.cast(
                NDArray,
                nk.signal_resample(
                    respiration,
                    sampling_rate=in_fs,
                    desired_sampling_rate=self.fs,
                ),
            )
            utils.update_duration(state, len(respiration) / self.fs)
            respiration = (respiration - np.mean(respiration)) / np.std(respiration)
            respiration = respiration.astype(np.float32)
            assert np.isfinite(respiration).all(), "breath contains nan or inf"
            return respiration

        with np.load(utils.match_first(state, self.path)) as data:
            return parse(**data)
