import numpy as np
from numpy.typing import NDArray
from wuji.Preprocessor.signal import interp_heartbeat
import typing as t
from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("sleep_net_zero.heartbeat")
class HeartbeatWorker(BaseWorker):
    def __init__(self, path: str, fs: float) -> None:
        self.path = path
        self.fs = fs

    def __call__(self, state: dict) -> NDArray:
        def parse(
            position: NDArray,
            fs: NDArray,
            n_samples: NDArray,
            rri: t.Union[NDArray, None] = None,
            jji: t.Union[NDArray, None] = None,
        ) -> NDArray[np.float32]:
            in_fs = float(fs)
            duration = n_samples.item() / in_fs
            utils.update_duration(state, duration)
            ibi = jji if rri is None else rri
            assert ibi is not None
            heartbeat = interp_heartbeat(
                positions=position,
                values=ibi,
                fs=fs,
                target_fs=self.fs,
                n_samples=n_samples.item(),
            )
            assert np.isfinite(heartbeat).all(), f"heartbeat contains nan or inf"
            return heartbeat / 1000

        with np.load(utils.match_first(state, self.path)) as data:
            return parse(**data)
