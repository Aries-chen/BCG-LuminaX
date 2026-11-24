import numpy as np
import os
from numpy.typing import NDArray
from wuji.Preprocessor.signal import interp_heartbeat
from wuji_preprocess import utils
from wuji_preprocess.workers.utils import compute_hr_from_xxi, interpolate_hr
from pathlib import Path
import typing as t
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("bcg.hr")
class HeartbeatWorker(BaseWorker):
    def __init__(self, path: str, quality_path: str, fs: float, window_sec: int, quality_threshold: float) -> None:
        self.path = path
        self.fs = fs
        self.qpath = quality_path                       # BCG信号质量
        self.window_sec = window_sec                    # 计算心率的窗口，奇数，秒
        self.qthreshold = quality_threshold             # BCG信号可用的质量阈值

    def __call__(self, state: dict) -> NDArray:
        def parse(
            position: NDArray,
            fs: NDArray,
            n_samples: NDArray,
            jji: t.Union[NDArray, None] = None,
        ) -> NDArray[np.float32]:
            quality_data = np.load(utils.match_first(state, self.qpath))
            quality = quality_data['quality']
            if np.mean(quality) > self.qthreshold:
                in_fs = float(fs)
                hr, hr_fs = compute_hr_from_xxi(
                    position, jji, sample_rate=in_fs, n_samples=n_samples, window_sec=self.window_sec)
                duration = int(len(hr) / hr_fs)
                heart_rate = interpolate_hr(
                    hr, duration, target_fs=int(self.fs))
                return heart_rate
            else:
                raise ValueError("Data quality is not sufficient.")

        with np.load(utils.match_first(state, self.path)) as data:
            return parse(**data)
