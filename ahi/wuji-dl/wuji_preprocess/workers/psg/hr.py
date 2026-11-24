import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import typing as t
import os
import re
from wuji_preprocess import utils
from wuji_preprocess.workers.utils import compute_hr_from_xxi, interpolate_hr
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("psg.hr")
class HeartbeatWorker(BaseWorker):
    def __init__(self, path: str, quality_path: str, fs: float, window_size: int, quality_threshold: float) -> None:
        self.path = path
        self.fs = fs
        self.qpath = quality_path                   # ECG信号质量
        self.window_size = window_size              # 心率计算窗口宽度，秒
        self.qthreshold = quality_threshold         # ECG信号可用的质量阈值

    def __call__(self, state: dict) -> NDArray:
        # 获取所有ECG质量文件
        quality_files = utils.match_all(state, self.qpath)
        best_quality = self.qthreshold
        best_channel = None
        for quality_file in quality_files:
            quality_data = np.load(quality_file)
            quality = quality_data['quality']
            avg_quality = np.mean(quality)
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_channel = os.path.basename(
                    quality_file).split('_')[-1].split('.')[0]

        if best_channel is None:
            raise ValueError(
                "No ECG quality data found with sufficient quality")

        ecg_data_path = re.sub(r'\*(?=[^*]*$)', best_channel, self.path)
        ecg_data_file = utils.match_first(state, ecg_data_path)

        def parse(
            position: NDArray,
            fs: NDArray,
            n_samples: NDArray,
            rri: NDArray | None = None,
        ) -> NDArray[np.float32]:
            in_fs = float(fs)
            if len(rri) == 0:
                raise ValueError(
                    "No ECG quality data found with sufficient quality")
            else:
                hr, hr_fs = compute_hr_from_xxi(
                    position, rri, sample_rate=in_fs, n_samples=n_samples, window_sec=self.window_size)
                duration = int(len(hr) / hr_fs)
                heart_rate = interpolate_hr(
                    hr, duration, target_fs=int(self.fs))
                return heart_rate

        with np.load(ecg_data_file) as data:
            return parse(**data)
