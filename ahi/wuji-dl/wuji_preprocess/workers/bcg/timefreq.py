import numpy as np
import neurokit2 as nk
from numpy.typing import NDArray
from wuji.biosignal_transformer.bcg.bcg2timefreq import get_bcg_feature
import typing as t
from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker
from wuji_preprocess import utils


@BaseWorker.register("bcg.timefreq")
class BCGTimeFrequencyWorker(BaseWorker):
    def __init__(
        self, path: str, seg_duration: float, seg_overlap_ratio: float, tar_fs: t.Union[int, float] = None
    ) -> None:
        self.path = path
        self.seg_duration = seg_duration
        self.seg_overlap_ratio = seg_overlap_ratio
        self.tar_fs = tar_fs

    def __call__(self, state: dict) -> NDArray:
        def parse(bcg: NDArray, fs: NDArray) -> NDArray[np.float32]:
            if self.tar_fs is not None:
                bcg = nk.signal_resample(
                    bcg, sampling_rate=fs, desired_sampling_rate=self.tar_fs)
                fs = self.tar_fs
            utils.update_duration(state, int(bcg.shape[0] / int(fs)))
            return get_bcg_feature(
                bcg, int(fs), self.seg_duration, self.seg_overlap_ratio
            )

        with np.load(utils.match_first(state, self.path)) as data:
            return parse(**data)
