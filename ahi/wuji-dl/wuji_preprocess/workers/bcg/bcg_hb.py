import numpy as np
from numpy.typing import NDArray
from wuji.Preprocessor.signal import interp_heartbeat
import typing as t
from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("sleep_net_zero.bcg_hb")
class BCGHBWorker(BaseWorker):
    def __init__(self, path: str) -> None:
        self.path = path
        

    def __call__(self, state: dict) -> NDArray:
        with np.load(utils.match_first(state, self.path)) as data:
            utils.update_duration(state, int(data['data'].shape[0] )/ int(data['fs']) )  
            return data['data']
