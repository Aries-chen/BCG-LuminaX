import datetime

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("start_second")
class StartSecondWorker(BaseWorker):
    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self, state: dict) -> int:
        with open(utils.match_first(state, self.path)) as f:
            dt = datetime.datetime.fromisoformat(f.readline().strip())
        return dt.hour * 3600 + dt.minute * 60 + dt.second
