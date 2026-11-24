import typing as t

from wuji_preprocess.base import BaseWorker


@BaseWorker.register("state")
class StateWorker(BaseWorker):
    def __init__(self, key: str) -> None:
        self.key = key

    def __call__(self, state: dict) -> t.Any:
        return state[self.key]
