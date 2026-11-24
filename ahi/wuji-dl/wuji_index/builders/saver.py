import pandas as pd
import os

from wuji_index.base import BaseIndexBuilder


@BaseIndexBuilder.register("saver")
class Saver(BaseIndexBuilder):
    def __init__(self, dst_path: str) -> None:
        self.dst_path = dst_path

    def __call__(self, index: pd.DataFrame) -> pd.DataFrame:
        index.to_csv(self.dst_path, index=False)
        return index
