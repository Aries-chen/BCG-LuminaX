import pandas as pd
import os

from wuji_index.base import BaseIndexBuilder


@BaseIndexBuilder.register("clinical_csv_loader")
class ClinicalCSVLoader(BaseIndexBuilder):
    def __init__(self, src_root: str, index_path: str, in_column: str = "psg_uid", out_column: str = "path") -> None:
        self.src_root = src_root
        self.index_path = index_path
        self.in_column = in_column
        self.out_column = out_column

    def get_path(self, psg_uid: str) -> str:
        return os.path.join(self.src_root, psg_uid.replace("/", "-") + ".npz")

    def __call__(self, index: pd.DataFrame) -> pd.DataFrame:
        csv = pd.read_csv(self.index_path)
        csv[self.out_column] = csv[self.in_column].apply(self.get_path)
        ok = csv[self.out_column].apply(os.path.exists)
        return csv[ok]
