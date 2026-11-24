import pandas as pd
from pathlib import Path

from wuji_index.base import BaseIndexBuilder


@BaseIndexBuilder.register("dir_collector")
class DirCollector(BaseIndexBuilder):
    def __init__(self, src_root: str) -> None:
        self.src_root = Path(src_root)

    def __call__(self, index: pd.DataFrame) -> pd.DataFrame:
        path = sorted(self.src_root.glob("*.npz"))
        patient_id = [p.name for p in path]
        return pd.DataFrame({"path": path, "patient_id": patient_id})
