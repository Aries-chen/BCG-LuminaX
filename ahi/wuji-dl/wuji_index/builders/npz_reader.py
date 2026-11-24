import numpy as np
import pandas as pd
import typing as t

from wuji_index.base import BaseIndexBuilder


@BaseIndexBuilder.register("npz_reader")
class NpzReader(BaseIndexBuilder):
    def __init__(self, check: t.Union[str, list[str]], load: t.Union[str, list[str]]) -> None:
        self.check = set(check) if isinstance(check, list) else {check}
        self.load = load if isinstance(load, list) else [load]
        self.check = self.check.union(self.load)
        self.n_load = len(self.load)

    def read_npz(self, path: str, ok: list[bool], loaded: list[list]):
        with np.load(path) as npz:
            cur_ok = self.check.issubset(npz.files)
            ok.append(cur_ok)
            if not cur_ok:
                return
            for i in range(self.n_load):
                loaded[i].append(npz[self.load[i]].item())

    def __call__(self, index: pd.DataFrame) -> pd.DataFrame:
        ok = []
        loaded = [[] for _ in range(self.n_load)]
        for path in index["path"]:
            self.read_npz(path, ok, loaded)
        index = index[ok].copy()
        for i in range(self.n_load):
            index[self.load[i]] = loaded[i]
        return index
