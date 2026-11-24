import numpy as np
import pandas as pd
import random

from wuji_index.base import BaseIndexBuilder


@BaseIndexBuilder.register("subject_splitter")
class SubjectSplitter(BaseIndexBuilder):
    def __init__(
        self, _in_column: str = "patient_id", _out_column: str = "split", **ratio: float
    ) -> None:
        self.in_column = _in_column
        self.out_column = _out_column
        self.ratio = ratio

    def __call__(self, index: pd.DataFrame) -> pd.DataFrame:
        subjects = sorted(set(index[self.in_column]))
        n = len(subjects)
        siz = (np.array(list(self.ratio.values())) * n).astype(int)
        quotient, remainder = divmod(n - sum(siz), len(siz))
        siz += quotient
        siz[:remainder] += 1
        bound = [0] + siz.cumsum().tolist()
        random.seed(42)
        random.shuffle(subjects)
        splits = [subjects[l:r] for l, r in zip(bound[:-1], bound[1:])]
        for key, value in zip(self.ratio.keys(), splits):
            index.loc[index[self.in_column].isin(value), self.out_column] = key
        return index
