import pandas as pd
from pathlib import Path
import typing as t

from wuji_dl.factory import Factory


class BaseIndexBuilder(Factory):
    def __call__(self, index: pd.DataFrame) -> pd.DataFrame: ...
