import pandas as pd
import yaml

from wuji_dl.factory import create_instance
from wuji_index.base import BaseIndexBuilder


def build_index(config_path: str) -> None:
    with open(config_path) as f:
        config: list = yaml.safe_load(f)
    info = pd.DataFrame()
    for item in config:
        builder = create_instance(BaseIndexBuilder, item)
        info = builder(info)
