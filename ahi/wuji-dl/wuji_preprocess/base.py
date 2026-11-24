from pathlib import Path
import typing as t

from wuji_dl.factory import Factory


class BaseTaskCollector(Factory):
    def __call__(self, src_root: str, dst_root: str) -> list[tuple[Path, Path]]:
        raise NotImplementedError


class BaseWorker(Factory):
    def __call__(self, state: dict) -> t.Any: ...
