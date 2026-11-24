from pathlib import Path

from wuji_preprocess.base import BaseTaskCollector


@BaseTaskCollector.register("writeback")
class WritebackTaskCollector(BaseTaskCollector):
    def __call__(self, src_root: str, dst_root: str) -> list[tuple[Path, Path]]:
        task = []
        for sub in Path(src_root).iterdir():
            task.append((sub, sub))
        return task

class HSPWritebackTaskCollector(BaseTaskCollector):
    def __call__(self, src_root: str, dst_root: str) -> list[tuple[Path, Path]]:
        task = []
        for sub in Path(src_root).iterdir():
            for ses in sub.iterdir():
                task.append((ses, ses))
        return task
