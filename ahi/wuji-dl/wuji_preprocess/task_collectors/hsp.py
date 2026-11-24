from pathlib import Path

from wuji_preprocess.base import BaseTaskCollector


@BaseTaskCollector.register("hsp")
class HSPTaskCollector(BaseTaskCollector):
    def __call__(self, src_root: str, dst_root: str) -> list[tuple[Path, Path]]:
        task = []
        for sub in Path(src_root).iterdir():
            for ses in sub.iterdir():
                task.append((ses, Path(dst_root, sub.name, ses.name + ".npz")))
        return task
