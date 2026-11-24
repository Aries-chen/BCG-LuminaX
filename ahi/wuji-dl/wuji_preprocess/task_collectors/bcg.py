from pathlib import Path

from wuji_preprocess.base import BaseTaskCollector


@BaseTaskCollector.register("bcg")
class BCGTaskCollector(BaseTaskCollector):
    def __call__(self, src_root: str, dst_root: str) -> list[tuple[Path, Path]]:
        task = []
        for sub in Path(src_root).iterdir():
            task.append((sub, Path(dst_root, sub.name + ".npz")))
        return task
