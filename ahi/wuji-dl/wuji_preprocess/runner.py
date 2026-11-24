import logging
import multiprocessing as mp
import numpy as np
from pathlib import Path
import typing as t
import yaml
import traceback

from wuji_dl.factory import create_instance, get_class
from wuji_preprocess.base import BaseTaskCollector, BaseWorker


_logger: logging.Logger = logging.getLogger("wuji_preprocess.runner")


class Pipeline:
    def __init__(self, config: dict[str, dict[str, t.Any]], crash: bool):
        self.crash = crash
        self.workers = [(k, create_instance(BaseWorker, v)) for k, v in config.items()]

    def __call__(self, src: Path, dst: Path) -> None:
        info = {}
        state = {"base_dir": src}
        for key, worker in self.workers:
            backup = state.copy()
            try:
                info[key] = worker(state)
            except Exception as e:
                _logger.warn(f"Failed to process {key} to {dst}: {e}")
                _logger.warn(traceback.format_exc())
                state = backup
                if self.crash:
                    raise
        dst.parent.mkdir(exist_ok=True, parents=True)
        if dst.suffix == ".npz":
            np.savez(dst, **info)
        elif dst.is_dir():
            for k, v in info.items():
                v.to_csv(dst / k, index=False)
        else:
            raise ValueError(f"Invalid dst type: {dst}")
        _logger.info(f"OK: {src} => {dst}")


def preprocess(
    config_path: str,
    src_root: t.Union[str, None] = None,
    dst_root: t.Union[str, None] = None,
    num: t.Union[int, None] = None,
    log: t.Union[str, None] = None,
    jobs: int = 1,
    crash: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, filename=log)
    with open(config_path) as f:
        config: dict = yaml.safe_load(f)
    task_collector = get_class(BaseTaskCollector, config.pop("_task"))()
    if "_src" in config:
        src_root = config.pop("_src")
    if "_dst" in config:
        dst_root = config.pop("_dst")
    assert None not in (src_root, dst_root)
    pipeline = Pipeline(config, crash)
    tasks = task_collector(src_root, dst_root)
    with mp.Pool(jobs) as pool:
        pool.starmap(pipeline, tasks[:num])
