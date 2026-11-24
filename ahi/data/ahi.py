import pandas as pd
import torch
import typing as t
from wuji_dl import slicing

from data.default_dataset import *
import pdb
# from augment.default_augmenter import DefaultAugmenter
from tempfile import mkdtemp
import os.path as path
from concurrent.futures import ThreadPoolExecutor
import shutil
import torch.types


@dataclass
class Feature:
    head: str
    input_channels: int = 1
    token_len: int = 1
    dtype: str = "double"  # 仅在使用memmap时有意义
    is_input: bool = True
    pad_value: torch.types.Number = 0
    is_attribute: bool = False


class AHINetInput(t.TypedDict):
    """
    task interface
    """

    length: torch.Tensor
    ah_event: t.Optional[torch.Tensor]
    bcg_raw: t.Optional[torch.Tensor]



class AHIBatch(t.TypedDict):
    """
    task interface
    """

    id: list[t.Union[int, str]]
    net_input: AHINetInput
    ahi_event: torch.Tensor


PAD_STAGE = -1


class AHIDataset(DefaultDataset):
    def __init__(
        self,
        index: str,
        split: str,
        max_tokens: int,
        token_sec: int,
        features: list[dict] = [],
        stride_tokens: int = 0,  # 0 for truncation
        r_margin: float = 0,
        id: str | None = None,
        memmap: bool = False,
        aug: t.Optional[t.Dict] = None,
        weighted_sampling: t.Union[str, None] = None,
        **kwargs: t.Any,
    ) -> None:
        csv = pd.read_csv(index)
        csv = csv[(csv["split"] == split) & (csv["duration"] >= token_sec)]
        data: list[SampleIndex] = []
        rs = []  # 记录右侧读到多远
        total_n = 0
        self.features = [Feature(**args) for args in features]

        attributes = [feat.head for feat in self.features if feat.is_attribute]
        for i, (_, row) in enumerate(csv.iterrows()):
            n = int((row["duration"] - r_margin) // token_sec) - 3
            for l, r in slicing.window(n, max_tokens, stride_tokens):
                data.append(
                    SampleIndex(
                        id=i if id is None else row[id],
                        path=row["path"],
                        start=l + total_n if memmap else l,
                        end=r + total_n if memmap else r,
                        payload={attr: row[attr] for attr in attributes}, # 这里有个ah_event的row，对应了每隔npz的标签。1或者0
                    )
                )
            rs.append(r)
            total_n += r
        self.memmap = memmap
        if self.memmap:
            temp_dir = mkdtemp(dir=".")
            data_memmap = {
                feat.head: np.memmap(
                    path.join(temp_dir, feat.head),
                    mode="w+",
                    shape=(
                        (total_n * feat.token_len, feat.input_channels)
                        if feat.input_channels != 1
                        else (total_n * feat.token_len)
                    ),
                    dtype=np.dtype(feat.dtype),
                )
                for feat in self.features
            }

            self.path = temp_dir

            def load_data(i, path):
                with np.load(path) as npz:
                    for feat in self.features:
                        if feat.input_channels == 1:
                            data_memmap[feat.head][
                                sum(rs[:i])
                                * feat.token_len : sum(rs[: i + 1])
                                * feat.token_len
                            ] = npz[feat.head][0 : rs[i] * feat.token_len].flatten()
                        else:
                            data_memmap[feat.head][
                                sum(rs[:i])
                                * feat.token_len : sum(rs[: i + 1])
                                * feat.token_len,:
                            ] = npz[feat.head][0 : rs[i] * feat.token_len,:]
                for d in data_memmap.values():
                    d.flush()

            with ThreadPoolExecutor(max_workers=kwargs["num_workers"]) as executor:
                for i, (_, row) in enumerate(csv.iterrows()):
                    executor.submit(load_data, i, row["path"])
        else:
            data_memmap = None
        extractors = {
            feat.head: default_extractor(
                feat.head, feat.token_len, feat.input_channels  # rename(
            )
            for feat in self.features
        }
        #pdb.set_trace()
        collators = {
            feat.head: (
                feat.is_input,
                (
                    pad_collator(feat.head, pad_value=feat.pad_value)
                ),
            )
            for feat in self.features
        }
        super().__init__(
            data,
            data_memmap=data_memmap,
            extractors=extractors,
            collators=collators,
            dataloader_config=kwargs,
            # augmenter=augmenter,
            weighted_sampling=weighted_sampling ,#if split == "train" else None,
        )

    def __del__(self):
        if self.memmap:
            shutil.rmtree(self.path)
