import numpy as np
from numpy.typing import NDArray
import pandas as pd

from wuji_preprocess import utils
from wuji_preprocess.base import BaseWorker


@BaseWorker.register("psg.ah_event_detect")
class AHEventWorker(BaseWorker):
    DEFAULT_MAPPING = {"Apnea": 0, "Hypopnea": 1}

    def __init__(self, path: str, style: str, mapping: dict[str, int] = DEFAULT_MAPPING) -> None:
        self.path = path
        self.style = style # anchor or cls
        self.mapping = mapping

    def __call__(self, state: dict) -> NDArray:
        with open(utils.match_first(state, self.path)) as f:
            df = pd.read_csv(f)
            df["Type"] = df["Type"].map(self.mapping)
            if len(df) == 0:
                return np.zeros((int(state['duration'] / 1), 1)) if self.style == "cls" else np.zeros((int(state['duration'] / 60), 2))
                #return np.empty((0, 3), dtype=np.float32)
            else:
                nested_list = df[["Start", "Duration"]].values.tolist()
                labels = get_cls_labels(nested_list, state['duration'], 1) if self.style == "cls" else get_universal_labels(nested_list, 60, state['duration'], 2)
                return labels


def get_universal_labels(events, win_length = 30, max_time_stamp = 15.0, anchor_num = 3):
  """Generate the label matrix of an audio sample based on its annotations."""
  #no_of_div = 15
  #events = smoothe_events(events)
  #win_length = max_time_stamp/no_of_div  # 分子是音频长度（秒）， no_of_div是区间数,区间数目需要跟模型的输出维度一致，一般和音频长度近似
  no_of_div = int(max_time_stamp / win_length) + 1
  labels = np.zeros((no_of_div, 3*anchor_num)) # 1*3 ,目前只检测一类事件

  for e in events:

    start_time = float(e[0])
    stop_time = min(start_time + float(e[1]),max_time_stamp) - 0.00001

    start_bin = int(start_time // win_length)   #开始在第几个区间
    stop_bin = int(stop_time // win_length)

    start_time_2 = start_time - start_bin * win_length  # 在区间内的具体位置，或者可以看作是余数也可以
    stop_time_2 = stop_time - stop_bin * win_length

    n_bins = stop_bin - start_bin

    if n_bins == 0:  #如果开始结束在同个区间内
      for i in range(anchor_num):
        if labels[start_bin, i*3] == 0:
          labels[start_bin, i*3:i*3+3] = [1, start_time_2, stop_time_2]
          break
    elif n_bins == 1:
      for i in range(anchor_num):
        if labels[start_bin, i*3] == 0:
          labels[start_bin, i*3:i*3+3] = [1, start_time_2, win_length]
          break
      for i in range(anchor_num):
        if labels[stop_bin, i*3] == 0 and stop_time_2 > 0.0:
          labels[stop_bin, i*3:i*3+3] = [1, 0.0, stop_time_2]
          break

    elif n_bins > 1:
      for i in range(anchor_num):
        if labels[start_bin, i*3] == 0:
          labels[start_bin, i*3:i*3+3] = [1, start_time_2, win_length]
          break
      for i in range(1, n_bins):
        for j in range(anchor_num):
          if labels[start_bin + i, j*3] == 0:
            labels[start_bin + i, j*3:j*3+3] = [1, 0.0, win_length]
            break
      for i in range(anchor_num):
        if labels[stop_bin, i*3] == 0 and stop_time_2 > 0.0:
          labels[stop_bin, i*3:i*3+3] = [1, 0.0, stop_time_2]
          break
  for i in range(anchor_num):
    labels[:, [1 + i*3, 2 + i*3]] /= win_length
  return labels

def smoothe_events(events):
    snore_events = []
    music_events = []
    speech_events = []
    for e in events:
        if len(e) == 2:#if e[2] == "snore":
            snore_events.append([e[0],e[1]])
        elif e[2] == "speech":
            speech_events.append(e)
        elif e[2] == "music":
            music_events.append(e)
    snore_events.sort(key=lambda x: x[0])
    #speech_events.sort(key=lambda x: x[0])
    #music_events.sort(key=lambda x: x[0])
    start_speech = -1000
    stop_speech = -1000

    max_snore_silence = 0.4
    min_dur_snore = 0.1

    count = 0

    while count < len(snore_events) - 1:
        if (snore_events[count][1] >= snore_events[count + 1][0]) or (snore_events[count + 1][0] - snore_events[count][1] <= max_snore_silence):
            snore_events[count][1] = max(snore_events[count + 1][1], snore_events[count][1])
            del snore_events[count + 1]
        else:
            count += 1
    smooth_events = snore_events

    return smooth_events

def get_cls_labels(events, segment_len, interval = 0.01):
    cls_labels = np.zeros((int(segment_len / interval), 1))
    for e in events:
        start_time = e[0]
        stop_time = e[0] + e[1]
        start_bin = int(start_time // interval)
        stop_bin = int(stop_time // interval)
        cls_labels[start_bin:stop_bin] = 1
    return cls_labels
   #turn event list into classification label , interval is 0.01s, zero means no event, one means event