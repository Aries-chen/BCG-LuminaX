import sys
import os
from pathlib import Path

# 优先调用本地的wuji-dl
# sys.path.insert(0, "/nvme2/zjw/sleep-net-zero")
# sys.path.insert(0, "/nvme2/wyc/temp/wuji-dl")
# sys.path.insert(0, "/nvme2/wyc/temp/wuji-dl/wuji_index")
# sys.path.append("/local_data/wyc/wuji-dl/wuji_index")
# sys.path.append("/local_data/wyc/sleep-net-zero")


#from wuji_preprocess.runner import preprocess
#from wuji_index.runner import build_index
from wuji_dl.runner import test_runner, fit_runner

# preprocess("temp/shhs.yaml", jobs=1, num=1) # num = 1
# build_index("/nvme2/wyc/sleep-net-zero/index/sleep_staging/bcg.yaml")

fit_runner("configs/ahi/beta.yaml", devices=(1,), fast_dev_run=True)
# test_runner("configs/sleep_staging/exp.yaml", ckpt="log/sleep-staging/exp-fit-42-0901-181616/checkpoints/epoch=6-step=7182.ckpt", devices=(1,), fast_dev_run=True)

