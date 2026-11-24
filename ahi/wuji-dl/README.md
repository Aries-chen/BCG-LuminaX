# WujiDL: 基于 Lightning 和 Weights and Biases 的深度学习框架

## 安装

```bash
pip install .
```

## 配置 wandb 登录

```bash
export WANDB_API_KEY=[登录密钥]
export WANDB_USERNAME=[用户名]
export WANDB_ENTITY=[团队名]
```

## 训练

```bash
wuji-fit CONFIG [--find-unused-parameters] [--ckpt CKPT]
```

按 CONFIG 路径下的配置文件训练。
--find-unused-parameters ：多卡训练用。
--ckpt CKPT ：从 CKPT 路径下的保存点恢复训练。

## 测试

```bash
wuji-test CONFIG CKPT
```

按 CONFIG 路径下的配置文件测试，模型从 CKPT 路径下的保存点加载。

### validate/predict

与 wuji-test 作用相同，只是换一下用的数据集和方法名称

```bash
wuji-dl validate CONFIG CKPT
wuji-dl predict CONFIG CKPT
```

## CLI 公用参数

```
--devices DEVICES   所用 GPU 的个数或编号列表。只用一个 GPU 时，编号后要跟逗号。如果没有指定，默认用所有 GPU 。
--seed SEED         随机数种子。默认为 42 。
--fast-dev-run      调试模式。禁用 logging 和 checkpointing ，只跑 1 batch 。
--fast-dev-run N    调试模式。禁用 logging 和 checkpointing ，只跑 N batch 。
--project PROJECT   手动指定 wandb 的 project 名。如果没有指定，在配置文件里查找；如果配置文件也没有指定，从环境变量查找；找不到默认为 `lightning_logs` 。
--group GROUP       手动指定 wandb 的 group 名。如果没有指定，在配置文件里查找；如果配置文件也没有指定，默认为配置文件的文件名的主干部分。
--job-type JOB_TYPE 手动指定 wandb 的 job type 。如果没有指定，在配置文件里查找；如果配置文件也没有指定，训练时默认为 fit ，测试时默认为 test 。
```

## 用配置文件创建对象

用法：`wuji_dl.factory.create_instance(BaseClass, config, *args, **kwargs)`

配置文件 yaml 语法：
```yaml
_name: package.Class
argument1: value1
argument2: value2
...
```

BaseClass: 限定 package.Class 必须是 BaseClass 的一个子类。
*args: 位置参数
**kwargs: 默认字典参数

如果基类继承了 `wuji_dl.factory.Factory` 基类，则 `_name` 也可以事先注册，方法如下：

```python
@BaseClass.register("subclass_name")
class SubClass(BaseClass):
    ...
```

如果 `SubClass` 所在包 `package` 事先引过了（自动引用功能待实现），则可以用 `subclass_name` 代替 `package.SubClass` 。

## 配置文件的解析

配置文件支持继承，由 `_base` 列表指定。`_base` 列表下的配置会和本文件的配置合并，如果有相同的键，出现迟的覆盖出现早的。

采用的合并策略是 DFS ，因此菱形继承的覆盖顺序不合预期。例如 B, C 都继承 A ，B 覆写了 A 的一项配置，而 C 没有。那么继承 B, C 的 D 将会采取 C 的配置，而不是 B 覆写的配置，而 C 的配置是原样从 A 那里继承过来的。

配置文件按 omegaconf 语法解析，所以支持引用一些变量之类的。特别地，数据路径等变量应该存放在 `configs/env.yaml` 中，将代码和机器具体信息隔离开。

## 配置文件的基本构成

### task

一个字典，用于创建 `LightningModule` 类型的对象。实现上自由发挥。

### data

一个字典，用于创建 `LightningDataModule` 类型的对象。

有一个已经写好的类 `wuji_dl.data.DefaultDataModule` ，其对于 `train/val/test/predict` 四类行为分别创建一个 `wuji_dl.data.BaseDataset` 的实例。

`wuji_dl.data.BaseDataset` 是一个抽象基类。在 `torch.utils.data.Dataset` 的基础上，还需要实现一个 `dataloader()` 方法来返回一个 `torch.utils.data.DataLoader` 的实例。

`wuji_dl.data.DefaultDataModule` 接受以下参数，分别为一个字典，默认为空字典：
- `train/val/test/predict` 用于创建四类数据集的实例。
- `common` 用于合并存放以上四个字典的公共键值。以上四个字典的新键值会覆盖 `common` 的已有键值。

Dataset 的实现自由发挥。

### callbacks

有两种组织形式：字典的列表，或字典的字典。

采用字典的字典是为了继承的方便。在合并两个配置的时候，字典的不同键互不干扰，而列表就只能丢弃前面那个配置了。对于字典的字典，将丢弃所有的键，取所有的值，转化成字典的列表。

列表每项用于创建一个 `lightning.Callback` 类型的对象。具体有哪些 callback 看 lightning 文档。

### trainer

传给 `lightning.Trainer` 的参数。具体参数怎么写看 lightning 文档。

### 其它功能列表

- `wuji_dl.metrics` 实现了一些 metric ，用法如 torchmetrics 。
- `wuji_dl.ops` 存放一些模型之间共用的层。
- `wuji_dl.ops.functional` 存放一些模型之间共用的函数。
- `wuji_dl.optim` 用来配置 Optimizer 和 LRScheduler 。
- `wuji_dl.display` 用来辅助 logging 的显示。
- `wuji_dl.slicing` 只是用来切分数据的，不知道放哪里就放根目录了。

## wuji-preprocess: 睡眠数据集的预处理框架

### 命令行用法
```
SYNOPSIS
    wuji-preprocess CONFIG_PATH <flags>

POSITIONAL ARGUMENTS
    CONFIG_PATH
        配置文件路径

FLAGS
    -s, --src_root=SRC_ROOT
        指定输入根目录，默认由配置文件的 _src 项指定
    -d, --dst_root=DST_ROOT
        指定输出根目录，默认由配置文件的 _dst 项指定
    -n, --num=NUM
        只处理前 NUM 条数据，默认处理全部
    -l, --log=LOG
        日志输出到名为 LOG 的文件下，默认输出到控制台
    -j, --jobs=JOBS
        用 JOBS 个进程同时预处理，默认为 1
```

### 配置文件格式
```
_src: 输入目录
_dst: 输出目录
_task: task collector 的类名
key1: worker1
key2: worker2
...
```

每条数据会被处理为一个 npz 文件，npz 文件的键为 `key1, key2, ...` ，值分别由 `worker1, worker2, ...` 产生。

### task collector

基类：`wuji_preprocess.base.BaseTaskCollector` ；预设见 `wuji_preprocess.task_collectors` 。

指定 task collector 的类名以后，构造函数是没有参数的。

用于指定输入、输出目录的格式。有两种预设：

- `bcg`: 输入根目录的每个子目录包括一条数据的特征。每条数据输出到输出根目录下的一个 npz 文件。
- `hsp`: 输入根目录的每个子目录表示一个被试，被试目录的每个子目录包括一条数据的特征。每个被试对应输出根目录下的一个子目录，每条数据输出到被试目录下的一个 npz 文件。

### worker

基类：`wuji_preprocess.base.BaseWorker` ；预设见 `wuji_preprocess.workers` 。

每个 worker 用于处理数据的某种特征。用 `wuji_dl.factory.create_instance` 实例化。

`__call__` 方法接受输入 `state` ，为存储输入和中转信息的字典，返回一个可以存储进 npz 的值或数组。

#### 输入信息

输入信息在 pipeline 开头初始化。

- base_dir: 存放数据特征的目录。

#### 中转信息

中转信息由 worker 产生和调用。

- duration: 数据的时长，单位为秒。

### wuji_preprocess.utils

放了一些用来操作 state 的函数。

- `match_first(state, pattern: str)` 从 base_dir 下匹配形如 `pattern` 的文件，有多个取字典序最前的。模式匹配采取 Unix shell 的语法。
- `update_duration(state, duration: float)` 表示当前数据的更新为 `duration` 秒，多次更新取最短的一次。

## wuji-index: 睡眠数据集索引的构建

### 命令行用法
```
SYNOPSIS
    wuji-index CONFIG_PATH

POSITIONAL ARGUMENTS
    CONFIG_PATH
        配置文件路径
```

### 配置文件格式
```
- builder1
- builder2
...
```

从空的 `pandas.DataFrame` 开始，按顺序执行一遍所有的 builder ，前一个的输出是后一个的输入。每个 builder 都是用 `wuji_dl.factory.create_instance` 实例化的。

### builder

基类：`wuji_index.base.BaseIndexBuilder` ；预设见 `wuji_preprocess.builders` 。

无论是初始化、拆分还是保存，都可以抽象为一个输入输出均为 `pandas.DataFrame` 的操作，见预设。
