# Loggers README

`utils/loggers` 的目标是把训练流程与具体日志后端解耦。

- 训练代码只负责“记录什么”
- 日志后端负责“写到哪里”

当前目录位于：[utils/loggers](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers)

## 目录结构

- [__init__.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers/__init__.py)
  对外暴露统一入口 `build_experiment_logger`
- [base.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers/base.py)
  定义日志接口基类 `BaseExperimentLogger`
- [builder.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers/builder.py)
  根据配置构造具体日志器
- [null_logger.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers/null_logger.py)
  空日志器，不做任何记录
- [tensorboard_logger.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers/tensorboard_logger.py)
  TensorBoard 日志实现

## 设计思路

日志系统采用“统一接口 + 可替换后端”的模式。

训练代码只调用这些统一方法：

- `log_scalar`
- `log_scalars`
- `log_histogram`
- `log_text`
- `close`

训练代码不直接依赖 `SummaryWriter`、CSV、WandB 等具体实现。

## 各文件作用

### `base.py`

定义统一日志接口：

```python
class BaseExperimentLogger:
    def log_scalar(self, tag, value, step):
        return None
```

作用：

- 规定日志器最小能力
- 保证训练代码与具体后端分离

### `null_logger.py`

空实现，所有方法都不做事。

作用：

- 当不需要记录日志时使用
- 避免训练代码里出现大量 `if logger is not None`

适用场景：

- 快速调试
- 不想产生日志文件
- 不安装 TensorBoard 依赖时退化运行

### `tensorboard_logger.py`

TensorBoard 后端。内部真正使用的是：

```python
from torch.utils.tensorboard import SummaryWriter
```

说明：

- `SummaryWriter` 才是 TensorBoard 的写入器
- 它只应该出现在 TensorBoard 后端里
- 训练主逻辑不应该直接 import 它

作用：

- 将标量、文本、直方图写入 TensorBoard 日志目录
- 供浏览器中的 TensorBoard 页面可视化

### `builder.py`

统一工厂入口：

```python
backend = args.get("experiment_logger", "tensorboard")
```

作用：

- 根据配置选择日志后端
- 当前支持：
  - `tensorboard`
  - `none`

使用方式：

```python
logger = build_experiment_logger(args)
```

### `__init__.py`

对外统一导出：

```python
from .builder import build_experiment_logger
```

这样外部模块只需要：

```python
from utils.loggers import build_experiment_logger
```

## 当前项目中的接入方式

在 [MiN.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/models/MiN.py) 中：

```python
from utils.loggers import build_experiment_logger
```

初始化：

```python
self.experiment_logger = build_experiment_logger(args)
```

记录配置：

```python
self.experiment_logger.log_text("config", self._format_args())
```

记录训练损失：

```python
self.experiment_logger.log_scalar("train/batch_loss", loss.item(), self.global_train_step)
```

记录评估精度：

```python
self.experiment_logger.log_scalar("eval/task_acc", float(eval_res['all_class_accy'] * 100.0), self.global_eval_step)
```

训练结束关闭：

```python
self.experiment_logger.close()
```

## 如何启用 TensorBoard

配置里默认后端就是 TensorBoard：

```json
"experiment_logger": "tensorboard"
```

如果环境里没有安装 TensorBoard，请先安装：

```bash
pip install tensorboard
```

启动训练后，日志会写入每次实验目录下的 TensorBoard 子目录。

再启动可视化：

```bash
tensorboard --logdir logs --port 6006
```

浏览器打开：

[http://localhost:6006](http://localhost:6006)

## 如何关闭日志

如果不想记录实验日志，在配置中改成：

```json
"experiment_logger": "none"
```

这时 `builder.py` 会返回 `NullLogger`，训练代码无需修改。

## 推荐配置项

建议在模型配置 JSON 中增加这些字段：

```json
"experiment_logger": "tensorboard",
"tensorboard_subdir": "tensorboard",
"tensorboard_flush_secs": 10,
"logger_histograms": false
```

含义：

- `experiment_logger`
  选择日志后端
- `tensorboard_subdir`
  TensorBoard 子目录名称
- `tensorboard_flush_secs`
  刷盘间隔
- `logger_histograms`
  是否记录直方图

## 如何新增新的日志后端

如果以后想接入 CSV 或 WandB，推荐步骤如下：

1. 在 `utils/loggers/` 下新增后端文件  
   例如：
   - `csv_logger.py`
   - `wandb_logger.py`

2. 继承 `BaseExperimentLogger` 并实现接口

```python
from .base import BaseExperimentLogger

class CSVLogger(BaseExperimentLogger):
    def log_scalar(self, tag, value, step):
        ...
```

3. 在 [builder.py](/home/luxiaogen/workspace/code/papers_poj/others/MiN-NeurIPS2025/MiN/utils/loggers/builder.py) 中注册

```python
if backend == "csv":
    return CSVLogger(...)
```

这样训练代码无需修改。

## 推荐使用原则

- 训练代码只调用统一日志接口
- 不在训练代码中直接 import `SummaryWriter`
- TensorBoard 细节只放在 `tensorboard_logger.py`
- 需要禁用日志时优先切换到 `NullLogger`
- 新增后端时只改 `utils/loggers`，不要改业务训练逻辑

## 一句话总结

`utils/loggers` 的作用是把“记录什么”和“写到哪里”分开：

- 训练代码只负责记录指标
- 各个日志后端负责落盘和可视化

这样可以保证日志系统可替换、可扩展、可复用。
