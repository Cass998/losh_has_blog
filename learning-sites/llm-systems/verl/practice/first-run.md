---
title: 第一次可验证实验：目标只是一整个 step
description: 用固定源码、GSM8K、小模型和一组证据检查点，跑通从数据到 rollout、reward、advantage 与 actor update 的闭环
lesson:
  stage: 03 · 实验
  order: 09
  time: 60–120 分钟
  level: 动手入门
  question: 怎样证明自己真的跑通了一次 RL 训练，而不是只看到进程没有退出？
---

# 第一次可验证实验：目标只是一整个 step

第一次运行不追论文指标，也不追 GPU 利用率。唯一目标是取得一条完整证据链：**数据被正确读取 → 模型生成回答 → 奖励可解释 → advantage 有效 → actor 完成更新 → 新权重具备同步条件。**

如果只保留一张“终端还在跑”的截图，你无法判断它卡在加载、生成还是训练。下面每一步都有明确产出；即使机器暂时跑不下全链路，你也能知道已经验证到哪里。

::: warning 资源预期
官方 quickstart 建议使用容器和具备足够显存的 GPU。下面的单卡 0.5B 配置是**链路检查起点**，不是对所有 CUDA/vLLM/模型组合的显存承诺。OOM 时先保留已完成证据，再按本页末尾缩小规模；不要一次改十项。
:::

## 实验合同：开始前先写下不变量

在 `runs/smoke-001/README.md` 写：

```text
目标：完成 1 个 GRPO training step，不评价最终能力
源码：e5687fce0516d31e1fdc4580499074a9bd94c751
数据：GSM8K，检查 data_source/prompt/reward_model/extra_info
算法：GRPO，rollout.n=2，critic 自动关闭
成功：日志出现 rollout/reward/old_log_prob/adv/update_actor 的有限指标
停止：出现 NaN/Inf、空 response、全组零 advantage 或不可解释的截断
```

这是你的判断标准。没有它，程序跑完后很容易把任何输出解释成成功。

先建立实验目录：

```bash
mkdir -p runs/smoke-001
```

## 0. 固定代码与环境

在 veRL 仓库根目录执行：

```bash
git rev-parse HEAD
python -V
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_0", torch.cuda.get_device_name(0))
PY
```

期望第一行是本站绑定提交。若不是，不代表不能运行，但后面所有字段和配置都要按你的提交重新核对。把输出保存为 `runs/smoke-001/environment.txt`，同时记录容器镜像或依赖锁文件。

安装以[官方 veRL 安装文档](https://verl.readthedocs.io/en/latest/start/install.html)为准；不要在已有复杂训练环境里边报错边随机升级 torch、vLLM 和 flash-attn。版本组合是实验输入。

## 1. 只准备数据，不启动 Ray

固定源码自带 GSM8K 预处理脚本：

```bash
python examples/data_preprocess/gsm8k.py \
  --local_save_dir "$HOME/data/gsm8k"
```

它生成 `train.parquet` 与 `test.parquet`。先检查一行：

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={"train": "~/data/gsm8k/train.parquet"},
    split="train",
)
row = ds[0]
print("columns:", ds.column_names)
print("data_source:", row["data_source"])
print("prompt:", row["prompt"])
print("reward_model:", row["reward_model"])
print("extra_info keys:", sorted(row["extra_info"]))
PY
```

你要能解释四个字段：

| 字段 | 这次实验的职责 | 错了会怎样 |
| --- | --- | --- |
| `data_source` | 选择/路由评分逻辑与分组指标 | reward 函数可能走错分支 |
| `prompt` | chat message 列表，经模板变成 token | 模板或角色格式错误 |
| `reward_model.ground_truth` | GSM8K 最终答案 | 所有正确答案也得 0 |
| `extra_info` | index、原题、split 等元数据 | 调试与自定义 reward 缺上下文 |

**本阶段产出：** `data-sample.txt`。若这里不正确，不要启动 GPU 任务。

## 2. 选择一条“来源明确”的启动基线

固定提交中可用的 FSDP GRPO 示例是：

```text
examples/grpo_trainer/run_qwen3_4b_fsdp.sh
```

不要复制网上某段旧命令再混用。先阅读脚本的 `DATA/MODEL/ACTOR/ROLLOUT/REF/TRAINER` 六组参数。下面通过环境变量把它缩成单卡、0.5B、两候选的一步链路：

```bash
export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
export TRAIN_FILE="$HOME/data/gsm8k/train.parquet"
export TEST_FILE="$HOME/data/gsm8k/test.parquet"
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export ROLLOUT_N=2
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_GPU_MEM_UTIL=0.45
export TOTAL_EPOCHS=1
export SAVE_FREQ=-1
export TEST_FREQ=-1
export PROJECT_NAME=verl_learning_smoke
export EXPERIMENT_NAME=grpo_05b_one_step
```

这些值表达的是：一张 GPU、同题两份回答、短序列、小 batch、不存 checkpoint。它们不是推荐训练超参数。

## 3. 先让 Hydra 打印最终配置

在真正分配模型前，运行：

```bash
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  --cfg job --resolve \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  > runs/smoke-001/resolved-config.yaml
```

搜索并人工确认：

```bash
rg -n "adv_estimator:|use_v1:|trainer_mode:|train_batch_size:|n:|enable:|loss_mode:|model.path:" \
  runs/smoke-001/resolved-config.yaml
```

应至少满足：

- `trainer.use_v1: true`；
- `algorithm.adv_estimator: grpo`；
- rollout `n: 2`；
- critic 不被显式强制开启（`need_critic()` 对非 GAE 自动关闭）；
- reference 是否启用与你的 `actor.use_kl_loss` 一致；
- 模型和 Parquet 路径完全正确；
- 最终只有 1 个 training step，且关闭训练前 validation。

**本阶段产出：** 完整解析配置，而不是只保存 shell 命令。Hydra 的 defaults 与插值可能让两者不同。

## 4. 启动一步，并保留原始日志

```bash
set -o pipefail
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  trainer.rollout_data_dir=runs/smoke-001/rollouts \
  2>&1 | tee runs/smoke-001/train.log
```

不要因日志出现 “warning” 就判断失败。用阶段检查表：

| 证据点 | 你在证明什么 | 失败时先看 |
| --- | --- | --- |
| Ray 与 TaskRunnerV1 初始化 | 运行的是 V1 主链 | commit、resolved config、runtime env |
| actor/rollout/ref worker 建立 | 角色与资源可放置 | GPU 数、TP、模型加载、Ray 状态 |
| generation 有 response | 推理后端真正完成 rollout | token 长度、EOS、KV cache、server 日志 |
| reward 有有限值且非全异常 | 数据与评分契约连通 | ground truth、输出格式、reward 日志 |
| `old_log_prob` 完成 | 训练后端能对动作重算概率 | micro batch、token 长度、数值差异 |
| advantage 非 NaN 且并非所有组为 0 | GRPO 分组与 reward 有学习信号 | uid/index、完整组、组内 reward 方差 |
| actor update 完成 | 至少一次反向与 optimizer step | loss、grad norm、OOM、finite check |
| step 计数结束 | 闭环达到预期终止条件 | `total_training_steps` 是否生效 |

“reward 全是 0 但 actor update 没报错”只是程序链路成功，不是训练语义成功。把这两种结论分开写。

## 5. 做一次最小结果审计

从 rollout dump 任选一条，记录：

```text
prompt / uid：
response 原文：
response token 数：
ground truth：
reward 与理由：
同组其他 reward：
advantage 正负：
是否被截断：
```

再检查日志中的所有 loss/ratio/KL/grad 指标是否有限。你不需要第一步 reward 上升，但必须能解释每个观测值来自哪一阶段。

## 资源不够时按因果顺序缩小

一次只动一层，保留前后错误差异：

1. **模型与序列**：更小模型、降低 prompt/response 上限；
2. **单次前向峰值**：micro batch=1 或动态 batch 的 token 上限；
3. **rollout cache**：降低 `gpu_memory_utilization`、并发/候选数；
4. **训练状态**：gradient checkpointing、参数/优化器 offload；
5. **角色放置**：需要时从共置转向更多设备或分离模式。

降低 `train_batch_size` 不一定解决一次前向 OOM；降低 `rollout.n` 会改变 GRPO 组统计；缩短 response 会改变任务和 reward 分布。每个“省显存”按钮都可能有算法代价。

## 完成标准

你的 `runs/smoke-001/` 至少包含：

```text
README.md
environment.txt
data-sample.txt
resolved-config.yaml
train.log
rollouts/（若成功生成）
```

并能回答：最后一个成功阶段是什么、是否真的有非零学习信号、若换机器哪些变量最可能改变结果。

接下来读[读懂 Hydra 配置](./configuration)，把刚才的每个 override 放回配置树；然后读[写一个可靠奖励函数](./reward-function)，亲自为 reward 边界写测试。
