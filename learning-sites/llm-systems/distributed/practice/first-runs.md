---
title: FSDP2 与 Megatron 第一组可验证实验
description: 用固定 global batch、两步数值指纹和逐维扩展完成 single、FSDP2、TP、PP 基线
lesson:
  track: 分布式训练
  stage: 04 · 实验与生产
  time: 150–240 分钟
  level: 有 1–2 张可用 GPU
  question: 怎样证明一个分布式配置真的保持了训练语义，而不是仅仅“没有报错”？
---

# FSDP2 与 Megatron 第一组可验证实验

这一课不追求大模型吞吐。目标是建立一套以后扩到多节点仍可复用的门禁：**同一初始化、同一数据顺序、同一 global tokens/update，先跑单 rank，再一次只打开一个并行维度。**

## 先定义成功

每个 run 至少保存：

```text
resolved config + code commit
rank → host/local_rank/device/mesh coordinate
first 3 losses
selected parameter/gradient/update checksums
global valid tokens per step
peak allocated/reserved HBM
step time: data / fwd-bwd / collective / optimizer
checkpoint path + resume next loss
```

“进程 exit 0”只证明控制流结束，不证明数据没有重复、loss 没有多除一次、参数真的更新或 checkpoint 可恢复。

## 环境门禁

固定源码和运行环境：

```bash
git rev-parse HEAD
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
print("gpus", torch.cuda.device_count())
PY
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
```

TorchTitan 固定课程提交是 [`fec3e19`](https://github.com/pytorch/torchtitan/tree/fec3e196a4ceb87bfc87fb4f1a36a538d7e98ee4)，Megatron-LM 是 [`82e9dc6`](https://github.com/NVIDIA/Megatron-LM/tree/82e9dc69c9e6f8c27681f2cb6856a188187edf6b)。使用各自锁定/声明的 PyTorch、CUDA、NCCL、Transformer Engine 组合；不要用一个碰巧能 import 的环境冒充可复现环境。

### 通信先于模型

两卡至少验证 all-reduce 与点对点，并记录带宽。优先使用安装环境中的 NCCL Tests；若它都不稳定，不进入训练框架。`nvidia-smi topo -m` 还要确认两卡/网卡真实路径。

## TorchTitan：先只解析配置

固定源码提供 config manager，可在分配 GPU 前看最终值：

```bash
python -m torchtitan.config.manager \
  --module llama3 \
  --config llama3_debugmodel \
  --training.steps 3 \
  --training.seq_len 512 \
  --training.global_batch_size 16
```

检查输出中的 six degrees、local/global batch、dtype、checkpoint/output path。布尔项使用 `--foo` / `--no_foo` 形式，而不是猜 `--foo=False`。官方固定说明见 [`docs/debugging.md`](https://github.com/pytorch/torchtitan/blob/fec3e196a4ceb87bfc87fb4f1a36a538d7e98ee4/docs/debugging.md)。

## Run A：单 GPU 数值基线

在 TorchTitan 仓库根目录：

```bash
NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --training.steps 3 \
  --training.seq_len 512 \
  --training.local_batch_size 8 \
  --training.global_batch_size 16 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.expert_parallel_degree 1 \
  --dump_folder outputs/lab-single
```

这里 global batch=16、local batch=8，所以每次 update 累积 2 个 local batches。保存 run 的 resolved config、loss、有效 token 数和 checkpoint；它是后续对照，不要跑完就删。

若当前固定提交的 CLI 使用连字符而非下划线，先以 config manager `--help` 为准；两种拼写支持范围随 Tyro/配置层版本变化。

## Run B：两卡 FSDP2

只改变 world 与 DP shard：

```bash
NGPU=2 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --training.steps 3 \
  --training.seq_len 512 \
  --training.local_batch_size 8 \
  --training.global_batch_size 16 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.expert_parallel_degree 1 \
  --dump_folder outputs/lab-fsdp2
```

此时 batch degree=2，每 rank local batch=8，gradient accumulation=1，global batch 仍是 16。比较：

- parameter global shape 相同、local shard 约减半；
- global valid tokens/update 相同；
- loss/update 在明确 BF16 容差内；
- FSDP pre-forward all-gather、backward reduce-scatter 出现；
- peak HBM 符合参数 materialization 时间线，而非简单恰好减半。

初始化路径可能随 shard layout 产生与单卡不同的随机流。做严格比较时按 TorchTitan [`docs/debugging.md`](https://github.com/pytorch/torchtitan/blob/fec3e196a4ceb87bfc87fb4f1a36a538d7e98ee4/docs/debugging.md) 创建并加载 seed checkpoint，不能仅设置同一个整数 seed 就宣称参数逐元素相同。

## Run C：两卡 TP

仍用两卡，但把 DP shard 退回 1、只开 TP=2：

```bash
NGPU=2 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --training.steps 3 \
  --training.seq_len 512 \
  --training.local_batch_size 8 \
  --training.global_batch_size 16 \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.context_parallel_degree 1 \
  --dump_folder outputs/lab-tp2
```

两个 TP ranks 合作处理同一 batch，所以 batch degree=1，仍需两次 gradient accumulation。验收重点变成 local linear/vocab shapes、TP/SP layouts、collective 和 loss，而不是样本是否分到两卡。

## Run D：两卡 PP

```bash
NGPU=2 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --training.steps 3 \
  --training.seq_len 512 \
  --training.local_batch_size 8 \
  --training.global_batch_size 16 \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 2 \
  --parallelism.context_parallel_degree 1 \
  --dump_folder outputs/lab-pp2
```

打印每 rank 的 layer range、first/last stage 标记与 schedule。检查末 stage loss 对所有 ranks 的可见/聚合逻辑；记录 microbatch P2P 次序。若当前 backend/配置对 PP-only 有约束，使用固定 integration test 中支持的 backend 组合，而不是同时再开 TP/FSDP 来掩盖问题。

## Megatron：最小 mock-data 基线

Megatron 参数面较大，第一遍建议从固定仓库的 functional test 配置或官方 example 改小。下面是**教学模板**，运行前先用当前提交 `pretrain_gpt.py --help` 核对选项，并以仓库环境为准：

```bash
torchrun --standalone --nproc_per_node=1 pretrain_gpt.py \
  --use-mcore-models \
  --transformer-impl local \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 1024 \
  --num-layers 4 \
  --hidden-size 256 \
  --ffn-hidden-size 1024 \
  --num-attention-heads 4 \
  --seq-length 256 \
  --max-position-embeddings 256 \
  --micro-batch-size 4 \
  --global-batch-size 8 \
  --train-iters 3 \
  --lr 1e-4 \
  --min-lr 1e-4 \
  --lr-decay-style constant \
  --lr-decay-iters 3 \
  --bf16 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --log-interval 1 \
  --eval-iters 0
```

再做 TP=2 时只改 `nproc_per_node=2`、`tensor-model-parallel-size=2`，保持 global batch/数据/模型不变并按要求启用 sequence parallel。PP=2 同理先恢复 TP=1，再改 PP；不要第一步直接复制 8B FP8 性能脚本。

可直接参考固定测试配置 [`gpt3_mcore_te_tp2_pp2_cp2_1node/model_config.yaml`](https://github.com/NVIDIA/Megatron-LM/blob/82e9dc69c9e6f8c27681f2cb6856a188187edf6b/tests/functional_tests/test_cases/gpt/gpt3_mcore_te_tp2_pp2_cp2_1node/model_config.yaml)，但它依赖项目测试数据/环境变量，不是脱离 harness 的一行命令。

## 数值比较不能只看 loss 小数

建议选 2–3 个参数做指纹：

```text
parameter logical FQN
global norm before step
global grad norm after backward
global norm after step
sum / squared-sum / selected slices
```

分片参数先按 logical tensor 语义汇总，不能比较不同 ranks 的 local shard checksum。BF16、collective order 和 fused kernels 可造成非 bitwise 差异；先确定单步容差，再观察差异是否随 steps 单调放大。

## 实验表

| Run | world | DP shard | TP | PP | global batch | 主要门禁 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 1 | 1 | 1 | 1 | 16 | reference |
| B | 2 | 2 | 1 | 1 | 16 | state shards/RS-AG |
| C | 2 | 1 | 2 | 1 | 16 | layer layouts/TP collectives |
| D | 2 | 1 | 1 | 2 | 16 | stages/P2P/schedule |

Run B/C/D 不是互相替代；它们分别隔离三个维度。全部通过后才做 `DP×TP` 或 `TP×PP`。

## 停止条件

出现以下任一项就不要扩卡：

- resolved world/degrees 与计划不一致；
- global valid tokens/update 不一致；
- 任一 rank 没有参数 update；
- 数值偏差无法被 precision/kernel 解释；
- 2-rank collective 不稳定；
- checkpoint 不能恢复下一步；
- peak HBM/step breakdown 没有记录。

## 通关标准

你应能交付四个 run bundle，证明 batch/token 语义一致；逐维解释 local tensor、collective 和 HBM 变化；用 seed checkpoint/逻辑 tensor checksum 做数值比较，并能在失败时回滚到最近的单维基线。

下一步沿这些实验进入[TorchTitan 源码主线](../internals/torchtitan)。
