---
title: 分布式训练学习地图与版本边界
description: 以单卡数值基线、通信微实验、逐维扩展和故障演练组织 FSDP2、TorchTitan 与 Megatron 学习
lesson:
  track: 分布式训练
  stage: 00 · 路线
  time: 30–40 分钟
  level: 全阶段
  question: 怎样避免从巨型集群脚本开始，并让每次扩展都保持数值与性能可解释？
---

# 分布式训练学习地图与版本边界

可靠顺序是：**单进程数值基线 → 两卡 collective/等价性 → 单节点策略 → 多节点拓扑 → 多维组合 → 故障/恢复演练。**跳过前几层后，loss 错位、hang、OOM 与网络问题会叠在一起。

## 两条路线

### 路线 A：从 DDP 到 FSDP2

1. [进程、拓扑与集合通信](../fundamentals/collectives)
2. [训练显存账本](../fundamentals/memory)
3. [DDP 与 DeepSpeed ZeRO](../data-parallel/ddp-zero)
4. [FSDP2、DTensor 与 DeviceMesh](../data-parallel/fsdp2)
5. [FSDP2 与 Megatron 实验](../practice/first-runs)
6. [TorchTitan 源码主线](../internals/torchtitan)
7. [Checkpoint](../practice/checkpointing)与[排障](../practice/debugging)

适合 Transformers/PyTorch 原生模型、先做数据并行/状态分片的读者。

### 路线 B：Megatron 多维并行

1. 先完成共同地基；
2. [Megatron Core 总体设计](../model-parallel/megatron)
3. [TP/SP](../model-parallel/tensor-sequence)
4. [PP](../model-parallel/pipeline)
5. [CP/EP](../model-parallel/context-expert)
6. [多维组合](../model-parallel/multidimensional)
7. [Megatron 源码主线](../internals/megatron-flow)
8. Checkpoint 与排障。

适合单个模型/序列已不能靠普通 state sharding 高效扩展，需要模型结构感知并行的读者。

## 固定版本

| 项目 | 提交 | 学习主线 |
| --- | --- | --- |
| Megatron-LM/Core | [`82e9dc6`](https://github.com/NVIDIA/Megatron-LM/tree/82e9dc69c9e6f8c27681f2cb6856a188187edf6b) | TP/PP/CP/EP、distributed optimizer、训练 loop/checkpoint |
| TorchTitan | [`fec3e19`](https://github.com/pytorch/torchtitan/tree/fec3e196a4ceb87bfc87fb4f1a36a538d7e98ee4) | PyTorch FSDP2/DTensor/DeviceMesh 与多维并行参考 |
| DeepSpeed | [`53a2ac4`](https://github.com/deepspeedai/DeepSpeed/tree/53a2ac44fb664bea838df3981ba4366b91643070) | ZeRO stages/offload/config/checkpoint |

这些仓库和 PyTorch/NCCL/CUDA/Transformer Engine 强耦合。复现固定源码时使用项目声明的容器/依赖组合，不把你系统里的最新 nightly 任意拼上。

## 六道门禁

```mermaid
flowchart LR
    A[single-rank two-step] --> B[2-rank collective]
    B --> C[DDP numerical equivalence]
    C --> D[state-shard single node]
    D --> E[multi-node network]
    E --> F[multidimensional topology]
    F --> G[save-resume-reshard]
```

| 门禁 | 通过证据 |
| --- | --- |
| 单卡 | 固定 batch 两步 loss/grad/parameter checksum |
| collective | all-reduce/all-gather/all-to-all 正确，带宽/延迟基线 |
| DDP | 相同 global batch 下 loss/update 与单卡在容差内 |
| state shard | 每卡 HBM 符合账本，保存/恢复一致 |
| 多节点 | NCCL tests、接口/拓扑、无 oversubscription |
| 多维 | rank coordinates/groups、local tensor shapes、通信 trace |
| 恢复 | 不同 world/degree（支持范围内）恢复且下一步一致 |

## 每次实验只改一维

```text
Run A: DP=1 TP=1 PP=1 CP=1
Run B: DP=2, other degrees same
Run C: FSDP shard DP=2
Run D: TP=2, adjust DP to keep world fixed
Run E: PP=2, microbatch schedule fixed
```

保持 model config、global batch、tokens/update、data order、precision 和 checkpoint 起点；若策略要求改变其中一项，明确记录，不能再声称严格数值等价。

## 三种“正确”

1. **Shape/layout 正确**：每个 local tensor 与 collective contract 匹配；
2. **数值正确**：loss、grad、update 在明确容差内；
3. **训练语义正确**：global batch、sample order、loss normalization、optimizer schedule 一致。

只通过 shape 不代表数值正确；多卡 loss 看起来接近也可能 global batch 已改变。

## 建议实验记录

```text
code/container/image digest:
model/data/tokenizer revisions:
nodes × GPUs and topology:
rank → hostname/local_rank/device:
mesh names/sizes and group members:
global/local tensor layouts:
micro/global batch and sequence:
precision + checkpoint/recompute:
optimizer and LR schedule:
loss/grad/update checksums:
step time breakdown + TFLOP/s + HBM:
collective trace/profile:
checkpoint format and resume result:
```

## 第一遍不做什么

- 不从 64 GPU 多维配置开始；先两卡验证一维。
- 不把所有 collective 都叫“同步”；记录 op、group、shape、bytes。
- 不只看平均 GPU utilization；看每 rank step、straggler 与通信等待。
- 不假设框架能任意组合所有 degrees；先查支持矩阵与断言。
- 不以一次成功 resume 证明 checkpoint 健壮；还要故障中断、不同 topology 与损坏检测。

## 一周路线

| 日 | 任务 | 产出 |
| --- | --- | --- |
| 1 | collective + memory ledger | 两卡通信表、单卡显存账 |
| 2 | DDP/ZeRO/FSDP2 | 数值与 HBM 对照 |
| 3 | TP/SP | linear layer shard 推导 |
| 4 | PP | schedule/bubble 时间线 |
| 5 | CP/EP | sequence/expert dispatch 图 |
| 6 | TorchTitan/Megatron trace | two-step 源码 trace |
| 7 | checkpoint + failures | restore matrix、hang/OOM runbook |

## 通关标准

你应能从一个模型瓶颈选择切分维度，写出每次实验的单一变量；区分 layout/数值/训练语义正确；在引入新并行维度前给出门禁与回滚配置。

不确定选哪个框架时先读[框架选择地图](./decision-map)。
