---
title: vLLM 学习地图与版本边界
description: 四周完成推理地基、可验证实验、V1 源码调用链和生产诊断
lesson:
  track: vLLM
  stage: 00 · 路线
  time: 20 分钟
  level: 初学者
  question: 怎样学习 vLLM 才不会停在会抄启动参数？
---

# vLLM 学习地图与版本边界

学习 vLLM 的目标不是记住所有 backend，而是能从负载出发，预测一次 step 的数据、缓存、进程和时间花在哪里。本路线按每周 5–7 小时设计；有 GPU 就做真实服务，没有 GPU 也能完成公式、源码和结果分析。

## 当前版本边界

课程固定到 2026-07-15 的提交：

```text
vLLM commit: 61141ed265bfef41a0ca19e992567ea980919b96
主线: V1 online serving
入口: vllm serve
核心进程: API Server / EngineCore / GPU Worker / 可选 DP Coordinator
```

固定提交不是宣称它“最好”，而是让文件、函数和行为可以复查。vLLM 主分支变化很快；升级课程时至少要重新核对：

- `vllm/entrypoints/openai/api_server.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/executor/`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py` 与 `vllm/v1/worker/gpu/`

::: warning 官方架构页也可能同时出现历史类
当前官方 `Architecture Overview` 先介绍 V1 多进程架构，后面仍保留 `LLMEngine` / `AsyncLLMEngine` 的历史性说明。课程以 `vllm/v1` 实际路径为证据，不把两套调用链拼成一套。
:::

## 第 1 周：能计算，而不只是能描述

### Day 1–2：生成时间轴

学习[推理循环](../fundamentals/inference-loop)，完成：

```text
prompt tokens = 800
output tokens = 120
为什么不是一次 forward 结束？
prefill 产生多少个位置的 KV？
decode 第 37 步需要读取多少历史位置？
```

通过标准：能画出 prefill 与逐 token decode 的时间轴，并解释为什么 decode 往往更受权重/KV 读取带宽限制。

### Day 3–4：KV 显存账本

学习[KV Cache 与 PagedAttention](../fundamentals/kv-cache)，任选一个模型配置手算：

```text
KV bytes/token = 2 × layers × kv_heads × head_dim × dtype_bytes
```

再比较 MHA、GQA、MQA 的 `kv_heads` 差异。通过标准：输入上下文长度翻倍时，能判断权重显存不变而 KV 显存怎样变化。

### Day 5：性能指标

学习[批处理、延迟与吞吐](../fundamentals/performance)。给自己造三种负载：

- 短 prompt、长输出；
- 长 prompt、短输出；
- 多轮对话、共享长前缀。

分别预测 TTFT、ITL、prefix cache 的收益方向。

## 第 2 周：跑通一条可复现实验

先完成[第一台服务](../practice/first-server)，再做[基准测试与调参](../practice/benchmark)。实验只改变一个变量：

| 实验 | 自变量 | 至少记录 |
| --- | --- | --- |
| 并发扫描 | request rate / concurrency | TTFT p50/p99、ITL、throughput |
| token budget | `max_num_batched_tokens` | TTFT、ITL、preemption、GPU util |
| prefix reuse | cache on/off + shared prefix rate | cached tokens、TTFT、吞吐 |
| eager 对照 | optimization level / eager | 启动时间、稳态吞吐、显存 |

不要用单个请求判断吞吐，也不要只报平均延迟。排队系统的尾延迟往往先于平均值崩坏。

没有可用 GPU 时，保留完整命令和预期字段，使用官方 benchmark 示例结果练习分析；不要伪造实测数字。

## 第 3 周：沿一个 request id 读 V1

按顺序阅读：

1. [V1 多进程架构](../internals/architecture)
2. [一条请求的生命周期](../internals/request-lifecycle)
3. [Scheduler 与 KV Cache](../internals/scheduler-kv)
4. [Worker、Runner 与模型执行](../internals/model-execution)
5. [TP、PP、DP 与多节点](../internals/distributed)

每读一层，在笔记中填同一张表：

| 时刻 | 进程 | 对象 | 关键字段变化 | 下一个消费者 |
| --- | --- | --- | --- | --- |
| 请求接收 | API Server | HTTP body | messages → prompt/token ids | AsyncLLM |
| 核心入队 | EngineCore | Request | waiting、computed tokens | Scheduler |
| 本轮调度 | EngineCore | SchedulerOutput | token 数、block ids | Executor |
| 模型执行 | Worker | ModelRunnerOutput | sampled token ids | Scheduler |
| 文本流出 | API Server | RequestOutput | token ids → text delta | client |

通过标准：不用搜索类名，也能从 API handler 手动走到 `Scheduler.update_from_output()`。

## 第 4 周：从指标反推系统

学习[高级特性](../advanced/features)和[生产诊断](../advanced/production)，完成一份容量说明：

```text
模型与精度：
GPU 型号/数量/互联：
输入长度分布：
输出长度分布：
到达率与突发：
SLO（TTFT/ITL/E2E）：
可接受的排队长度：
显存边界：权重 / KV / activation / graph / workspace
瓶颈证据：
下一项最小实验：
回退条件：
```

参数不是结论。每个调参建议必须带“改善哪个指标、可能伤害哪个指标、怎样证伪”。

## 三条可选路径

### 只想部署服务

读 01 地基 → 第一次服务 → benchmark → architecture → production。源码细节只追到能定位日志和指标。

### 想改 Scheduler / Cache

完整读地基和源码主线；再从 `SchedulerOutput`、`KVCacheBlocks`、请求状态机与单元测试开始。不要第一步就改 attention kernel。

### 想做 kernel / 模型后端

先完成 model execution；补齐 CUDA execution model、Triton、`torch.compile`、attention 数据布局，再进入具体 backend。这里不把“会看 Python 调用链”冒充成“会优化 GPU kernel”。

## 最终检查

- 能手算 KV bytes/token；
- 能解释一个 step 的 token budget 如何在请求间分配；
- 能区分逻辑 block、物理 block、CUDA thread block；
- 能列出 TP=4、DP=1 的进程数量；
- 能给 TTFT 高但 ITL 正常提出三个可验证假设；
- 能在固定提交中指出请求入队、调度、分配缓存、执行、更新和流式输出的位置。

达到这些标准后，再追新版本变更会快很多，因为你比较的是模型边界，而不是重新背目录。
