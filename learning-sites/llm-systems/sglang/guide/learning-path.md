---
title: SGLang 学习地图与版本边界
description: 先选学习路线，再进入六周源码计划，完成 Runtime 原理、调用链、实验和生产诊断
lesson:
  track: SGLang
  stage: 00 · 路线
  time: 20 分钟
  level: 初学者
  question: 怎样学习 SGLang 才不会只剩一堆启动参数？
---

# SGLang 学习地图与版本边界

学习目标不是“会启动 SGLang”，而是能从一组请求预测：哪里匹配前缀、怎样形成 batch、哪些 KV slot 被占用、哪个进程执行模型，以及延迟为什么改变。本页给出四周压缩路线；要逐函数阅读、做实验并留下可复查证据，请以[六周源码学习计划](./study-plan)为主线，每周完成源码问题、实验和验收物。

## 当前版本边界

```text
SGLang commit: c879f3da5ceaaef3cb197c4e59ce683d420ce96c
课程日期: 2026-07-16
主线: SGLang Runtime（SRT）文本生成服务
入口: python -m sglang.launch_server / sglang.Engine
核心对象: TokenizerManager / Scheduler / RadixCache / ModelRunner
```

主分支变化很快，固定提交是为了让链接和行为可复查，并不表示它是稳定发行版。升级课程时至少重查：

- `srt/entrypoints/engine.py` 与 `http_server.py`；
- `srt/managers/tokenizer_manager.py`、`scheduler.py`、`schedule_batch.py`；
- `srt/mem_cache/radix_cache.py` 与 `memory_pool.py`；
- `srt/model_executor/model_runner.py`；
- `srt/server_args.py` 的默认值和兼容约束。

::: warning 不混读旧架构
网上资料可能描述早期 `RuntimeEndpoint`、旧 frontend DSL 或已经改名的参数。设计思想可参考，进程数、默认后端和实际调用链必须以固定提交为准。
:::

## 第 1 周：先让缓存与调度可计算

### Day 1：推理时间轴

读[推理循环与进程边界](../fundamentals/runtime)，画出：

```text
请求 A: prompt 800, output 100
请求 B: prompt 200, output 20
哪一段是 extend/prefill？
哪一段逐步 decode？
为什么两个请求会在某一步进入同一 ScheduleBatch？
```

通过标准：能解释 Scheduler 是 GPU-owning 进程中的控制器，而不是一个独立于所有 worker 的全局服务。

### Day 2–3：Radix tree

读[RadixAttention](../fundamentals/radix-attention)。手工插入三条 token 序列：

```text
[1, 2, 3, 4]
[1, 2, 3, 8]
[1, 2, 9]
```

画出压缩边，标记查询 `[1,2,3,7]` 时的最长命中、分裂位置和未命中后缀。再解释 `lock_ref` 为什么必须沿祖先传播。

### Day 4–5：性能模型

读[调度、缓存与性能](../fundamentals/performance)，为三类负载预测瓶颈：

- 多轮对话，共享长 system prompt；
- 独立长文档问答，prompt 长而前缀少；
- 短 prompt、长 reasoning 输出。

通过标准：能区分 cache hit rate、TTFT、ITL 和 output throughput，不用一个“tokens/s”概括全部。

## 第 2 周：跑通并建立基线

完成[第一台服务](../practice/first-server)、[基准、指标与调参](../practice/benchmark)，再按[实验工作簿](../practice/lab-workbook)保存命令、原始结果和失败证据。第一轮只保留最小配置，第二轮每次只改一个变量：

| 实验 | 自变量 | 至少记录 |
| --- | --- | --- |
| 并发扫描 | request rate / concurrency | TTFT、ITL、E2E、throughput |
| radix 对照 | cache on/off + 共享前缀比例 | cache hit、TTFT、吞吐 |
| chunk 对照 | chunked prefill budget | 长请求 TTFT、短请求尾延迟 |
| overlap 对照 | overlap scheduler on/off | GPU 利用率、ITL、CPU 时间 |
| memory 对照 | `mem_fraction_static` | token capacity、OOM、吞吐 |

实验记录必须包含模型 revision、容器/包版本、GPU、命令、输入/输出长度分布和原始 JSON。没有 GPU 时可以阅读 benchmark 输出并写分析，但不能把预期数字写成实测。

## 第 3 周：沿一个 `rid` 读完源码

按顺序阅读：

1. [进程与通信架构](../internals/architecture)
2. [进程与消息逐跳追踪](../internals/message-flow)
3. [一条请求的生命周期](../internals/request-lifecycle)
4. [Scheduler 与 ScheduleBatch](../internals/scheduler)
5. [Chunked Prefill 源码状态机](../internals/chunked-prefill)
6. [RadixCache 与内存池](../internals/cache-pools)
7. [ModelRunner 与执行后端](../internals/model-execution)

始终维护同一张表：

| 时刻 | 进程 | 对象 | 关键字段变化 | 下一消费者 |
| --- | --- | --- | --- | --- |
| HTTP 接收 | main | `GenerateReqInput` | messages/text → request | TokenizerManager |
| token 化 | main | `TokenizedGenerateReqInput` | input ids、rid | Scheduler/DP controller |
| 前缀匹配 | scheduler rank | `Req` | `prefix_indices`、`last_node` | PrefillAdder |
| batch 构造 | scheduler rank | `ScheduleBatch` | pool indices、seq/extend lens | ModelRunner |
| 执行结果 | scheduler rank | output struct | sampled ids、logprobs | DetokenizerManager |
| 文本流出 | detokenizer/main | request state | decoded text delta | client |

通过标准：能从 `/generate` handler 手动追到 `ModelRunner.forward()`，再追回 TokenizerManager 的异步结果循环。

## 第 4 周：扩规模但保留因果关系

先用[并行、PD、HiCache 与 Ray](../advanced/distributed)建立选择边界，再分别深读[PD 与 HiCache 的 KV 状态机](../advanced/pd-hicache)、[结构化输出与 RL 接入](../advanced/features)、[RL rollout 生命周期](../advanced/rl-lifecycle)和[生产诊断](../advanced/production)，完成一份容量说明：

```text
模型、精度与 revision：
GPU 数量、互联与跨节点网络：
输入/输出长度分布：
共享前缀比例：
到达率、突发与优先级：
SLO（TTFT / ITL / E2E）：
权重 / KV / graph / workspace 显存：
当前瓶颈证据：
下一项单变量实验：
失败时回退条件：
```

只有在单实例基线已解释清楚后，再判断是 TP、DP、PD 分离还是 HiCache。它们解决不同约束，不是“卡越多越快”的同义词。

## 三条压缩路线

### 只想部署

地基三课 → 第一次服务 → benchmark → architecture → production。源码追到能定位指标与故障边界即可。

### 想改 Scheduler / Cache

完整读地基与源码主线，重点手工模拟 `Req`、`ScheduleBatch`、`PrefillAdder`、`RadixCache` 的状态变化，再从已有单元测试进入。

### 想做 kernel / backend

先读 model execution，补齐 attention 的 Q/K/V 布局、prefill/decode kernel 形态、CUDA Graph 和分布式 collective，再进入具体 backend。不要把能追 Python 调用栈误认为会写 kernel。

## 最终检查

- 能画出一个压缩 radix tree 并算最长前缀；
- 能解释两级 pool 和 radix tree 分别存什么；
- 能列出 `dp=1,tp=4,pp=1` 的 scheduler rank 数量；
- 能解释默认 overlap loop 与普通 loop 的先后关系；
- 能给 TTFT 高但 ITL 正常提出三个可证伪假设；
- 能从固定源码指出请求接收、prefix match、batch、forward、detokenize 的位置。

达到这些标准，再追新版特性时是在比较边界，不是重新背目录。
