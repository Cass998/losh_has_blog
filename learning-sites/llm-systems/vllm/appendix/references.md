---
title: vLLM 源码、论文与术语
description: 固定源码索引、官方设计文档、核心论文、术语对照与版本核验方法
lesson:
  track: vLLM
  stage: 04 · 索引
  time: 按需查阅
  level: 全阶段
  question: 遇到文档、博客与本地行为冲突时，应该相信哪一层证据？
---

# vLLM 源码、论文与术语

本课程的解释对象固定为 vLLM 提交 [`61141ed`](https://github.com/vllm-project/vllm/tree/61141ed265bfef41a0ca19e992567ea980919b96)。在线 stable 文档会继续演进；安装的 PyPI 版本也可能早于或晚于课程提交。先核对版本，再判断“源码不一致”究竟是理解错误还是版本差异。

## 证据优先级

遇到冲突时按问题类型选择：

1. **正在运行的行为**：实际版本源码、启动最终配置、日志、最小实验；
2. **该提交的设计意图**：同提交 `docs/`、测试、PR/issue 讨论；
3. **当前推荐用法**：[官方 stable 文档](https://docs.vllm.ai/en/stable/)与 release notes；
4. **原始算法**：论文，用于理解机制而非推断当前 CLI；
5. **博客/视频**：建立直觉，必须回源核验版本。

论文不会告诉你 2026 年某个参数的默认值；最新文档也不保证描述旧 wheel 的内部类名。

## 固定源码主线

| 主题 | 源码入口 |
| --- | --- |
| CLI / server startup | [`entrypoints/cli/serve.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/cli/serve.py)、[`api_server.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/openai/api_server.py#L746) |
| Chat route / renderer | [`chat_completion/api_router.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/openai/chat_completion/api_router.py#L53)、[`serving.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/openai/chat_completion/serving.py#L233) |
| Frontend async engine | [`AsyncLLM`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/async_llm.py#L70) |
| API ↔ Core client | [`core_client.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/core_client.py#L467) |
| Core loop | [`EngineCore`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/core.py#L97)、[`run_busy_loop`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/core.py#L1326) |
| Scheduler | [`scheduler.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/sched/scheduler.py#L417) |
| KV request manager | [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/kv_cache_manager.py#L114) |
| Physical block metadata | [`block_pool.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/block_pool.py#L143) |
| Executor interface / backends | [`executor/abstract.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/executor/abstract.py)、[`executor/`](https://github.com/vllm-project/vllm/tree/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/executor) |
| GPU Worker | [`gpu_worker.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/worker/gpu_worker.py#L130) |
| ModelRunner V1 / V2 | [`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/worker/gpu_model_runner.py#L446)、[`gpu/model_runner.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/worker/gpu/model_runner.py#L120) |
| Metrics | [`v1/metrics/loggers.py`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/metrics/loggers.py) |

## 官方设计与使用文档

这些链接锁定到课程提交，适合与源码同步阅读：

- [Architecture Overview](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/design/arch_overview.md)
- [PagedAttention kernel 说明](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/design/paged_attention.md) — 文档自身标明是历史实现，不能当当前 kernel 调用栈
- [Prefix Caching Design](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/design/prefix_caching.md)
- [Metrics Design](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/design/metrics.md)
- [`torch.compile` Integration](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/design/torch_compile.md)
- [CUDA Graph Modes](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/design/cuda_graphs.md)
- [Optimization and Tuning](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/configuration/optimization.md)
- [Parallelism and Scaling](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/serving/parallelism_scaling.md)
- [Production Metrics](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/usage/metrics.md)
- [Benchmark CLI](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/benchmarking/cli.md)
- [Security](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/usage/security.md)

## 核心论文

### 系统主线

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM/PagedAttention，重点读内存碎片、分页映射、共享与实验 workload。
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling/continuous batching 的重要背景。
- [FasterTransformer: Enabling Fast Large Scale Transformer Inference](https://arxiv.org/abs/2206.01062) — kernel 与并行推理背景；不要据此推断 vLLM 当前实现。

### 调度与解耦

- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill、stall-free scheduling 的相关系统讨论。
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) — goodput 与 prefill/decode 解耦。
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — prefill/decode 资源特性与 phase splitting。

### Speculative decoding

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)

论文阅读只问三件事：假设的 workload 是什么；机制减少了哪类成本；实验的 latency/throughput/quality 口径是什么。然后回到当前源码看哪些假设仍成立。

## Ray 官方资料

- [Ray Core Key Concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)
- [Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Logical Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)
- [Ray Observability](https://docs.ray.io/en/latest/ray-observability/index.html)

只学 vLLM 所需最小集合：actor 生命周期、资源声明、placement group/gang scheduling、State API 和 runtime environment。Ray Data、Serve、Train 是更上层产品，不是理解 `RayDistributedExecutor` 的前置条件。

## 术语对照

| 术语 | 本课程中的精确定义 |
| --- | --- |
| prefill / context | 对尚未计算的一段输入上下文做 forward、产生 KV；V1 Scheduler 内部不一定用独立 phase 状态机 |
| decode / generation | 在已有上下文后逐步产生输出 token 的阶段/工作类型 |
| continuous batching | 每次 engine iteration 可加入、保留或移除请求，而非等待整批都结束 |
| token budget | 一个 Scheduler step 最多安排的 token 数，不等于请求数 |
| KV block | 一组 token positions 的 KV 分配/寻址单位，不是 Transformer layer 或 CUDA thread block |
| block table | 某请求 logical blocks 到 physical block ids 的映射 |
| slot mapping | 本轮 token position 写入物理 KV tensor slot 的映射 |
| prefix cache hit | 找到具有相同链式 hash/extra identity 的已计算前缀 KV blocks |
| preemption | 为释放 KV 容量暂停请求；V1 常通过之后 recompute 恢复 |
| TTFT | 请求到第一个输出 token/chunk 的时间，含排队、prefill 与前后端开销 |
| ITL | 相邻输出 token/chunk 的间隔分布 |
| TPOT | 通常为首 token 后生成时长除以后续输出 token 数；与 ITL 相关但不完全相同 |
| goodput | 在明确 SLO 内成功完成的请求/工作量速率 |
| EngineCore | 每 DP rank 的集中调度/KV 元数据与执行协调进程 |
| Executor | Core 调用 worker 的统一后端接口，可由本地、multiprocessing 或 Ray 实现 |
| Ray actor | 被 Ray 调度的有状态专属 Python process；vLLM 用它承载长期 GPU Worker |
| placement group | 原子预留多个 resource bundles 的 Ray 机制，支持 gang scheduling |

## 版本核验清单

```bash
vllm --version
python - <<'PY'
import inspect
import vllm
from vllm.v1.core.sched.scheduler import Scheduler

print(vllm.__version__)
print(inspect.getsourcefile(Scheduler))
PY

vllm serve --help=all > vllm-help.txt
```

同时保存：容器 digest、PyTorch/CUDA/driver、model revision、完整启动命令和启动日志。若从源码安装，保存 git commit 与 dirty status。

## 从课程毕业的四个作品

1. 一张自己的 V1 进程/所有权图；
2. 一个能复现 request → Scheduler → block → Runner → SSE 的三-token trace；
3. 一份 workload contract 与 rate sweep 报告，包含 SLO goodput 而非单个 tok/s；
4. 一个事故推演：queue saturation、KV preemption、NCCL hang、chat template 错误各怎样用不同证据定位。

能产出这四件东西，说明你不再只是“会启动 vLLM”，而是能解释、测量并维护一套推理服务。
