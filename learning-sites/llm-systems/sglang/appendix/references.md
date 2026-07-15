---
title: SGLang 源码、论文与术语
description: 固定提交源码入口、官方文档、核心论文、Ray 资料、术语对照与版本核验方法
lesson:
  track: SGLang
  stage: 04 · 索引
  time: 按需查阅
  level: 全阶段
  question: 文档、博客和实际行为冲突时，怎样建立可复现的证据链？
---

# SGLang 源码、论文与术语

本课程解释对象固定为 SGLang 提交 [`c879f3d`](https://github.com/sgl-project/sglang/tree/c879f3da5ceaaef3cb197c4e59ce683d420ce96c)。在线文档、nightly image 和 PyPI wheel 会继续变化；类名、参数、默认 backend 与支持矩阵都可能不同。先确认版本，才讨论行为。

## 证据优先级

1. **正在运行的系统**：实际 commit/wheel、最终参数、启动日志、metrics 与最小复现；
2. **同提交源码和测试**：回答所有权、分支条件和数据流；
3. **同提交官方文档**：理解设计意图与推荐配置；
4. **当前官方文档/release notes**：了解新版本用法，不反推旧版本；
5. **论文**：理解算法假设，不推断当前 CLI 默认值；
6. **博客/视频**：建立直觉后回源核验。

## 固定源码主线

| 主题 | 源码入口 |
| --- | --- |
| HTTP server / routes | [`srt/entrypoints/http_server.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/entrypoints/http_server.py) |
| Engine / process launch | [`engine.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/entrypoints/engine.py#L183) |
| TokenizerManager | [`tokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/tokenizer_manager.py#L262) |
| Scheduler request receiver / rank broadcast | [`request_receiver.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/scheduler_components/request_receiver.py#L73) |
| Scheduler / event loop | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/scheduler.py#L301)、[`run_event_loop`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/scheduler.py#L1457) |
| Request / ScheduleBatch | [`Req`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/schedule_batch.py#L677)、[`ScheduleBatch`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/schedule_batch.py#L1760) |
| Radix tree cache | [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/radix_cache.py#L280) |
| Request/KV pools | [`memory_pool.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/memory_pool.py#L238) |
| Extend/decode allocation | [`allocation.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/allocation.py#L303) |
| TP worker | [`tp_worker.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/tp_worker.py#L494) |
| ModelRunner | [`model_runner.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/model_executor/model_runner.py#L228) |
| Attention layer | [`radix_attention.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/layers/radix_attention.py#L73) |
| Data parallel route | [`data_parallel_controller.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/data_parallel_controller.py#L129) |
| Ray actor wrapper | [`ray/scheduler_actor.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/scheduler_actor.py#L31) |
| Ray engine / placement groups | [`ray/engine.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/engine.py#L229) |
| Ray DP controller / ZMQ route | [`ray/data_parallel_controller.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/data_parallel_controller.py#L39) |
| PD prefill / decode queues | [`prefill.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/disaggregation/prefill.py#L106)、[`decode.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/disaggregation/decode.py#L284) |
| HiCache factory / trees | [`registry.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/registry.py#L80)、[`hiradix_cache.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/hiradix_cache.py#L75)、[`unified_radix_cache.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/unified_radix_cache.py#L306) |
| Grammar | [`srt/constrained/`](https://github.com/sgl-project/sglang/tree/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/constrained) |
| Grammar mask before sampling | [`ModelRunner._preprocess_logits`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/model_executor/model_runner.py#L1384)、[`update_regex_vocab_mask`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/sampling/sampling_batch_info.py#L236) |
| RL weight update | [`scheduler_components/weight_updater.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/scheduler_components/weight_updater.py#L74) |
| Metrics | [`metrics_collector.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/observability/metrics_collector.py) |
| Server arguments | [`server_args.py`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/server_args.py) |

## 官方文档路线

课程提交中的文档可与源码同步阅读：

- [Install](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/get-started/install.mdx)
- [Quick Start](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/get-started/quickstart.mdx)
- [Server Arguments](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/server_arguments.mdx)
- [Benchmark and Profiling](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/developer_guide/benchmark_and_profiling.mdx)
- [Structured Outputs](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/structured_outputs.mdx)
- [Speculative Decoding](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/speculative_decoding.mdx)
- [PD Disaggregation](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/pd_disaggregation.mdx)
- [HiCache Design](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/hicache_design.mdx)
- [SGLang for RL](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/sglang_for_rl.mdx)
- [Post-training / veRL integration](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/references/post_training_integration.mdx)
- [Observability](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/advanced_features/observability.mdx)
- [Production Metrics](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/references/production_metrics.mdx)
- [Multi-node Deployment](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/references/multi_node_deployment/multi_node.mdx)

若锁定链接返回移动/重命名，直接在该 commit 的 `docs_new/docs/` 中搜索文件；不要把主分支的新参数硬套到固定课程版本。

## 论文主线

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — Runtime、RadixAttention、语言前端与系统优化的原始主线。
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling/continuous batching 背景。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — paged KV memory management 的对照系统。
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) — PD 解耦与 goodput 背景。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding 的目标分布与接受机制。

读论文只追问：假设的 workload 是什么；减少了计算、内存、通信还是排队；实验口径是什么；这些假设是否仍适用于当前源码和硬件。

## Ray 最小资料集

- [Ray Core Key Concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)
- [Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Logical Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Ray State CLI/API](https://docs.ray.io/en/latest/ray-observability/reference/cli.html)

理解 SGLang 当前 Ray 路径只需掌握 actor 生命周期、GPU logical resource 与状态观测。Ray actor 承载 Scheduler 进程，但请求数据面仍是 ZMQ，tensor collective 仍属于 PyTorch distributed/NCCL。

## 术语对照

| 术语 | 本课程中的精确定义 |
| --- | --- |
| Runtime | SGLang 的 serving 执行系统；不是单指 HTTP server |
| TokenizerManager | 接收 tokenized/HTTP 侧请求、维护 rid 状态并与 Scheduler/Detokenizer 通信的前端管理器 |
| Scheduler | 拥有请求调度、batch、KV metadata，并驱动本 rank ModelWorker 的 GPU 进程 |
| Req | Scheduler 内一个生成请求的可变状态，不等于 HTTP request object |
| ScheduleBatch | 一次调度/执行使用的请求集合与 pool/cache metadata |
| RadixCache | 用 radix tree 将 token prefix 映射到可复用 KV indices 的元数据结构 |
| RadixAttention | 系统/算法总称；源码中同名 layer 是 attention backend 的统一入口，不是 radix tree 本体 |
| ReqToTokenPool | request row 与 token positions 到 KV indices 的逻辑映射 |
| TokenToKVPool | 实际 K/V tensor slot 的分配与存储层 |
| chunked prefill | 把长 prompt 的未计算 token 分多个 Scheduler steps 执行 |
| overlap schedule | CPU 准备下一 batch 与 GPU/结果处理重叠的调度路径 |
| TP / PP / DP | 分别切层内 tensor、切 layers/stages、复制模型副本并路由请求 |
| DP Attention | MoE 场景中重组 attention data parallel 与 expert parallel 的策略，不等于普通 DP 的别名 |
| Ray SchedulerActor | 由 Ray 管生命周期、包含 Scheduler+TpModelWorker 的单 GPU actor；不替代 ZMQ/NCCL |
| PD disaggregation | 将 prefill 与 decode 放到独立实例，并转移 KV 与请求控制权 |
| HiCache | 将 radix-prefix KV 扩展到 GPU、host、远端存储的层次缓存机制 |
| grammar | 根据 schema/regex/EBNF 状态在采样前约束合法 token 的机制 |
| TTFT | 请求到首个输出 token/chunk 的时间，含 queue、tokenize、prefill 与前后端开销 |
| ITL/TPOT | 相邻输出 token 间隔/首 token 后平均每输出 token 时间；相关但统计口径可能不同 |
| goodput | 在明确延迟、成功率和质量 SLO 内完成的有效工作速率 |

## 版本核验

```bash
python - <<'PY'
import inspect
import sglang
from sglang.srt.managers.scheduler import Scheduler

print(getattr(sglang, "__version__", "unknown"))
print(inspect.getsourcefile(Scheduler))
PY

python -m sglang.launch_server --help > sglang-help.txt
```

同时保存容器 digest、git commit/dirty status、PyTorch/CUDA/driver/NCCL、model revision、tokenizer/chat template、完整最终参数和启动日志。若后台自动解析或覆盖参数，也要保存解析后的结果。

## 从课程毕业的五个作品

1. 一张自己画的 HTTP → TokenizerManager → Scheduler → ModelRunner → Detokenizer 所有权图；
2. 一个三 token rid trace，能指出 Req、ScheduleBatch、pool indices 和 streamed result；
3. 一份统一服务 rate sweep，报告 TTFT/ITL/success/goodput，而非单个 tok/s；
4. 一份 TP/PP/DP/Ray/PD/HiCache 决策记录，说明每层由什么证据触发；
5. 一次事故演练：queue saturation、KV 不足、collective hang、PD handoff 失败各怎样被不同证据识别。

能独立完成这五项，说明你不只是“会启动 SGLang”，而是能解释、测量、扩展并维护它。
