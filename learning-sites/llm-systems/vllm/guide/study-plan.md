---
title: vLLM 六阶段源码学习计划
description: 用 30 个学习日完成环境复现、原理推导、V1 调用链、GPU 执行、分布式与生产验收
lesson:
  track: vLLM
  stage: 00 · 计划
  time: 30 日，每日 60–120 分钟
  level: 从会用 Python 到能定位源码
  question: 每天具体读什么、做什么、留下什么证据，才算真正学会 vLLM？
---

# vLLM 六阶段源码学习计划

这不是“把文档从头读到尾”的目录。每一天都必须留下一个可检查产物：计算表、源码卡、命令输出、时序图、实验记录或故障复盘。只读不写、只启动不测、只看类名不追数据，都不算完成。

课程实现固定在：

```text
repository: https://github.com/vllm-project/vllm
commit:     61141ed265bfef41a0ca19e992567ea980919b96
commit date: 2026-07-15
main path:  vllm serve → V1 online serving
```

当前 stable 文档用于了解推荐用法；固定提交源码用于判断这一版**实际怎么执行**。两者冲突时先记录差异，不把新文档的行为倒灌到旧提交。

## 开始前：把证据环境固定下来

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm.git
cd vllm
git fetch origin 61141ed265bfef41a0ca19e992567ea980919b96
git switch --detach 61141ed265bfef41a0ca19e992567ea980919b96
git rev-parse HEAD
```

预期最后一行完整等于固定 SHA。若不是，就不要引用课程行号。

再建立一份学习记录，不需要新工具：

```text
notes/vllm/
├── 00-environment.md
├── 01-memory.xlsx-or-md
├── 02-request-trace.md
├── 03-scheduler-kv.md
├── 04-forward-sampling.md
├── 05-distributed.md
└── 06-production.md
```

每张“源码卡”统一回答五个问题：

```text
symbol / fixed-commit link:
call condition:
input → output:
state mutation:
how to verify:
```

## 三种学习条件

| 条件 | 能完成什么 | 暂时不能声称什么 |
| --- | --- | --- |
| 无 GPU | 原理、源码、CPU 单测、配置与日志分析 | 不能声称真实吞吐或显存结果 |
| 单 GPU | 完整 API、Scheduler/KV、forward、benchmark | 不能验证 TP/PP/NCCL 与多节点 Ray |
| 多 GPU/多节点 | TP/PP/DP/Ray、collective 与故障实验 | 仍需真实业务负载才可给生产容量结论 |

没有 GPU 时不要伪造数字。把命令、预期字段和无法执行的原因留在记录里，等有设备再补实测。

## 阶段 1：推理地基（Day 1–5）

### Day 1：画出生成依赖链

- 阅读[Prefill、Decode 与自回归循环](../fundamentals/inference-loop)。
- 手画 `prompt → first logits → token 1 → token 2`，区分“token 已采样”和“该 token 的 KV 已写入”。
- 从 [`LlamaForCausalLM.forward()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/model_executor/models/llama.py#L516-L533)确认 forward 返回 hidden states，logits 由独立的 `compute_logits()` 生成。

产物：一张含输入、输出与数据依赖箭头的时间轴。

验收：能解释为什么生成 100 个 token 至少有 100 次串行采样依赖，而 prompt 内的多个位置可以一次批量计算。

### Day 2：算 KV，而不是背“很占显存”

- 阅读[KV Cache 与 PagedAttention](../fundamentals/kv-cache)。
- 任选一个真实模型，从 `config.json` 取 layers、KV heads、head dim、dtype。
- 计算每 token、8K context、32 并发的 KV bytes；再改成 GQA/MQA 比较。

产物：公式、所有参数来源与单位换算。

验收：结果能反向代入；把 dtype 从 BF16 改 FP8 后，能说明哪些项变、哪些不变。

### Day 3：从论文到当前实现

- 读 vLLM/PagedAttention [论文摘要、§3–§4 与实验](https://arxiv.org/abs/2309.06180)。
- 再读固定源码 [`KVCacheManager`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/kv_cache_manager.py#L114-L193) 与 [`BlockPool`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/block_pool.py#L143-L196)。
- 写清论文的 block table 思想与当前多 KV group、prefix hash、connector 分支的关系。

产物：两栏“论文稳定思想 / 固定提交新增复杂度”。

验收：不会把历史 `paged_attention.md` 的某个 kernel 当成所有当前 backend 的唯一实现。

### Day 4：建立性能坐标系

- 阅读[延迟、吞吐与 continuous batching](../fundamentals/performance)。
- 为短问答、长文档 QA、长文本生成分别预测 TTFT、ITL、E2E 与 KV 压力。
- 写出一个 SLO goodput 定义，不能只写 tok/s。

产物：三类 workload 的指标假设表。

验收：能说明“吞吐更高但 p99 更差”为什么不矛盾。

### Day 5：纸上调度

- 阅读 [`Scheduler.schedule()` 的统一 token 差值模型](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/sched/scheduler.py#L417-L510)。
- 给三个请求设定 prompt、output、computed 与 token budget，手算一轮计划。
- 分别加入 chunked prefill、prefix hit 和 speculative tokens。

产物：四张调度表。

验收：每个 scheduled token 都能对应到 `num_tokens_with_spec - num_computed_tokens` 或明确的上限裁剪。

## 阶段 2：可复现实验（Day 6–10）

### Day 6：环境与首次启动

- 完成[第一台服务](../practice/first-server)。
- 固定 GPU、driver、CUDA、PyTorch、vLLM、模型 revision 和所有非默认参数。
- 保存冷启动日志，标记权重加载、内存 profile、compile、graph capture 与 ready。

验收：`/v1/models`、确定性非流式、SSE 流式和 `/metrics` 四项都通过。

### Day 7：受控失败

- 请求超长输入、`max_tokens=1`、错误模型名和客户端中断。
- 分别记录 HTTP 结果、finish reason、服务端日志、running/KV 是否回落。

验收：能证明断连触发的 abort 最终释放请求，而不是只看到客户端停止读取。

### Day 8：开放到达与饱和吞吐

- 完成[基准测试](../practice/benchmark)的 `request-rate=inf` 与有限 rate sweep。
- 固定输入/输出 token 长度并至少重复三次。

验收：找到 SLO 开始失守的 rate，而不是只记录最右端吞吐。

### Day 9：prefix cache 对照

- 准备一个稳定长前缀和 20 个不同问题。
- 跑冷请求、热前缀、前缀开头插入随机值三组。
- 同时记录 cached tokens、TTFT、E2E，排除 compile warmup。

验收：命中证据来自 token/metric，不来自“第二次看起来快”。

### Day 10：一次只改一个旋钮

- 扫 `max-num-batched-tokens` 或 `max-num-seqs`，不要同时改 TP、cache 与 dtype。
- 记录收益指标、受损指标和回退条件。

产物：完整实验契约与结果表。

验收：别人能用你的命令、模型 revision 与数据集 hash 重跑。

## 阶段 3：HTTP 到 EngineCore（Day 11–15）

### Day 11：启动所有权

- 读 [`ServeSubcommand.cmd()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/cli/serve.py#L44-L148)。
- 标出 single API、multi API、headless、DP LB 四条启动分支。
- 读 [`build_async_engine_client_from_engine_args()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/openai/api_server.py#L147-L193)，确认配置与 `AsyncLLM` 的创建/销毁边界。

验收：能从 CLI 参数预测会启动几个 API/Core/Worker 进程。

### Day 12：渲染与参数转换

- 读 chat route 与 [`_create_chat_completion()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/entrypoints/openai/chat_completion/serving.py#L249-L321)。
- 记录 `messages → rendered input/token ids → SamplingParams`。
- 刻意用错误 chat template，证明问题发生在 Scheduler 之前。

### Day 13：跨进程请求契约

- 读 [`AsyncLLM.add_request()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/async_llm.py#L280-L415)。
- 解释为何先在 `OutputProcessor` 注册 collector，再向 Core 发送请求。
- 区分 HTTP request、engine input、`EngineCoreRequest` 和 Core `Request`。

### Day 14：Core busy loop

- 读 [`EngineCoreProc.run_busy_loop()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/core.py#L1326-L1384)。
- 跟进 ADD 分支到 `Scheduler.add_request()`。
- 给同一 request id 标注 API process、Core process 两边各自拥有的状态。

### Day 15：完整前半链验收

- 完成[HTTP 到采样的固定源码链](../internals/full-code-path)前半部分。
- 不看课程，使用 `rg` 从 route 自己走到 `EngineCore.step()`。

验收：每跳都能写出调用条件、输入、输出、状态变化和进程边界。

## 阶段 4：Scheduler、KV 与 GPU（Day 16–21）

### Day 16：running loop

- 逐行读 `Scheduler.schedule()` 的 running 分支。
- 把所有限制 `num_new_tokens` 的条件列成优先级表。

### Day 17：waiting admission

- 跟 `get_computed_blocks() → allocate_slots() → RUNNING`。
- 用 block size=4 做一次 hit/miss/不足返回 `None` 的手算。

### Day 18：block 生命周期

- 读 `BlockPool` 的 touch、allocate、free、eviction。
- 解释 `ref_cnt=0` 的 cached block 为何既在 free queue 又保留可复用内容。

### Day 19：Worker 与 Runner

- 阅读[Worker、ModelRunner 与模型执行](../internals/model-execution)。
- 用 [`VllmConfig.use_v2_model_runner`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/config/vllm.py#L530-L572)判断你的模型走 V1 还是 V2 Runner，不凭文件名猜。

### Day 20：forward 与采样

- 完成[模型 forward 与采样源码解读](../internals/model-forward-sampling)。
- 跟 `SchedulerOutput → InputBatch → block table/slot mapping → model → logits → sampler`。

### Day 21：源码实验

- 完成[源码跟踪实验](../practice/source-lab)。
- 跑最小 Scheduler/KV/Sampler 单测；GPU 可用时再做真实 trace。

验收：同一条结论同时有代码锚点和运行证据。

## 阶段 5：并行与 Ray（Day 22–26）

### Day 22：先分清 TP、PP、DP

- 阅读[TP、PP、DP 与 Ray](../internals/distributed)。
- 为 `TP=4, PP=2, DP=3` 画 engine core、worker 和模型副本数量。
- 写明 TP/PP 是单 replica 内切分，DP 是多个 Core/KV 副本。

### Day 23：Executor 后端选择

- 读 [`Executor.get_class()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/executor/abstract.py#L47-L92)。
- 对 `uni/mp/ray/external_launcher` 分别记录启动条件与 worker 调用方式。

### Day 24：Ray 控制面

- 读 [`RayDistributedExecutor._init_workers_ray()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/executor/ray_executor.py#L143-L260)。
- 画 placement group bundle → actor → adjusted rank → PP×TP 网格。
- 明确 Ray 安排 actor，NCCL/设备 communicator 搬模型 tensor。

### Day 25：单节点并行对照

- 完成[分布式与 Ray 实验](../practice/distributed-lab)的 single、TP、PP/TP、DP、Ray 对照。
- 记录每卡权重、KV capacity、ready time、TTFT、ITL、collective 时间。

### Day 26：多节点或故障桌面演练

- 有两节点：验证 Ray resource、placement group、rank/IP 和 NCCL。
- 无两节点：用真实日志样例完成“actor pending / distributed init hang / first forward hang”的分层判断。

验收：不再用“Ray 挂了”同时指代资源不可调度、端口不可达和 NCCL collective 卡住。

## 阶段 6：特性、生产与毕业项目（Day 27–30）

### Day 27：三类优化各省什么

- 阅读[prefix、chunked prefill 与 speculative decoding](../advanced/features)。
- 为三种 workload 选择一个优先实验，并写无效条件。

### Day 28：容量与诊断

- 阅读[生产诊断与容量规划](../advanced/production)。
- 用权重账、KV 账与 rate sweep 给出容量区间，而不是单个理论并发。

### Day 29：故障演练

从下面抽两个：

- waiting 单调增长；
- preemption 暴涨；
- TTFT 正常但 ITL 变坏；
- Ray actor pending；
- NCCL first-forward hang；
- 客户端断开后 running 不下降。

每个必须写：最强假设、反证指标、最小缓解、长期修复、回归门禁。

### Day 30：毕业项目

交付一份“模型上线证据包”：

```text
1. 固定环境与启动命令
2. 从 HTTP 到 sampler 的源码时序图
3. KV bytes/token 与容量账
4. baseline + rate sweep + 单变量优化
5. TP/PP/DP/Ray 选择及反例
6. SLO、dashboard、告警与回滚条件
7. 一个故障的证据化复盘
```

## 最终评分表

| 能力 | 不通过 | 通过 |
| --- | --- | --- |
| 原理 | 会说“分页更快” | 能从碎片、block table、共享与容量推导收益边界 |
| 源码 | 罗列类名 | 能沿真实调用链说明条件、数据契约和状态变化 |
| 实践 | 服务返回 200 | 有确定性、流式、失败、取消、指标和负载验收 |
| 性能 | 只报平均 tok/s | 有长度分布、到达过程、p99、goodput 与稳定区 |
| 分布式 | “Ray 负责分布式” | 能分开算法、actor 编排、rank、NCCL 与故障层 |
| 生产 | 凭经验调参数 | 每个动作都有指标假设、反证与回退条件 |

完成本计划后，进入[V1 源码地图](./source-map)，先用一页确认全局，再沿[完整固定源码链](../internals/full-code-path)逐函数复核。
