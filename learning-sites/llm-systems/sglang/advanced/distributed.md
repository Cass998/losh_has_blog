---
title: SGLang 并行、PD 解耦、HiCache 与 Ray
description: 区分 TP、PP、DP、DP Attention、Ray 进程编排、Prefill/Decode 解耦和分层 KV 缓存
lesson:
  track: SGLang
  stage: 04 · 扩展
  time: 110–150 分钟
  level: 了解集合通信与 KV Cache
  question: 扩展 SGLang 时，究竟是在切模型、复制服务、拆分阶段，还是改变 KV 的存放层级？
---

# SGLang 并行、PD 解耦、HiCache 与 Ray

先把六个经常混用的概念拆开：**TP/PP 切一个模型副本，DP 复制副本，DP Attention 改变 MoE 中 attention 与 expert 的并行关系，Ray 管 Scheduler 进程生命周期，PD 把 prefill 与 decode 拆成不同实例，HiCache 把 KV 扩展到更慢、更大的存储层。**它们可以组合，但解决的不是同一个问题。

## 一张选择地图

```mermaid
flowchart TD
    Q{当前瓶颈是什么?}
    Q -->|单卡放不下模型| M[切模型]
    M --> TP[TP: 层内切权重/张量]
    M --> PP[PP: 按层切 stages]
    Q -->|请求吞吐不足| D[DP: 复制模型副本]
    Q -->|MoE 通信或负载问题| E[EP + DP Attention]
    Q -->|prefill 干扰 decode| PD[PD disaggregation]
    Q -->|重复前缀大于 GPU KV 容量| H[HiCache: L1/L2/L3]
    Q -->|需要统一创建/放置进程| R[Ray actors]
```

先用单实例、统一调度、GPU KV 做基线。只有指标证明对应瓶颈存在，才引入下一层；否则增加的是故障面，而不一定是吞吐。

## TP、PP、DP 与 EP 各切什么

| 维度 | 被切分或复制的对象 | 常见通信 | 适合解决 | 主要代价 |
| --- | --- | --- | --- | --- |
| TP | 每层权重、attention heads、中间张量 | 层内 all-reduce/all-gather 等 | 单卡放不下、利用高速互联 | 高频通信，受拓扑影响大 |
| PP | Transformer layers/stages | stage 间 activation | 模型跨节点、减少跨节点层内 collective | bubble、stage 不均衡、控制更复杂 |
| DP | 完整模型或 TP×PP 副本 | 请求路由；MoE 场景另有 collective | 提高请求吞吐、故障隔离 | 权重与 KV 复制、负载不均 |
| EP | MoE experts | token dispatch/combine、all-to-all | 利用稀疏专家结构 | 热专家、路由不均与网络成本 |

普通拓扑的 GPU 数可先用下式核对：

$$
N_{GPU}=DP\times TP\times PP
$$

EP、attention TP/DP、context parallel 可能复用或重组既有 rank group，不能机械地再乘一个维度。最终应以启动日志中的 rank 映射为准。

### 多节点的可核对起点

固定提交的[官方多节点文档](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/docs_new/docs/references/multi_node_deployment/multi_node.mdx)以两节点、每节点八卡的**全局 TP=16**为例；两侧参数必须一致，只有 `node-rank` 不同：

```bash
# node 0
python -m sglang.launch_server \
  --model-path /models/model \
  --tp 16 --nnodes 2 --node-rank 0 \
  --dist-init-addr 10.0.0.10:5000

# node 1：其余参数保持一致
python -m sglang.launch_server \
  --model-path /models/model \
  --tp 16 --nnodes 2 --node-rank 1 \
  --dist-init-addr 10.0.0.10:5000
```

这里的 `--tp 16` 是跨两节点的全局 TP world size，不是“每节点再各起 16 卡”。若要改成节点间 PP，先从实际 `tp_size × pp_size × dp_size` 与进程 rank 映射验证所需 GPU，不能把上面的 TP 示例直接再乘 PP。参数名会随版本演进，运行课程外版本时先看 `python -m sglang.launch_server --help`。启动前独立验证模型路径、CUDA/PyTorch/NCCL、节点互通与 GPU collective；HTTP 健康不等于跨节点 collective 健康。

## DP Controller：把请求交给哪个副本

当 `dp_size > 1` 时，[`DataParallelController`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/data_parallel_controller.py#L129) 创建多个 Scheduler groups，并通过 ZMQ 将 tokenized request 分发给某个 DP rank。

可选路由策略包括 round robin、累计请求量、累计 token 量，以及 PD 场景的 bootstrap room。它说明两个事实：

1. DP 的控制对象是**请求**，不是一个 forward 内的 tensor shard；
2. request count 相同不代表负载相同，长 prompt、长输出、prefix hit 与 KV occupancy 都会改变成本。

```mermaid
flowchart LR
    T[TokenizerManager] --> C[DataParallelController]
    C -->|request A| D0[DP rank 0<br/>Scheduler + TP group]
    C -->|request B| D1[DP rank 1<br/>Scheduler + TP group]
    C -->|request C| D2[DP rank 2<br/>Scheduler + TP group]
    D0 --> O[Detokenizer]
    D1 --> O
    D2 --> O
```

做容量诊断时必须按 DP rank 看 queue、running requests、token usage 和 cache hit。总平均正常而某个 rank 堵塞，优先修路由，不要先扩总 GPU。

## DP Attention：为什么 MoE 需要另一种组合

MoE 模型的 attention 是 dense 计算，experts 是稀疏计算。若二者沿完全相同的 TP group 切分，attention 可能产生不必要的同步，而 experts 又需要较大的 EP group。DP Attention 的思路是让 attention 部分在若干 ranks 上采用数据并行式的独立输入，同时让 MoE experts 使用更适合的 expert parallel 分布。

它不是“打开后所有模型都更快”的通用开关。需要核对：

- 模型是否为受支持的 MoE 架构；
- attention TP、MoE DP/EP rank 怎样映射；
- token dispatch 是否负载均衡；
- 节点间 all-to-all 是否成为新瓶颈；
- 相同 workload 下 TTFT、ITL、吞吐和输出正确性是否同时通过。

## Ray 在 SGLang 中到底做什么

固定提交中的 [`SchedulerActor`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/scheduler_actor.py#L31) 给出了最精确的边界：**每个 actor 管一张 GPU，并运行 Scheduler + `TpModelWorker`；Ray 用于进程生命周期，正常 request/response 仍由 ZMQ 传输。**

```mermaid
flowchart TD
    L[launch_server / Engine] --> R[Ray control plane]
    R --> A0[SchedulerActor<br/>GPU 0]
    R --> A1[SchedulerActor<br/>GPU 1]
    T[TokenizerManager] -->|ZMQ tokenized request| A0
    T -->|ZMQ tokenized request| A1
    A0 -->|ZMQ token ids / metadata| D[DetokenizerManager]
    A1 -->|ZMQ token ids / metadata| D
    D -->|ZMQ decoded result| T
    A0 <-->|distributed collectives| A1
```

上图是启用 tokenizer 的标准文本路径：请求由 TokenizerManager 发给 Scheduler，输出先经 DetokenizerManager 增量解码，再回到 TokenizerManager/HTTP 流。`--skip-tokenizer-init` 等特殊模式会改变输出载荷，但不能据此把标准结果链画成 Scheduler 与 TokenizerManager 双向直连。

因此要分三层排障：

| 层 | 负责什么 | 典型证据 |
| --- | --- | --- |
| Ray | actor 创建、GPU resource assignment、存活与异常传播 | actor state、assigned GPU、Ray logs |
| SGLang + ZMQ | TokenizerManager、Scheduler、Detokenizer 的请求与结果流 | rid trace、socket/port、各进程日志 |
| PyTorch distributed/NCCL | TP/PP/EP rank 间 tensor 通信 | rank mapping、collective timeout、NCCL logs/profile |

`--use-ray` 不会自动选择 TP/PP/DP，也不会让坏的 NCCL 网络变快。它适合你已经有 Ray 集群资源管理需求、需要跨节点统一 actor 生命周期或希望由 Ray 分配 GPU 的场景；单机基线通常先保留原生进程路径。

### `--use-ray` 从入口替换了哪一段

固定提交的 [`launch_server.run_server()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/launch_server.py#L15) 只有在 `ServerArgs.use_ray=True` 时进入 Ray HTTP launcher；没安装额外依赖会提示安装 `sglang[ray]`。这个分支替换的是 **Scheduler 进程的创建与回收**：

```text
普通路径: Engine → multiprocessing.Process → Scheduler
Ray 路径: RayEngine → placement group → SchedulerActor → Scheduler
共同后半段: Scheduler → TpModelWorker → ModelRunner → NCCL/attention backend
共同请求面: TokenizerManager/DP Controller → ZMQ → Scheduler
标准结果面: Scheduler → ZMQ → DetokenizerManager → ZMQ → TokenizerManager
```

[`RayEngine._launch_scheduler_processes()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/engine.py#L249) 的启动事务是：

1. 取得调用者传入/current placement group；没有时按节点数自动建 group；
2. 计算所需 Scheduler actors；普通路径为 `DP×TP×PP`，DP Attention 为 `TP×PP`，见 [`_compute_world_size()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/engine.py#L105)；
3. 为每个 rank 调 [`_create_scheduler_actor()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/engine.py#L178)，声明 `num_gpus=1` 并把 actor 固定到 placement bundle；
4. `ray.get(actor.get_info())` 等所有 actor 完成模型加载、KV pool 与 NCCL 初始化；任一 actor 失败就 kill 已创建 actors；
5. 才异步调用每个 actor 的 `run_event_loop()`，把 init info 交回 TokenizerManager/HTTP warmup。

因此 HTTP 进程出现不代表 actor 初始化成功；actor `ALIVE` 也不代表 NCCL 和真实生成已经通过。启动验收要覆盖上述五步。

### Placement group 是资源约束，不是 rank group

Ray placement group 预留 CPU/GPU 资源并约束 actor 放置；TP/PP/DP group 仍由 SGLang/PyTorch distributed 建立。固定源码有两种模式：

| 模式 | bundle 形态 | 关键条件 |
| --- | --- | --- |
| 自动 PG | 每节点一个 bundle，包含该节点所需 GPU；单节点 `STRICT_PACK`，多节点 `SPREAD` | 总 GPU 能被 `nnodes` 整除；rank-0 Scheduler 所在节点必须与 Engine 共置 |
| 自定义 PG | 每个 GPU 一个 bundle | [`_validate_custom_placement_group()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/engine.py#L151) 要求每 bundle 不超过 1 GPU、GPU bundles 不少于 world size |

自定义 PG 可用 `SGLANG_RAY_BUNDLE_INDICES` 指定 rank 使用哪些 bundle；源码检查数量、重复和越界。这个变量改变**放置映射**，不会改变 `tp_rank/pp_rank/dp_rank` 数量。自动 PG 的 [`_find_engine_bundle()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/engine.py#L47) 若找不到 Engine 所在节点会直接失败，因为 rank 0 与 Engine 间的本地控制/数据地址有共置假设。

### DP 开启时谁还在主进程

`dp_size>1` 时不是“再创建一个 Ray Router actor”。[`RayDataParallelController`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/ray/data_parallel_controller.py#L39) 在 Engine 进程内复用普通 DP controller 的 event loop，以 daemon thread 运行；它为每个 DP rank 建 ZMQ PUSH socket，再让 Ray 创建各 TP/PP actors。actor 初始化完成后，controller 才开始按请求路由。

这给出清楚的故障边界：

- actor 一直 `PENDING`：先查 Ray resources/placement bundles；
- actor `DEAD` 且模型未 ready：查 actor stderr、模型加载、GPU 与 NCCL init；
- actors 全 `ALIVE` 但只有一个 DP rank 堵塞：查 DP controller 的 token/cache-aware 路由与 ZMQ；
- 所有 ranks 同时卡在 forward：查 PyTorch distributed/NCCL/kernel，不要用 `ray status` 代替 collective trace。

### 单机 Ray 最小实验

固定版本的 [`python/pyproject.toml`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/pyproject.toml#L122) 为 Ray extra 声明 `ray[default]>=2.55.1`；先安装这份匹配依赖，再建立单机 Ray runtime：

```bash
python3 -m pip install -e 'python[ray]'
ray start --head --num-gpus=2
ray status

RAY_ADDRESS=auto python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 2 \
  --use-ray \
  --host 127.0.0.1 --port 30000 \
  --enable-metrics 2>&1 | tee sglang-ray.log
```

若使用 wheel，就按该 wheel 官方 extra 安装，不要执行源码目录的 editable 命令。测试结束用 `ray stop`；不要在共享集群上停止别人的 head。

另开终端验证：

```bash
ray status
ray list actors --detail | rg 'sglang_scheduler|State|GPU'
ray list placement-groups --detail

curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Reply exactly ray-ready","sampling_params":{"temperature":0,"max_new_tokens":8},"rid":"ray-smoke"}'
```

验收条件：actor 数等于 world size=2；名称中的 TP ranks 为 0/1；每个 actor 获得一张 logical GPU；placement group 已创建；真实生成完成；停止一个 actor 后请求明确失败且 Engine 日志记录 actor termination。再与同模型、同 TP 的非 Ray 基线比较启动时间、steady-state 延迟和故障日志，才能判断 Ray 带来的是所需编排能力还是额外控制面。

多节点时还要先让各节点加入同一 Ray 集群，并确认 Engine 节点拥有 placement bundle、模型/环境在所有 worker 可见。Ray 能把 actor 放到远端，不会替你同步本地模型目录、修复 NCCL interface 或开放 ZMQ/distributed 端口。

::: warning 集群安全
Ray 控制面、SGLang 内部 ZMQ 和 distributed init 地址都应放在可信私网。不要把 actor/control ports 直接暴露到公网；公网流量只应进入带 TLS、认证、限流和审计的 API 网关。
:::

## PD 解耦：为什么要拆 prefill 与 decode

Prefill 对长 prompt 做大量并行矩阵计算，偏 compute-intensive；decode 每步 query 很小却要反复读大段 KV，偏 memory-bandwidth-intensive。统一 batch 中插入大 prefill，可能拉长正在 decode 请求的 inter-token latency；同一组 GPU 也很难同时为两种形态调到最佳。

PD disaggregation 将它们放到不同实例：

```mermaid
sequenceDiagram
    participant R as Client / Router
    participant D as Decode Scheduler
    participant P as Prefill Scheduler
    participant K as KV transfer backend
    R->>D: request + bootstrap room / prefill address
    D->>D: create receiver + prefix match
    D->>D: preallocate target KV / metadata
    R->>P: same request + bootstrap room / decode target
    P->>D: bootstrap handshake / destination metadata
    P->>P: compute prompt KV
    P->>K: transfer missing KV + metadata
    K-->>D: transfer complete
    D->>D: commit metadata + build prebuilt batch
    D-->>R: continue decode and stream tokens
```

Router 可以并发发出 prefill/decode 两个 HTTP 请求；图中的先后表示**状态依赖**：decode 侧必须先建立 receiver 并预留目标空间，prefill 才能把 KV 写到正确位置；只有 transfer 完成并提交 metadata 后，decode 请求才进入可执行的 prebuilt batch。逐函数状态机见[PD 解耦与 HiCache](./pd-hicache)。

固定源码以 [`DisaggregationMode`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/disaggregation/utils.py#L68) 区分 prefill/decode；Scheduler 在 [`init_disaggregation()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/scheduler.py#L1103) 装配对应 mixin/连接。Prefill 主线见 [`SchedulerDisaggregationPrefillMixin`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/disaggregation/prefill.py#L422)，decode 主线见 [`SchedulerDisaggregationDecodeMixin`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/disaggregation/decode.py#L1985)。

SGLang 在该版本包含 Mooncake、NIXL 等 KV transfer 路径。选后端不是只看“支持”二字，还要测：

$$
T_{PD}=T_{queue,p}+T_{prefill}+T_{KV-transfer}+T_{queue,d}+T_{decode}
$$

若 `T_KV-transfer`、router handoff 或 decode 等待抵消了隔离收益，PD 反而更慢。它通常在 prompt 很长、prefill 对 ITL 干扰明显、两阶段需要不同 GPU 配比时才值得。

### PD 的容量不能只配一边

稳态必须同时满足：

$$
\lambda < \mu_{prefill},\qquad \lambda < \mu_{decode},\qquad bandwidth_{KV}>demand_{KV}
$$

prefill 富余但 decode 饱和，KV 会在中间堆积；decode 富余但 prefill 饱和，首 token 仍排队。监控必须同时覆盖两个 queue、handoff 失败/超时、KV 传输带宽和端到端 rid。

## HiCache：把 KV 从一层扩成三层

RadixCache 解决“哪些 prefix 相同”；HiCache 进一步解决“命中的 KV 放在哪里”。概念层级是：

| 层 | 典型位置 | 容量/速度 | 所有权 |
| --- | --- | --- | --- |
| L1 | GPU HBM | 最快、最小 | 单实例私有 |
| L2 | host memory | 更大、更慢 | 单实例私有 |
| L3 | distributed storage | 最大、延迟/带宽最不稳定 | 多实例共享 |

[`UnifiedRadixCache.init_hicache()`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/unified_radix_cache.py#L499) 将层次缓存接入 Radix 元数据；[`HostKVCache`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/pool_host/base.py#L81) 管 host pool；[`HiCacheController`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/managers/cache_controller.py#L202) 协调 transfer；[`HiCacheStorage`](https://github.com/sgl-project/sglang/blob/c879f3da5ceaaef3cb197c4e59ce683d420ce96c/python/sglang/srt/mem_cache/hicache_storage.py#L141) 定义更远端存储接口。

一次命中大致经历：本地 radix match → 查询/预取较慢层 → 等待、超时或 best-effort 决策 → 将 KV 恢复到可执行位置 → 本轮 forward → 按 write policy 回写。

### 策略改变的是延迟风险

- `best_effort`：来不及就重算，尾延迟较可控，但可能浪费算力；
- `wait_complete`：尽量等缓存，命中收益高，但慢存储会进入关键路径；
- timeout：在等待与重算之间设上界；
- write-through：更快进入远端层，但写带宽进入请求成本；
- write-back/selective：减少写放大，但一致性与淘汰更复杂。

多 rank 执行还需对命中长度达成一致；某 rank 不能读取完整 prefix 时，整个 group 只能采用共同可用的安全前缀。测 HiCache 时至少报告 L1/L2/L3 分层命中、load/store 带宽、restore latency、重算 token 与端到端 TTFT，不能只报“cache hit rate”。

## 组合时的启动顺序

1. 单 GPU 统一服务，确认语义与 workload；
2. TP/PP 解决模型放置，保存 rank 与网络基线；
3. DP 解决吞吐，并验证逐 rank 负载；
4. 有 MoE 证据再试 EP/DP Attention；
5. 有 prefill 干扰证据再做 PD；
6. 有大量可复用长前缀且 GPU cache 不够，再做 HiCache；
7. 只有需要统一进程资源编排时，再把 Scheduler 生命周期交给 Ray。

每一步只改变一个主变量，并保留前一步可回滚配置。

## 通关练习

给定：四节点、每节点八卡的 MoE 模型；模型单节点放不下；短 prompt chat 的 ITL 正常，但长文档流量进入后 ITL p99 翻十倍；prefix 跨租户不可共享。请提出两阶段实验，而不是一次打开所有开关。

一个合理答案是：先以节点内 TP、节点间 PP 放下模型并测统一调度；随后隔离长文档 workload，比较 chunked prefill 与 PD。因 prefix 不可跨租户共享，不能用全局 HiCache 命中率掩盖隔离要求；Ray 只有在集群进程编排需要时才进入实验。

PD bootstrap、目标端预分配、KV transfer 和 `prebuilt` 接管的逐函数解释见[PD 解耦与 HiCache](./pd-hicache)；再看这些系统能力怎样接入[结构化输出与 RL 工作流](./features)。
