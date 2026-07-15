---
title: 源码驱动实践：可下载脚本、两步门禁与失败实验
description: 用 CPU/Gloo、DTensor/DCP 和固定框架配方验证 rank、梯度、参数 shard、schedule、loss 与 checkpoint，而非只看进程退出码
lesson:
  track: 分布式训练
  stage: 04 · 实践
  time: 1–3 天
  level: 已完成源码主线
  question: 如何把源码结论变成能复现、能失败、能验收的训练实验？
---

<script setup>
import rankMeshAuditUrl from './code/rank_mesh_audit.py?url'
import ddpGlooEquivalenceUrl from './code/ddp_gloo_equivalence.py?url'
import dtensorDcpRoundtripUrl from './code/dtensor_dcp_roundtrip.py?url'
</script>

# 源码驱动实践：可下载脚本、两步门禁与失败实验

实践按风险递增：**纯 Python rank 表 → CPU/Gloo DDP → CPU DTensor/DCP → 两 GPU FSDP2 → TorchTitan → Megatron → DeepSpeed**。每层先通过 correctness gate，再谈吞吐。脚本退出 0 只是一项证据；还要保存 global batch、logical checksum、collective trace 与恢复后的下一步。

## 本站实际验证边界

| 项目 | 本站生成环境状态 | 可以声称什么 |
| --- | --- | --- |
| `rank_mesh_audit.py` | Python 3 stdlib，可直接执行 | 脚本语法与 lexicographic group 表逻辑可验证 |
| DDP/DCP 脚本 | 已做 Python 语法检查；当前主机没有安装 `torch` | 是可下载、针对固定 API 编写的实验；不能声称本机运行通过 |
| FSDP2/TorchTitan/Megatron/DeepSpeed GPU 配方 | 当前主机无 PyTorch/GPU/项目容器 | 只标“待在固定官方环境执行”，不伪造 loss、显存或性能 |

这种边界必须保留在实验报告中。CPU/Gloo 正确不等于 CUDA/NCCL overlap 正确，能静态 import 也不等于 8 GPU functional test 通过。

## 0. 建立实验目录与环境清单

下载固定源码见[42 天计划](../guide/source-study-plan)。另建输出目录，不污染 detached worktrees：

```bash
mkdir -p ~/distributed-labs/{manifests,logs,checkpoints,profiles}
```

每次 run 前保存：

```bash
{
  date -Is
  uname -a
  git rev-parse HEAD
  python --version
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("nccl", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
PY
  nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
  nvidia-smi topo -m
} | tee ~/distributed-labs/manifests/run-$(date +%Y%m%d-%H%M%S).txt
```

容器环境再保存 image digest、mounted source commit 与 lockfile hash；集群再保存 node list、NIC、launcher 参数和关键 env。不得把 token、密钥或内部 registry credential 写进公开日志。

## 1. Rank/mesh 审计：零依赖先手算

下载 <a v-bind="{ href: rankMeshAuditUrl, download: 'rank_mesh_audit.py' }"><code>rank_mesh_audit.py</code></a>。它按用户给定的 axis 顺序生成一个**教学用 lexicographic mesh**，用来和框架真实日志 diff；它不冒充 Megatron `RankGenerator` 或 TorchTitan `DeviceMesh` 的默认 order。

```bash
python3 rank_mesh_audit.py --world-size 16 dp=2 pp=2 cp=2 tp=2
```

每行包含 rank coordinate，以及固定其他 coordinates、只沿一个 axis 变化得到的 group members。验收：

1. 输出 16 个 ranks；
2. 每个 axis group size等于 degree；
3. 每个 rank恰好属于一个该 axis group；
4. group members在所有成员日志中相同；
5. 再与运行框架打印的真实 groups比较，差异必须能由 axis order解释。

### 故意失败

```bash
python3 rank_mesh_audit.py --world-size 15 dp=2 pp=2 cp=2 tp=2
```

预期立即报 degree product 16 与 world 15 不一致。这比启动 15 个进程后在 NCCL timeout 才发现更好。

## 2. DDP 两 rank 数值等价

下载 <a v-bind="{ href: ddpGlooEquivalenceUrl, download: 'ddp_gloo_equivalence.py' }"><code>ddp_gloo_equivalence.py</code></a>，在安装了 PyTorch distributed/Gloo 的环境运行：

```bash
torchrun --standalone --nproc-per-node=2 ddp_gloo_equivalence.py
```

脚本不是“hello world”。它固定四个 global samples：rank 0/1 各处理两个，DDP local mean gradient再跨 ranks average；另建单进程 reference，用完整四样本 global mean做一次 backward/SGD。它断言：

- DDP gradient 等于 reference global-batch gradient；
- 两 ranks 的 gradients 相同；
- step 后 logical parameters 等于 reference；
- 两 ranks parameters 相同；
- 两 local losses 的平均等于 global reference loss。

成功输出形如：

```text
PASS {'backend': 'gloo', 'world_size': 2, 'accumulate': False, ...}
```

不应把 checksum 数字写死进教程，因为 PyTorch 初始化/算子细节可能随版本改变；本次 run 应保存实际值与版本。

### 梯度累积与 `no_sync`

```bash
torchrun --standalone --nproc-per-node=2 ddp_gloo_equivalence.py --accumulate
```

每 rank 将 local batch拆成两个 microbatches，第一个 forward+backward放在 `ddp.no_sync()`，第二个同步。每个 microbatch loss除以 2，最终仍应等价于 local-batch mean，再由 DDP平均成 global mean。

对照[DDP 源码页](../internals/pytorch-ddp-runtime)：`no_sync` 必须包含 forward；只包 backward会让 `_post_forward` 已经准备同步状态。若修改脚本做失败实验，用 shell timeout保护：

```bash
timeout 30s torchrun --standalone --nproc-per-node=2 your_broken_variant.py
```

超时后保存每 rank最后一个 collective和 stack；不要只写“DDP 卡死”。

## 3. DDP bucket/unused parameter 观察

在两 GPU/NCCL 环境基于同一小模型做四个单变量 runs：

| Run | 改动 | 应观察什么 |
| --- | --- | --- |
| A | 默认 bucket | backward grad ready order、bucket count |
| B | `bucket_cap_mb` 调小 | 更多 bucket launch，payload总量数量级不变 |
| C | `gradient_as_bucket_view=True` | grad/bucket storage alias与峰值变化 |
| D | rank-dependent optional branch + 正确 `find_unused_parameters` | graph traversal、used-map与同步开销 |

使用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 和 PyTorch profiler，但只在有限 ranks开详细 trace。验收不是“B 比 A 快”；小模型 bucket过小可能更慢。应回答：第一 bucket何时 ready、通信是否与剩余 backward重叠、最终哪里 wait。

## 4. DTensor + DCP shard round-trip

下载 <a v-bind="{ href: dtensorDcpRoundtripUrl, download: 'dtensor_dcp_roundtrip.py' }"><code>dtensor_dcp_roundtrip.py</code></a>，用一个**不存在的新目录**：

```bash
torchrun --standalone --nproc-per-node=2 dtensor_dcp_roundtrip.py \
  --checkpoint /tmp/dcp-dtensor-lab-001
```

脚本：

1. 建 2-rank CPU `DeviceMesh`；
2. 将一个 logical tensor按 `Shard(0)` 分布；
3. `dcp.save()` 只写各 local shards及metadata；
4. 重新创建全零、同 layout 的目标 DTensor与 step tensor；
5. `dcp.load()` 写入预分配目标；
6. `full_tensor()` 后逐元素比较，并打印每 rank local shard shape/sum。

这直接验证 load 的目标-state语义：DCP 不会替你的训练程序猜 model/optimizer object；先重建目标 topology/对象，再 load values。

### 进一步做 reshard matrix

在有 4 ranks的环境保存 `(Shard(0), world=4)`，分别尝试项目支持范围内的目标：

| Save | Load | 要核对 |
| --- | --- | --- |
| world 4, shard 4 | world 4, shard 4 | exact local/global values |
| world 4, shard 4 | world 2, shard 2 | metadata planner是否重分片，logical value不变 |
| HSDP 2×2 | 1D FSDP 4 | replicate/shard placements转换 |
| model-only | full training resume | 应明确失败或初始化缺失 optimizer，不可称完整恢复 |

不要默认任何 PyTorch版本都支持所有 topology变换；每个格子记录 `supported/pass/fail/not tested`。

## 5. 两 GPU FSDP2 状态机实验

在与目标 PyTorch提交匹配的环境建 `Embedding + 2 Transformer-like blocks + head` 小模型，采用下面顺序：

```python
mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
for block in model.blocks:
    fully_shard(block, mesh=mesh, reshard_after_forward=True)
fully_shard(model, mesh=mesh, reshard_after_forward=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

必须记录每个 parameter：FQN、logical shape、local shape、placements、FSDP group。然后做：

| Run | 单一变量 | 预期源码路径 |
| --- | --- | --- |
| A | block-wise，non-root reshard | 每 block forward AG、post-forward reshard、backward再 AG、grad RS |
| B | 所有 groups不 reshard | backward AG减少，full params驻留增加 |
| C | root-only fully_shard | collective unit巨大，peak/full materialization增加 |
| D | forward中故意抛异常 | 下轮前调用 root `reset_iter_state()`，失败轮grad丢弃 |

用 profiler 事件 `FSDP::pre_forward/all_gather/post_backward_reduce/root_post_backward` 对照[FSDP2 源码](../internals/fsdp2-source)。若 profiler无某事件，先核对 mesh size=1/no-op、policy与当前版本，不要补画一条不存在的通信。

### 数值门禁

单 rank reference和两 rank FSDP2必须：

- 同 initial logical state（建议先建 seed checkpoint）；
- 同 global samples与global valid tokens；
- 同 precision/recompute/loss scaling；
- 比较 logical full parameter/update，不比较 rank-local shard；
- save→restart→load→再一步与 uninterrupted run在容差内。

## 6. TorchTitan：从 config rejection 到 two-step trace

使用固定仓库的官方 `llama3_debugmodel` 配置。先只解析/打印 config，再从 1 GPU开始；详细命令见[第一组实验](./first-runs)。本实验额外要求源码 trace：

```text
main/config.build
→ Trainer.init_distributed + every mesh coordinate
→ meta model
→ parallelize_llama transform order
→ to_empty/init_weights
→ optimizer built after parallelization
→ dataloader batch rank
→ checkpoint load
→ train_step global valid tokens
→ FSDP/TP/PP hooks
→ optimizer + scheduler
→ checkpoint save
```

四个 runs：single、FSDP2=2、TP=2、PP=2，一次只开一维。每个 run至少 2 个 updates，不把 compile/lazy-init首 step当稳态性能。

### 必须主动触发的配置失败

1. world product不匹配；
2. seq length不满足 TP/SP/CP divisor；
3. PP启用但 model spec无 `pipelining_fn`；
4. global batch不能整除 `local_batch×batch_degree`；
5. seed checkpoint用 `WORLD_SIZE>1`。

把错误对应到固定源码行，而不是只贴 traceback末行。

## 7. Megatron：从 functional recipe 缩小

Megatron GPU实验应在项目固定 CI container/`/opt/venv` 中，从同提交 `tests/functional_tests/test_cases` 和 recipe开始。不要在裸 host拼装 Transformer Engine/APEX/DeepEP后声称等价。

最小 dense学习矩阵：

| Run | TP | PP | CP | EP | 交付证据 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 1 | 1 | 1 | 1 | reference loss/update |
| B | 2 | 1 | 1 | 1 | Column/Row local shapes、TP collectives |
| C | 1 | 2 | 1 | 1 | layer ownership、1F1B P2P timeline |
| D | 1 | 1 | 2 | 1 | TE attention前置、context slices、DP×CP loss |
| E MoE | 1 | 1 | 1 | 2 | dense/expert rank generators、A2A splits、token imbalance |

固定 global batch/seq/model/precision/checkpoint起点；若某维要求模型 divisibility变化，记录后就不再声称严格数值同构。

### PP timeline 验收

对 `PP=4,m=8`，每 stage记录：

```text
warmup count from total_stages-current_stage-1
each recv_forward/local_forward/send_forward
each recv_backward/local_backward/send_backward
when no_sync exits and DP grad sync starts
activation buffers simultaneously live
```

实际序列必须能与固定 `schedules.py` 的 warmup/steady/cooldown逐项对上。

## 8. DeepSpeed：三个 stages 的同语义对照

用同一个 toy Transformer、initial checkpoint、global batch，运行 stage 0/2/3；stage 1可作为 optimizer-state隔离实验。配置见[DeepSpeed 源码主线](../internals/deepspeed-zero-flow)。

启动后立即打印：

```python
print(type(engine).__name__)
print(type(engine.optimizer).__name__)
print(engine.zero_optimization_stage())
print(engine.gradient_accumulation_steps())
```

若实际 optimizer class/stage与 resolved JSON不一致，停止；不要继续跑出一组无法解释的数据。

### Stage 3 trace

对一个三层模型记录首两 iterations：

```text
module ds_id order
immediate fetch params and bytes
prefetched params and bytes
wait duration
available/live params peak
release/keep reason (reuse/persist/external/active)
gradient reduce-scatter
optimizer subgroup update
```

首轮建立 coordinator trace，第二轮才更能代表 prefetch稳态。将 prefetch bucket减半做单变量实验；若吞吐下降但 HBM也下降，这是预期 tradeoff，不叫“优化失败”。

### GAS boundary gate

设置 GAS=4，每 microbatch都调用 `engine.backward()` 和 `engine.step()`，记录 parameter checksum：前 3 次不得更新，第 4 次才更新；overflow时即使 boundary也可跳过。上层 Trainer集成下确认它没有再额外 optimizer/scheduler step。

## 9. 统一两步数值门禁

所有框架都使用同一格式：

| 时刻 | 必存逻辑值 |
| --- | --- |
| init | selected logical parameter slices/norm/sum-squared |
| step 1 forward | global loss sum、valid token count、最终平均 |
| step 1 backward | global grad norm、selected logical gradient |
| step 1 update | update success/overflow、parameter checksum、LR |
| step 2 | 同上，确认差异没有结构性放大 |
| resume | load后的step/RNG/data cursor/optimizer，下一 update与连续run比较 |

容差要先声明：FP32/Gloo可严格很多，BF16/fused/不同 collective order不应要求 bitwise。容差不是掩盖 global batch或重复 loss divide导致的数量级错误。

## 10. 失败实验清单

| 故意破坏 | 预期检测层 | 不合格记录 | 合格记录 |
| --- | --- | --- | --- |
| 一 rank少一次 collective | timeout/flight recorder | “NCCL挂了” | group、sequence number、每 rank最后op/shape |
| mesh degree product错误 | config/mesh validate | 只贴assert | resolved degrees、公式、源码拒绝点 |
| rank-dependent unused branch | DDP/ZeRO hooks | 直接加barrier | graph差异、missing hook、正确unused策略 |
| PP tensor shape不同 | P2P contract | 只看receiver报错 | source/dest expected/actual shape |
| EP token严重倾斜 | performance而非correctness | “网络慢” | splits、expert histogram、GEMM与A2A timeline |
| FSDP/ZeRO prefetch过大 | HBM peak | “stage3也OOM” | current+prefetched full bytes与live budget |
| async save时修改state | checkpoint一致性 | “偶发坏档” | staging/update/save时序与next-step checksum |

## 11. Run bundle 模板

```text
run-id/
├── manifest.txt
├── resolved-config.json-or-yaml
├── rank-map.tsv
├── tensor-layouts.tsv
├── stdout-rank-*.log
├── losses-and-checksums.jsonl
├── memory.csv
├── profiler-trace/
├── checkpoint-metadata/
└── verdict.md
```

`verdict.md` 只回答：假设是什么；哪些 gate通过；哪些失败；证据路径；结论适用边界；下一个只改哪个变量。不要把一次 2-rank toy pass升级成“框架可稳定训练 70B”。

## 通关标准

完成后，你应拥有：

- 一个可重复的 DDP global-batch等价门禁；
- 一个 DTensor shard DCP round-trip与reshard矩阵；
- FSDP2 unshard/reshard/RS profiler trace；
- TorchTitan single/FSDP2/TP/PP单变量runs；
- Megatron TP algebra、PP timeline、CP/EP role traces；
- DeepSpeed stage 0/2/3状态与GAS边界对照；
- 至少三种故意失败的逐 rank证据；
- 一个 save→restart→load→next-step恢复门禁。

最后回到[42 天源码学习计划](../guide/source-study-plan)整理 Obsidian 交付物。
