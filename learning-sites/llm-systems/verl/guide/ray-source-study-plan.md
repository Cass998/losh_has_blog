---
title: 21 天 Ray 源码学习计划
description: 固定 Ray 与 veRL 提交，用 21 天完成官方文档、运行时源码、CPU 实验和 veRL V1 映射，每天都有问题、产物与验收标准
lesson:
  stage: 01 · 路线
  order: 06
  time: 21 天 · 每天 60–120 分钟
  level: 从使用到源码
  question: 怎样避免漫无目的地读 Ray monorepo，并最终能解释 veRL 的每个进程、资源和远程调用？
---

<script setup>
import rayRuntimeProbeUrl from '../practice/code/ray_runtime_probe.py?url'
</script>

# 21 天 Ray 源码学习计划

这不是“每天看几页文档”的打卡表。21 天只有一个目标：

> 能从源码解释 veRL V1 如何借助 Ray 启动控制器和训练 workers、如何预留多 rank 资源、一次方法调用怎样跨进程，以及故障时应该在哪一层找证据。

每一天都必须产生一个可复查产物。只阅读、不画调用链、不运行实验，不能算完成。

## 完成后的能力地图

```mermaid
flowchart LR
    A[官方概念契约] --> B[ray.init 与进程拓扑]
    B --> C[Task / Actor 调用链]
    C --> D[ObjectRef 与对象数据面]
    D --> E[Resource / Placement Group]
    E --> F[Fault / State API]
    F --> G[veRL TaskRunner / ResourcePool / WorkerGroup]
    G --> H[独立定位启动、资源、RPC、collective 问题]
```

## 固定版本并下载源码

本文计划固定：

```text
Ray  98c9eb75e44207266cc35f0cd94e1ecd58e3f77b
veRL e5687fce0516d31e1fdc4580499074a9bd94c751
```

Ray 是大型 monorepo。学习 Ray Core 与 veRL 集成无需一开始下载全部 history 和全部目录；使用 partial clone + sparse checkout：

```bash
mkdir -p framework-sources
cd framework-sources
git clone --filter=blob:none --no-checkout https://github.com/ray-project/ray.git
cd ray
git sparse-checkout init --cone
git sparse-checkout set \
  python/ray \
  src/ray/gcs \
  src/ray/raylet \
  src/ray/object_manager \
  src/ray/core_worker \
  src/ray/common \
  src/ray/protobuf \
  doc/source/ray-core \
  doc/source/ray-contribute
git checkout 98c9eb75e44207266cc35f0cd94e1ecd58e3f77b
test "$(git rev-parse HEAD)" = "98c9eb75e44207266cc35f0cd94e1ecd58e3f77b"
```

另建 veRL 源码副本或在现有 clone 中 checkout 固定提交：

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git checkout e5687fce0516d31e1fdc4580499074a9bd94c751
git rev-parse HEAD
```

如已有工作目录且包含未提交改动，不要直接 checkout 覆盖；另建只读 clone。

## 实验环境与边界

CPU 实验只依赖完整 Ray 安装，不需要 veRL、Torch 或 GPU：

```bash
python -m venv .venv-ray-course
source .venv-ray-course/bin/activate
python -m pip install --upgrade pip
python -m pip install 'ray[default]'
python -c 'import ray; print(ray.__version__)'
```

课程提供：下载 <a v-bind="{ href: rayRuntimeProbeUrl, download: 'ray_runtime_probe.py' }"><code>ray_runtime_probe.py</code></a>。先读[实验说明与证据模板](../practice/ray-observation-lab)，再运行；不要把页面的“预期语义”伪装成自己的实测输出。

## 学习记录模板

每天在自己的笔记库新建一张卡片：

```text
日期 / Ray commit / veRL commit
今天的问题：
入口 API：
关键 source:line：
进程边界：
状态 owner：
同步点：
故障与恢复者：
实验命令：
产物路径：
我仍不能解释的点：
```

每次源码阅读遵循同一顺序：Python public API → Cython/CoreWorker → raylet/GCS → remote worker → 返回值/失败路径。不要从整个仓库横向搜索一个词后随意跳转。

## 第一周：建立进程与调用心智模型

### Day 1：概念契约与术语消歧

**阅读**

- [Ray Key Concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)
- [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)
- [Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Objects](https://docs.ray.io/en/latest/ray-core/objects.html)
- [Ray 源码运行时总图](../internals/ray-source-runtime)

**问题**

- job、driver、task、actor、actor task、ObjectRef 分别是什么？
- Ray actor 与 RL actor 为什么只是同名？

**实践与产物**

画一张六个概念的关系图，并为每个概念填一个 veRL 对应物。不能把 `PPOTrainer` 填成 Ray actor。

**验收**

能在不使用“就是分布式”这种含糊表述的情况下，用一段话解释六个对象。

### Day 2：`ray.init()` 的两个分支

**源码**

- [`worker.py#L1438-L1525`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/worker.py#L1438-L1525)：public API 与参数
- [`worker.py#L1678-L1726`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/worker.py#L1678-L1726)：地址解析
- [`worker.py#L1828-L1996`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/worker.py#L1828-L1996)：本地启动与 connect-only

**问题**

`ray.init()` 在无 cluster 和已有 cluster 时分别创建什么？为什么在远端 head 上错误使用默认 `ray.init()` 可能得到另一套本地 runtime？

**实践与产物**

把两个分支画成 sequence diagram；标出第一次出现 `Node(head=True)` 与 `Node(connect_only=True)` 的位置。

**验收**

图中必须出现 address resolution、Node、connect、driver CoreWorker 四个节点。

### Day 3：Node 与 raylet 进程拓扑

**源码**

- [`node.py#L1232-L1412`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/node.py#L1232-L1412)：GCS 与 raylet 启动
- [`node.py#L1516-L1642`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/node.py#L1516-L1642)：head/per-node process 集合
- [`main.cc#L656-L867`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/raylet/main.cc#L656-L867)：raylet 内部组件
- [`main.cc#L1014-L1138`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/raylet/main.cc#L1014-L1138)：NodeManager、注册与启动

**问题**

CoreWorker、ObjectManager、Plasma 与 raylet 分别是不是独立 OS 进程？

**实践与产物**

建立“进程 / 内嵌组件 / 线程”三列表。若运行实验，再保存 `ps` 证据，但不要仅靠进程名猜源码结构。

**验收**

能解释为什么本固定提交里 ObjectManager 是 raylet 内组件。

### Day 4：普通 Task 的 Python 入口

**源码**

- [`remote_function.py#L345-L505`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/remote_function.py#L345-L505)
- [`remote_function.py#L516-L563`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/remote_function.py#L516-L563)
- [`core_worker.cc#L1893-L2073`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/core_worker.cc#L1893-L2073)

**问题**

为什么 `.remote()` 在 task 还没执行时就能返回 ObjectRef？TaskSpec 中哪些 options 由 Python 层传入？

**实践与产物**

运行实验脚本 baseline 模式，保存 JSON，并手写 `RemoteFunction._remote → CoreWorker::SubmitTask → AddPendingTask` 调用链。

```bash
python ../practice/code/ray_runtime_probe.py \
  --mode baseline \
  --output-dir ray-artifacts
```

路径按你保存脚本的位置调整。

**验收**

产物同时含 ref 的字符串/hex 与 `ray.get` 后的真实结果，不把两者混为一物。

### Day 5：worker lease 与分布式调度

**源码**

- [`normal_task_submitter.cc#L34-L95`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/normal_task_submitter.cc#L34-L95)：依赖等待
- [`normal_task_submitter.cc#L270-L347`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/normal_task_submitter.cc#L270-L347)：lease request
- [`cluster_lease_manager.cc#L194-L330`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/raylet/scheduling/cluster_lease_manager.cc#L194-L330)：feasible/available 与 retry
- [`local_lease_manager.cc#L356-L545`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/raylet/scheduling/local_lease_manager.cc#L356-L545)：worker 分配与 spillback

**问题**

谁选择 worker？GCS 是否为每个普通 task 做中央调度？feasible 与 available 有何不同？

**实践与产物**

画 `owner CoreWorker → local raylet → optional remote raylet → worker` 序列图，分别标注 dependency wait、resource wait、worker startup。

**验收**

能根据状态区分“依赖没 ready”“资源不可满足”“worker 启动失败”。

### Day 6：PushTask 与返回路径

**源码**

- [`normal_task_submitter.cc#L360-L542`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/normal_task_submitter.cc#L360-L542)
- [`worker.py#L961-L1040`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/worker.py#L961-L1040)
- [`worker.py#L2871-L3019`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/worker.py#L2871-L3019)

**问题**

lease 成功后 task bytes 是否仍要由 GCS 转发？`ray.get` 可能等待哪些阶段？

**实践与产物**

给 Day 5 图补全 `PushTask → task reply → ray.get/materialize`。列出至少三个同步/等待点。

**验收**

能解释 `.remote()` 异步与 `ray.get()` 同步的精确边界。

### Day 7：第一周闭卷复盘

**任务**

不看笔记，从空白画出：本地 `ray.init`、普通 task、ObjectRef 返回的完整链。然后用源码纠错。

**产物**

- 一张纠错前图；
- 一张纠错后图；
- 一页“我原来错误理解的 5 件事”。

**验收**

纠错后图至少包含 driver、CoreWorker、raylet、worker、GCS，并正确说明 GCS 在普通 task 调度中的边界。

## 第二周：Actor、对象与 Placement Group

### Day 8：Actor 创建链

**源码**

- [`actor.py#L1875-L1978`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/actor.py#L1875-L1978)
- [`actor.py#L2185-L2213`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/actor.py#L2185-L2213)
- [`actor_task_submitter.cc#L92-L164`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/actor_task_submitter.cc#L92-L164)
- [`gcs_actor_manager.cc#L660-L850`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/gcs/actor/gcs_actor_manager.cc#L660-L850)

**问题**

Actor 为什么需要 GCS 持久状态和 owner tracking？creation task 与 ordinary task 的生命周期差在哪？

**产物与验收**

画 actor 从 REGISTERING 到 ALIVE 的状态图，标出 state owner；能指出 actor 创建何处进入 GCS。

### Day 9：Actor method、顺序与直连

**源码**

- [`actor.py#L2478-L2600`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/actor.py#L2478-L2600)
- [`core_worker.cc#L2415-L2512`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/core_worker.cc#L2415-L2512)
- [`actor_task_submitter.cc#L167-L243`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/actor_task_submitter.cc#L167-L243)
- [`actor_task_submitter.cc#L534-L640`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/actor_task_submitter.cc#L534-L640)

**问题**

GCS 是否转发每个 actor method？谁维护 pending actor tasks 和 actor connection？

**实践与产物**

用 baseline JSON 证明同一 actor 的 PID/actor ID 与状态连续性；源码链证明直连路径。实测和源码证据必须分两列。

**验收**

能区分 actor creation、actor task submission、method execution 三段。

### Day 10：Actor 失败与重建

**阅读与源码**

- [Actor fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html)
- [`actor_task_submitter.cc#L378-L470`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/task_submission/actor_task_submitter.cc#L378-L470)
- [`gcs_actor_manager.cc#L1599-L1695`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/gcs/actor/gcs_actor_manager.cc#L1599-L1695)

**实践**

```bash
python ray_runtime_probe.py \
  --mode actor-crash \
  --output-dir ray-artifacts \
  --hold-seconds 60
```

同时保存 actor state。不要假设 PID 必然如何变化，以 JSON 和 state 输出为证据。

**验收**

用一句话解释：“Ray 重建 actor process”为什么不等于“恢复 veRL optimizer/step/RNG”。

### Day 11：ObjectRef 与 ownership

**源码**

- [`worker.py#L806-L897`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/_private/worker.py#L806-L897)
- [`core_worker.cc#L1055-L1119`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/core_worker.cc#L1055-L1119)
- [`reference_counter.cc#L223-L244`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/reference_counter.cc#L223-L244)

**问题**

ObjectRef 保存 bytes 吗？owner、borrower、ref count 分别解决什么生命周期问题？

**产物与验收**

画 “ID/owner metadata ↔ value location” 两层图。图中 ObjectRef 不得直接等同 Plasma pointer。

### Day 12：对象本地存储与跨节点移动

**源码**

- [`core_worker.cc#L1326-L1474`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/core_worker/core_worker.cc#L1326-L1474)
- [`object_manager.cc#L178-L259`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/object_manager/object_manager.cc#L178-L259)
- [Object fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html)

**问题**

CoreWorker memory store 与 Plasma 的路径为什么不能被一句“所有结果都进 object store”抹平？ObjectManager 什么时候参与？

**产物与验收**

用一张图分别画 small/direct result、shared object、remote pull；不声称一个未经源码确认的固定 size threshold。

### Day 13：Logical Resource 与调度需求

**阅读**

- [Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html)

**实践**

```bash
python ray_runtime_probe.py \
  --mode unschedulable \
  --wait-timeout 5 \
  --hold-seconds 90 \
  --output-dir ray-artifacts
```

另一个终端执行 `ray status` 和 `ray list tasks --detail`。

**产物与验收**

保存 JSON、status 和 task state；能说明 custom resource 是调度 label/capacity，不会自动创造硬件或安装环境。

### Day 14：Placement Group prepare/commit

**源码**

- [`placement_group.py#L25-L78`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/util/placement_group.py#L25-L78)
- [`placement_group.py#L131-L227`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/util/placement_group.py#L131-L227)
- [`gcs_placement_group_scheduler.cc#L41-L253`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/gcs/gcs_placement_group_scheduler.cc#L41-L253)
- [`placement_group_resource_manager.cc#L43-L155`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/raylet/placement_group_resource_manager.cc#L43-L155)

**实践**

```bash
python ray_runtime_probe.py \
  --mode placement-group \
  --hold-seconds 90 \
  --output-dir ray-artifacts
```

保存 `ray list placement-groups --detail --format yaml` 和 actors 输出。

**验收**

能用源码解释 `pg.ready()` 的 resolve 条件，并明确 PG 不会自动创建 actor 或 torch process group。

## 第三周：故障、观测与 veRL 集成

### Day 15：Placement Group 失败与重调度

**源码**

- [`gcs_placement_group_manager.cc#L211-L356`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/gcs/gcs_placement_group_manager.cc#L211-L356)
- [`gcs_placement_group_scheduler.cc#L410-L507`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/gcs/gcs_placement_group_scheduler.cc#L410-L507)
- [`gcs_placement_group_manager.cc#L690-L760`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/src/ray/gcs/gcs_placement_group_manager.cc#L690-L760)

**问题**

infeasible、temporarily unavailable、prepare failure、node death 各进入什么路径？为什么只能称“两阶段式”，不应把它直接当成数据库严格 2PC？

**产物与验收**

建立 PG failure matrix，至少包含状态、重试触发、资源释放、用户可见证据四列。

### Day 16：State API 的证据边界

**源码与文档**

- [State CLI](https://docs.ray.io/en/latest/ray-observability/reference/cli.html)
- [`api.py#L793-L875`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/util/state/api.py#L793-L875)
- [`api.py#L1020-L1061`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/util/state/api.py#L1020-L1061)
- [`state_aggregator.py#L312-L374`](https://github.com/ray-project/ray/blob/98c9eb75e44207266cc35f0cd94e1ecd58e3f77b/python/ray/dashboard/state_aggregator.py#L312-L374)

**问题**

为什么 list 输出可能被 limit、timeout、dropped events、truncation 影响？

**产物与验收**

制定一份现场采集清单：version、status、actors、tasks、PGs、node/PID、worker first traceback。明确 state 是快照，不是强一致审计日志。

### Day 17：veRL 入口与 Single Controller

**源码**

- [`main_ppo.py#L42-L95`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/main_ppo.py#L42-L95)
- [`main_ppo.py#L98-L150`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/main_ppo.py#L98-L150)
- [V1 逐源码主线](../internals/v1-source-walkthrough)

**问题**

driver、TaskRunnerV1、PPOTrainer 分别在哪个进程？`ray.get(runner.run.remote(config))` 的同步边界是什么？

**产物与验收**

画真实对象/进程图；图中 PPOTrainer 必须位于 TaskRunner actor 内，不能画成独立 Ray actor。

### Day 18：`RayResourcePool` 到 Placement Group

**源码**

- [`base.py#L113-L163`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/ray/base.py#L113-L163)
- [`base.py#L184-L243`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/ray/base.py#L184-L243)
- [`trainer_base.py#L613-L667`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/v1/trainer_base.py#L613-L667)

**问题**

哪些角色映射到 global/独立 pool？`max_colocate_count` 如何影响 logical GPU fraction？它为什么不是显存隔离？

**产物与验收**

拿一份 resolved config，写 `role → pool → PG → bundle → logical CPU/GPU` 表；不能只抄默认配置。

### Day 19：`RayWorkerGroup` 到 TrainingWorker rank

**源码**

- [`base.py#L339-L415`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/ray/base.py#L339-L415)
- [`base.py#L538-L683`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/ray/base.py#L538-L683)
- [`engine_workers.py#L76-L148`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/workers/engine_workers.py#L76-L148)

**问题**

`RayWorkerGroup` 自己是 Ray actor 吗？`TrainingWorker` 在哪一步被包装成 remote actor？谁设置 rank/world/master 环境？谁最终初始化 collective？

**产物与验收**

画 `PG bundle → Ray actor process → TrainingWorker → Engine → torch rank` 五层图；每层写 owner。

### Day 20：WorkerGroup fan-out、TQ 与权重通道

**源码**

- [`worker_group.py#L123-L255`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/base/worker_group.py#L123-L255)
- [`base.py#L49-L67`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/ray/base.py#L49-L67)
- [`base.py#L866-L894`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/single_controller/ray/base.py#L866-L894)
- [Ray 运行时、角色与资源编排](../internals/workers)

**问题**

一次 controller 方法如何分发到 N ranks？Ray、`torch.distributed`、TransferQueue、CheckpointEngine 各负责哪段？

**产物与验收**

追踪一个 `compute_log_prob` 或 `train_mini_batch` 调用，形成 source:line、进程、输入通道、输出通道、同步点表。必须明确 trajectory 不是默认完整塞在 Ray RPC return 中。

### Day 21：综合反推与故障演练

选择一个真实启动故障或构造如下案例：

```text
8-GPU WorkerGroup；PG 长期 PENDING；driver 与 TaskRunner ALIVE；没有 TrainingWorker PID。
```

完成以下交付：

1. 从 `main_ppo` 到 `pg.ready()` 的源码链；
2. 当前尚未启动的对象列表；
3. `ray status` / PG state 应采集的字段；
4. 至少三个候选原因，按证据排序；
5. 为什么此时查 NCCL log 大概率没有意义；
6. 资源修正后，从 workers 创建到 process group 初始化的下一条链；
7. 若 actor 随后崩溃，Ray restart 与 veRL checkpoint recovery 的职责分界。

**最终验收**

闭卷回答：

- 普通 task 的 worker lease 谁发、谁批、谁执行？
- actor 创建为何经 GCS，actor method 为何通常不经 GCS 转发？
- ObjectRef 与 bytes location 为什么是两层？
- PG 的 prepare/commit 与 `pg.ready` 是什么关系？
- `TaskRunnerV1`、`PPOTrainer`、`RayResourcePool`、`RayWorkerGroup`、`TrainingWorker` 哪些是远程进程？
- Ray logical resource、CUDA memory、torch process group 各解决什么问题？
- trajectory、ObjectRef 和权重同步分别走什么路径？

其中任何一题只能用比喻、不能给出 source:line，就回到对应 Day 重做。

## 最终交付目录

建议保留：

```text
ray-study/
├── versions.txt
├── diagrams/
│   ├── init.md
│   ├── normal-task.md
│   ├── actor.md
│   ├── objects.md
│   ├── placement-group.md
│   └── verl-mapping.md
├── source-notes/
│   └── day-01 ... day-21
├── ray-artifacts/
│   ├── baseline.json
│   ├── placement-group.json
│   ├── unschedulable.json
│   └── actor-crash.json
└── state-snapshots/
    ├── ray-status.txt
    ├── actors.yaml
    ├── tasks.yaml
    └── placement-groups.yaml
```

产物不是为了堆文件，而是让每个设计结论都有“官方契约 + 固定源码 + 可观察实验”三类独立证据。

下一步从 Day 1 开始，同时把[Ray 源码运行时总图](../internals/ray-source-runtime)当作索引，而不是一次读完就算掌握。
