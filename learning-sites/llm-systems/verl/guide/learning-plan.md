---
title: 六周学习计划：从零到能改 veRL
description: 每周目标、每日任务、产出物和通关条件明确的 veRL 渐进学习计划
lesson:
  stage: 00 · 定位
  order: 01
  time: 20 分钟规划
  level: 路线图
  question: 每天学什么、做什么，六周后才能真的具备修改框架的能力？
---

# 六周学习计划：从零到能改 veRL

这份计划默认你会 Python、用过 PyTorch 或 Transformers，但没有系统学过强化学习；每周安排 5 天、每天 60–90 分钟，周末留给补课或复盘。目标不是六周“精通所有后端”，而是获得三项可迁移能力：解释算法、跟踪数据、用证据改系统。

若你已经通过某周的通关检查，可以跳过阅读，但仍建议完成该周产出物。

## 全程只维护一份学习档案

新建 `learning-log.md`，固定六个小节：

```markdown
## 我现在的心智模型
## 字段词典：字段 / shape / 写入者 / 读取者
## 调用链：函数 / 输入 / 输出 / 进程
## 实验账本：假设 / 改动 / 指标 / 结论
## 失败记录：现象 / 最小证据 / 根因 / 修复
## 仍然不会的问题
```

这份档案比“读完多少页”更能说明进度。每周只更新同一份，不另开零散笔记。

## 第 1 周：把 LLM 看成会做动作的策略

**本周问题：** 文本生成为什么能被强化学习训练？

| 天 | 阅读与任务 | 当天产出 |
| --- | --- | --- |
| Day 1 | [LLM 如何成为策略](/verl/fundamentals/llm)：手算一个三词词表的 softmax 与 log-prob | 写出 prompt、state、token action 的对应关系 |
| Day 2 | [够用的数学工具](/verl/fundamentals/math-toolkit)：期望、对数、梯度、重要性比率 | 完成页面内四道手算题 |
| Day 3 | [强化学习的完整闭环](/verl/fundamentals/rl)：reward、return、value、advantage | 画一条 4-token 轨迹并标字段 |
| Day 4 | [策略梯度这座桥](/verl/algorithms/policy-gradient)：解释 \(A>0\) 与 \(A<0\) | 用一句人话和一条公式解释更新方向 |
| Day 5 | 复盘：不看网页重画“生成 → 评价 → 更新” | 一页心智模型 v1 |

**通关条件：** 看到 `old_log_probs`、`response_mask`、`advantages` 时，能说明它们为何存在，而不只是翻译名字。

::: warning 不要在本周做的事
不用推导 Transformer，也不用证明策略梯度定理。你需要的是足够准确地预测数据流；证明可以作为以后选修。
:::

## 第 2 周：让 PPO 与 GRPO 公式变成张量

**本周问题：** 为什么不能简单地“奖励高就加大梯度”？

| 天 | 阅读与任务 | 当天产出 |
| --- | --- | --- |
| Day 1 | [PPO 与 GAE](/verl/algorithms/ppo)前半：TD error、GAE、return | 标出 \([B,T]\) 中哪些位置被 mask |
| Day 2 | PPO clipped objective：手算正负 advantage 下的 clipping | 一张 ratio/clip 决策表 |
| Day 3 | [GRPO、Dr.GRPO 与 DAPO](/verl/algorithms/grpo-family)：组内基线 | 手算同一 prompt 的 4 个 reward |
| Day 4 | 对照 `core_algos.py`：公式符号逐一映射到参数 | 源码字段词典 v1 |
| Day 5 | 选择 PPO 或 GRPO，口述一遍完整算法 | 3 分钟录音或文字讲解 |

**通关条件：** 能解释 PPO 是更新约束，GAE/GRPO 是优势估计方式；也能解释 DAPO 是一组配方，不是单个 `adv_estimator`。

## 第 3 周：跑通第一个可验证实验

**本周问题：** “程序在跑”与“训练语义正确”之间差什么证据？

| 天 | 阅读与任务 | 当天产出 |
| --- | --- | --- |
| Day 1 | [第一次可验证实验](/verl/practice/first-run)：检查环境、数据和版本 | 环境清单与固定提交 |
| Day 2 | 只准备/检查 Parquet；打印一条样本和 token 长度 | 数据契约表 |
| Day 3 | [读懂 Hydra 配置](/verl/practice/configuration)：先解析再启动 | 保存完整解析配置 |
| Day 4 | 跑到第一个训练 step；保留 rollout 样本和关键指标 | smoke-run 证据包 |
| Day 5 | [写一个可靠奖励函数](/verl/practice/reward-function)：先单测再接训练 | 至少 6 个边界测试 |

**通关条件：** 换一台机器时，你能说清模型、数据、命令、配置、环境和提交版本，而不是只给一段终端截图。

没有合适 GPU 也能完成 Day 1–3 和奖励函数测试；不要为了“必须训练”跳过数据与配置理解。

## 第 4 周：沿一个 `uid` 读完 V1 主链

**本周问题：** 一条样本到底存在哪里、由谁补齐字段？

| 天 | 阅读与任务 | 当天产出 |
| --- | --- | --- |
| Day 1 | [如何接入完整 LLM 系统](/verl/guide/llm-systems-integration) + [版本边界](/verl/guide/version-boundary) + [源码地图](/verl/guide/source-map) | 固定 commit，画出 SFT → rollout → RL update → weight sync 边界 |
| Day 2 | [整体架构](/verl/internals/architecture) + [入口与初始化](/verl/internals/entry-and-init) | 区分 Single Controller、HybridFlow、Ray actor、训练/推理进程 |
| Day 3 | [V1 逐源码主线](/verl/internals/v1-source-walkthrough)前半 + [从推理到训练](/verl/internals/rollout-to-update)前半 | `main_ppo.py` → TaskRunner → trainer → rollout/reward 时序图 |
| Day 4 | 两页后半 + [训练/推理后端契约](/verl/internals/backend-contracts) | old/ref/value/advantage/update 字段生命周期，外加 backend 输入/输出/换权契约 |
| Day 5 | 在固定源码中从 `main_ppo.py` 追到 `_update_actor()` 和下一轮 rollout | 带文件、函数、进程、字段和权重版本的完整调用链 |

**通关条件：** 能指出 `_step_once()` 九个阶段；能区分 `KVBatchMeta` 句柄、TransferQueue 中的值、临时 DataProto 和 worker 读取的 TensorDict。

## 第 5 周：理解分布式组织，而不是背 Ray 名词

**本周问题：** 控制器、角色、进程、GPU 和推理后端怎样组合？

| 天 | 阅读与任务 | 当天产出 |
| --- | --- | --- |
| Day 1 | [数据结构与组织](/verl/internals/data-structures)：画出数据所有权 | 谁持有 value、谁只持有 key |
| Day 2 | [Worker 与资源编排](/verl/internals/workers) + [Ray 固定源码主线](/verl/internals/ray-source-runtime) | Ray task/actor/PG → TaskRunner/ResourcePool/WorkerGroup 映射表 |
| Day 3 | [Ray CPU 观测实验](/verl/practice/ray-observation-lab)：正常 task/actor/PG 与不可调度/崩溃反例 | JSON、State CLI、PID/actor id/PG bundle 证据包 |
| Day 4 | 对照 resolved config 分析 colocate/separate、TQ 与权重同步，再读[故障定位](/verl/practice/debugging) | 部署取舍表 + 按进程/通道分层的最小证据清单 |
| Day 5 | 改错一个资源或启动条件，预测首个失败状态后在测试环境验证 | 一份含预测、反证、修复和回归的失败记录 |

**通关条件：** 遇到卡住、OOM 或 worker died 时，先定位阶段和角色，再调参数；能说出哪种证据会推翻自己的初始猜测。

如果你的主要职责是集群、调度或 veRL 资源编排，不要把 Day 2 压缩成一天；改走[21 天 Ray 源码计划](/verl/guide/ray-source-study-plan)，完成 GCS/raylet/CoreWorker/ObjectRef/Placement Group 的逐层反推。

## 第 6 周：做一次最小但完整的框架改造

**本周问题：** 怎样改功能或性能，同时保住算法语义和可回退性？

| 天 | 阅读与任务 | 当天产出 |
| --- | --- | --- |
| Day 1 | [扩展点地图](/verl/customization/extension-map)：选最小侵入层 | 一页设计：目标/契约/不变量 |
| Day 2 | [实现自定义算法](/verl/customization/custom-algorithm)或自定义 reward/replay | 单元测试先失败后通过 |
| Day 3 | 用 tiny batch 集成，检查字段、shape、有限值与指标 | 正确性证据 |
| Day 4 | [优化方法论](/verl/customization/optimization-playbook) + [性能采集](/verl/practice/profiling) | 固定 workload 的基线 profile |
| Day 5 | 做一项 A/B 改动；失败也要写结论 | 改动、数据、结论与回退方案 |

**通关条件：** 你的改动同时具备：明确扩展边界、自动测试、最小集成实验、算法不变量、基准数据和回退开关。

## 时间不够时的两条压缩路线

### 每周只有 3 小时

用 8 周完成，每周只做“核心阅读 60 分钟 + 手算/源码 60 分钟 + 产出物 60 分钟”。删掉拓展论文，不删通关检查。

### 已会 RL，只想读 veRL 源码

先完成入场诊断中的前五题；若能准确回答，按“第 3 周 → 第 4 周 → 第 5 周 → 第 6 周”走。第 2 周只用来核对当前配置实际选择的 estimator 与 loss。

## 每周五分钟复盘模板

```text
本周我能预测的新事情：
本周被源码推翻的一条理解：
我仍无法解释的一个字段：
下周最小可验证动作：
我将删除或推迟的旁支：
```

学习计划不是承诺读完所有页面，而是承诺每周产生一个可检验结果。下一步如果仍不确定术语，先读[小白常见问题](./common-questions)；否则直接开始[LLM 如何成为策略](/verl/fundamentals/llm)。
