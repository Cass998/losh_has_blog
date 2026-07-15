---
title: 扩展点地图：先选最小改动边界
description: 从配置、数据、奖励到采样器、算法、AgentLoop、训练与推理后端，系统梳理 veRL 的扩展层和验证责任
lesson:
  stage: 05 · 改造
  order: 19
  time: 45–60 分钟
  level: 进阶
  question: 想实现新任务或新算法时，应该改哪一层，怎样避免把局部需求变成框架分叉？
---

# 扩展点地图：先选最小改动边界

真正理解框架，不是知道哪里都能改，而是知道**哪里不必改**。同一个目标常有多条路径：格式分数可以写 reward；动态过滤可能属于 ReplayBuffer；多轮工具交互属于 AgentLoop；新的优势基线才进入算法层。

选择扩展点的原则：在能表达需求的前提下，选拥有最小契约、最容易单测、最少影响分布式状态的一层。

## 先用人话：不要为了换裁判而重建体育馆

如果只是更换评分规则，写 reward function；如果比赛流程加入工具交互，改 AgentLoop；如果参赛样本选择规则变化，改 sampler；只有目标函数本身变化，才碰 estimator/policy loss。

每向下走一层，获得更多控制，也承担更多责任：shape、mask、权重版本、进程导入、资源放置、失败清理和性能回归。

## 九层扩展阶梯

| 层级 | 适合解决什么 | 当前主要入口 | 最低验证责任 |
| --- | --- | --- | --- |
| 1 配置 | batch、长度、后端、并行、现有 recipe | Hydra override/component YAML | resolved config + one-step run |
| 2 数据 | 新 schema、chat template、多模态预处理 | `RLHFDataset` / `data.custom_cls` | schema、token 长度、截断测试 |
| 3 奖励 | 规则、verifier、RM、组合分数 | `reward.custom_reward_function` / Reward Loop | 反例、超时、安全、人工审计 |
| 4 工具/交互 | 多轮、工具、环境反馈 | `AgentLoopBase` / tool config | token 连续性、终止、资源释放 |
| 5 采样 | 过滤、优先级、staleness、oversampling | `ReplayBuffer` subclass | 完整 prompt group、不偏置/偏置说明 |
| 6 训练时序 | step 前后动作、同步/异步生命周期 | V1 trainer hooks/subclass | 状态机、异常清理、同步语义 |
| 7 算法 | 新 advantage、policy loss、baseline | registries in `core_algos.py` | tiny tensor 等式、mask、finite、梯度 |
| 8 Agent 框架 | 替换整套 rollout 调度 | custom AgentLoopManager | 非阻塞提交、TQ trajectory 契约 |
| 9 后端/硬件 | 新推理引擎、训练 engine、芯片 | external module/plugin registries | 分布式集成、权重同步、性能/精度 |

下面不是 API 百科，而是帮助你做边界判断。

## 1–3：多数任务改造到这里已经够了

### 只改配置

现有 estimator、policy loss、rollout backend 和 model engine 能组合出目标时，优先写一个可审计的 recipe YAML/shell，不要复制 trainer。先保存 resolved config，确认角色联动。

### 自定义数据集

当默认 `RLHFDataset` 无法表达字段或预处理时，用：

```yaml
data:
  custom_cls:
    path: /shared/my_dataset.py
    name: MyDataset
```

扩展前先问：是否只需在 Parquet 预处理时新增 `extra_info`？能离线完成的工作不必放进每个 dataloader worker。

### 自定义 reward

通过 path/name 加载纯函数，适合大多数新任务。它的契约是 response、ground truth、data source 与 extra info，不需要知道 actor 如何分片。详见[写一个可靠奖励函数](/verl/practice/reward-function)。

## 4：改变“模型如何与世界互动”

内置工具能力已支持函数工具和有状态工具。只有内置 ToolAgentLoop 无法表达状态机时，才继承 `AgentLoopBase` 并实现 `run()`。

```python
class MyAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **dataset_fields) -> AgentLoopOutput:
        ...
```

关键不变量：

- 优先 token-in/token-out；不要 decode 后再 encode 造成轨迹 token 与 log-prob 对不齐；
- 明确 EOS、最大轮次、工具异常和取消时如何终止；
- 归还 `response_ids`、`response_mask`、log-prob 与 reward 所需字段；
- sandbox、网络、文件和凭据都有资源/安全边界；
- 任一 coroutine 失败不会让同组永远停在 running。

[官方 Agent Loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)提供了当前接口背景；实际实现仍以固定提交的 `experimental/agent_loop` 为准。

## 5：改变“哪些轨迹进入下一次更新”

V1 的 `_build_replay_buffer()` 支持按路径加载 `ReplayBuffer` 子类：

```yaml
trainer:
  v1:
    sampler:
      custom_sampler:
        path: /shared/my_sampler.py
        name: MyReplayBuffer
      sampler_kwargs:
        min_reward_gap: 0.25
```

自定义 `sample(global_steps, partition_id, batch_size)` 时，不只是在 list 中挑 key。你必须维护：

- prompt group 完整性（尤其 GRPO 的 n 个 session）；
- finished/failure/pending 状态与清理；
- `global_steps` 和 staleness 约束；
- padding tag 与有效样本计数；
- 返回 `KVBatchMeta` 和可解释的辅助指标；
- 被过滤轨迹的 TQ 生命周期，避免泄漏存储。

动态采样会改变训练数据分布，它既是系统调度又是算法设计。把过滤率、难度/来源分布和 reward 方差加入 guardrail。

## 6：改变训练时序，先使用 hooks

V1 基类提供：`on_init_end`、`on_train_begin/end`、`on_validate_begin/end`、`on_step_begin/end`、`on_sample_begin/end`。

适合：在固定边界记录指标、切换模型引擎状态、控制权重同步或实现 mode 生命周期差异。若只需 step 前后行为，不要复制整个 `fit()`；复制训练循环会让 checkpoint、validation、TQ clear 和未来修复全部分叉。

必须画出异常路径：hook 中途抛错时，LLM replicas、TransferQueue、后台 executor 和 checkpoint 状态如何收尾。

## 7：改变训练目标，拆成 advantage 与 loss

问两个问题：

1. 我改变的是“每条/每个 token 相对基线好多少”吗？进入 advantage registry。
2. 我改变的是“给定 advantage 后怎样限制概率、聚合 token”吗？进入 policy loss registry。

不要把 sampling、reward shaping、advantage 和 loss 全塞进一个大函数。这样无法分别做等价测试，也很难判断 ablation 在改变什么。

具体实现见[实现自定义算法](./custom-algorithm)。

## 8：替换 AgentLoopManager 是一条系统边界

`TaskRunnerV1.init_agent_loop_manager()` 支持配置 fully-qualified class：

```bash
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=my_pkg.manager.MyAgentLoopManager
```

自定义 manager 的核心契约不是“返回一个大 batch”，而是：`generate_sequences(prompts)` 非阻塞提交，并在完成后把 trajectory 字段与 tag 写入 TransferQueue。ReplayBuffer 会在另一侧等待可采样 group。

这适合接入已有 agent framework；若只是多轮工具逻辑，优先改 AgentLoop，不要先替换 manager。

## 9：新训练/推理后端使用外部模块入口

固定源码在 `verl/__init__.py` 读取：

```bash
export VERL_USE_EXTERNAL_MODULES=my_package.register
```

模块导入时可向 rollout replica/server adapter、training engine 等 registry 注册实现。它必须在 driver 和所有 Ray worker 可导入；多机上依赖包、版本和环境变量要一致。

后端扩展责任远超“实现 generate”：还包括模型加载、并行拓扑、sleep/wake、KV cache、取消、健康检查、权重版本与 checkpoint engine 同步、dtype/数值一致性和故障清理。

## 变更前写一张契约卡

```text
目标：我具体想改变什么可观察行为？
层级：为什么现有更上层扩展点不够？
输入：字段 / shape / dtype / device / 权重版本
输出：字段 / shape / tag / 指标
不变量：mask、分组、有限值、顺序、清理
失败：超时/异常/空结果怎样传播与回收？
测试：纯函数 → 组件 → one-step → A/B
回退：哪个开关能恢复基线？
```

若“为什么更上层不够”写不出来，先停在更小边界。

## 选择例子

| 需求 | 推荐首选层 | 不应先做 |
| --- | --- | --- |
| 代码题接沙箱评分 | reward + sandbox | 改 PPOTrainer |
| 同题全对/全错不训练 | custom ReplayBuffer/recipe | 在 policy loss 静默丢 token |
| 新组内 baseline | advantage registry | 新建完整 trainer |
| 每条回答等权 | policy loss aggregation | 改 reward 数值补偿长度 |
| 新工具协议 | Tool/AgentLoop | 改 rollout engine kernel |
| 接入已有 agent 平台 | AgentLoopManager | 把整个平台塞进 reward function |
| 新推理引擎 | external backend registry | 在 trainer 中写大量 `if backend` |

## 通关检查

选择你真实想做的一项改动，写契约卡并指出最小层级。然后列出改动会影响的三类指标：任务正确性、算法分布、系统资源。下一步如果目标是算法，进入[实现自定义算法](./custom-algorithm)；如果目标是吞吐，进入[优化方法论](./optimization-playbook)。
