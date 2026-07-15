---
title: 术语、源码锚点与一手资料
description: veRL V1 术语表、固定提交源码入口、强化学习原论文与 Ray/Hydra/推理系统官方资料
lesson:
  stage: 附录
  order: 24
  time: 按需查阅
  level: 参考
  question: 遇到版本、公式或系统细节争议时，应该回到哪一份一手证据？
---

# 术语、源码锚点与一手资料

本站是课程，不替代源码、测试、官方文档和论文。框架事实优先以固定提交 `e5687fce0516d31e1fdc4580499074a9bd94c751` 为准；论文用于解释方法假设；运行时行为还需用你的 resolved config、依赖版本和最小实验确认。

## 术语速查

| 术语 | 本站中的准确含义 |
| --- | --- |
| rollout | 用行为策略/LLM server 对 prompt 逐 token 生成一条或多条 trajectory |
| trajectory | prompt、response、mask、概率、reward 与环境/版本元数据组成的轨迹 |
| actor / policy | 被优化的策略模型角色；不要与 Ray actor 混淆 |
| rollout engine | 为训练收集轨迹的推理系统，可使用 actor 权重但具有独立运行形态 |
| reference | 冻结策略，用于 KL 锚定；按配置可省略 |
| critic | 预测 value 的模型/角色；GRPO 等配方通常可省略 |
| reward manager | 把 response 与 ground truth/模型/环境反馈转成 score |
| return | 从当前位置起的折扣累计奖励 |
| advantage | 动作/轨迹相对 baseline 好多少的策略更新信号 |
| old policy | 本轮更新的稳定概率参照，不应自动等同 rollout/reference |
| staleness | 生成轨迹的权重版本与当前训练版本之间的延迟 |
| TQ | TransferQueue，V1 的 trajectory KV 数据通道 |
| KVBatchMeta | 携带 partition、keys、tags 的轻量批次句柄，不是完整值本体 |
| TensorDict | 按 batch 维组织 tensor/non-tensor 字段的容器 |
| DataProto | veRL 批协议；V0 主干，V1 在共享算法/接口边界仍使用 |
| WorkerGroup | 对一组 Ray worker/rank 统一分发计算的控制抽象 |
| Ray actor | 有状态远程 worker 进程抽象，不等于 RL actor policy |

## 固定提交的源码入口

- [veRL `e5687fce` 完整树](https://github.com/verl-project/verl/tree/e5687fce0516d31e1fdc4580499074a9bd94c751)
- [`main_ppo.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/main_ppo.py)：Hydra、Ray、TaskRunnerV1 入口
- [`ppo_trainer.yaml`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/config/ppo_trainer.yaml)：默认 V1、mode、TransferQueue 与全局配置
- [`trainer_base.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/v1/trainer_base.py)：V1 init/fit/step、字段读写与扩展 hooks
- [`trainer_sync.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/v1/trainer_sync.py)：同步模式权重与 rollout 生命周期
- [`agent_loop_tq.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/v1/agent_loop_tq.py)：AgentLoop output 写入 TQ
- [`replay_buffer.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/v1/replay_buffer.py)：prompt group、采样与 staleness
- [`core_algos.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/core_algos.py)：advantage/policy loss registry 与公式实现
- [`agent_loop.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/experimental/agent_loop/agent_loop.py)：AgentLoopBase/Manager 与多轮生成
- [`reward_loop.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/experimental/reward_loop/reward_loop.py)：并发 reward 计算组织
- [`protocol.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/protocol.py)：DataProto/dispatch 协议
- [`rl_dataset.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/utils/dataset/rl_dataset.py)：数据读取与 chat/token 契约
- [`test_core_algos_on_cpu.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/tests/trainer/ppo/test_core_algos_on_cpu.py)：算法 registry 与 estimator CPU 测试范式
- [`test_replay_buffer_on_cpu.py`](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py)：V1 sampler 行为证据

## veRL 官方文档

官方 `latest` 会继续演进，可能晚于本站绑定提交。用它理解设计和现行建议，再回固定源码核对类名/字段。

- [veRL 官方文档首页](https://verl.readthedocs.io/)
- [PPO on GSM8K quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)：官方数据准备与首个训练入口
- [How to Extend veRL](https://verl.readthedocs.io/en/latest/extend_guide.html)：reward、tool、AgentLoop、ReplayBuffer、manager 与后端扩展
- [Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)：多轮/TITO 设计与 server 调用
- [Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)：分布式与混合 reward 设计
- [TransferQueue Data System](https://verl.readthedocs.io/en/latest/data/transfer_queue.html)：V1 数据通道背景
- [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)：配置字段参考
- [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)：动态 batch、并行与显存/吞吐优化建议
- [veRL Profiler System](https://verl.readthedocs.io/en/latest/perf/verl_profiler_system.html)：全局/角色级 profiler 配置与扩展方式

## 强化学习与系统论文

### 核心方法

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：PPO surrogate/update 动机
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：GAE 的 bias/variance 折中
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 与数学推理训练
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)：Dr.GRPO、长度与归一化偏差分析
- [DAPO](https://arxiv.org/abs/2503.14476)：decoupled clip、dynamic sampling、token-level loss、overlong shaping recipe
- [Back to Basics / RLOO](https://arxiv.org/abs/2402.14740)：leave-one-out baseline 的 RLHF 分析

### 框架与推理系统

- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)：veRL 的 hybrid controller/dataflow 设计来源
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：动态 KV cache 管理与高吞吐 serving

论文公式与框架实现不应机械逐行相同。检查它属于 advantage、reward、sampling、policy loss 还是 system 层，再判断是等价改写、工程近似、可选 recipe 或版本变化。

## 分布式与配置的一手文档

- [Ray Core key concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)：task、actor、object 和资源的官方定义
- [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html)：有状态 worker 与故障语义
- [Ray logical resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)：CPU/GPU resource request 与调度
- [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)：job/task/actor/log/metrics 观测
- [Hydra Defaults List](https://hydra.cc/docs/advanced/defaults_list/)：组合顺序、override 与 `_self_`
- [Hydra terminology / overrides](https://hydra.cc/docs/advanced/terminology/)：CLI override 与配置树

技术实现问题优先引用这些官方文档，而不是不带版本的二手博客；但 veRL 如何使用 Ray/Hydra，仍以当前源码和 resolved config 为最终证据。

## 怎样处理资料冲突

按以下顺序：

1. 打印实际导入路径、commit 与 dirty diff；
2. 保存 resolved config 与依赖版本；
3. 在固定提交源码和测试中定位行为；
4. 用官方文档理解接口设计与版本迁移；
5. 用原论文核对方法假设和公式；
6. 写一个 tiny input/one-step 实验确认运行行为。

“文档说”“论文说”“代码里有这个类”都不是单独充分证据；必须先对齐版本与配置。

## 本课程的内容边界

课程重点是默认 V1 的 PPO/GRPO 学习闭环，没有穷举所有模型 engine、rollout backend、多模态、蒸馏、完全异步 recipe 和硬件组合。以下情况必须回源码/测试重新建契约：

- 改用 `trainer.use_v1=false`；
- 自定义 AgentLoop/Manager、ReplayBuffer、RewardManager；
- 使用非默认 policy loss、rollout correction 或新 estimator；
- 使用 Megatron/VeOmni/TorchTitan、MoE、NPU 或多机 RDMA；
- 升级 commit 后入口、字段、配置默认值或后端 API 改变。

发现本站与绑定提交不一致时，以可执行测试和源码为准；修正文档时同时记录旧/新 commit 与行为差异。
