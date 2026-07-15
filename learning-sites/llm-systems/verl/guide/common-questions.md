---
title: 小白常见问题：先拆掉十五个理解障碍
description: 用直白语言回答学习 LLM 强化学习和 veRL 时最容易卡住的问题，并给出准确的专业边界
lesson:
  stage: 00 · 定位
  order: 02
  time: 25–40 分钟
  level: 零基础可读
  question: 那些听起来相似的 RL 和 veRL 术语，到底分别在说什么？
---

# 小白常见问题：先拆掉十五个理解障碍

这一页不是术语表。每个问题都先给一个可以拿来推理的短答案，再补专业边界。第一次读只看“短答案”；真正遇到对应代码时再看后半段。

## 1. 我需要先学完高数、概率论和传统 RL 吗？

**短答案：不需要先学完，但不能完全绕开。** 读 veRL 的最低数学门槛是理解概率、对数、均值/方差、期望、导数方向和张量 shape；最低 RL 门槛是分清策略、奖励、回报、价值和优势。

你暂时不需要证明策略梯度定理，也不需要先做 Atari。课程会在[够用的数学工具](/verl/fundamentals/math-toolkit)中只补当前调用链会用到的部分。每当公式出现，先问它“输入是什么、输出是什么、正负号改变什么”，再追求推导。

## 2. LLM 只是在续写文字，为什么算强化学习？

**短答案：因为每次选择下一个 token 都可以看成一次动作。** 已有 prompt 和已经生成的 token 是状态，词表中的下一个 token 是动作，整段回答结束后环境或评分器给奖励。

专业地说，策略 \(\pi_\theta(a_t\mid s_t)\) 是语言模型在上下文 \(s_t\) 下对 token \(a_t\) 的条件分布。轨迹是 token 序列。LLM RL 常常只有终局结果奖励，因此 credit assignment 比“每步都有反馈”的环境更困难。

## 3. SFT 和 RL 到底差在哪里？

**短答案：SFT 学“示范里写了什么”，RL 学“自己写出的结果哪个好”。**

SFT 对固定目标 token 做最大似然训练；每个位置都有老师给出的答案。RL 先从当前策略采样回答，再通过规则、奖励模型或环境评价；训练信号取决于采样结果与基线。SFT 数据可以离线重复很多次，on-policy RL 则关心样本由哪一版策略产生。

## 4. reward 高，为什么不直接把 reward 当 loss？

**短答案：reward 通常是对离散文本的评价，不能直接对“选了哪个 token”求导。** 我们能求导的是模型给这个 token 的 log-prob，因此使用 \(\log \pi_\theta(a_t\mid s_t)\) 乘上优势来决定提高或降低其概率。

reward 还缺少参照：得到 0.8 是好是坏，取决于同题其他回答、价值函数或历史基线。advantage 正是在表达“比基线好多少”。

## 5. reward、return、value、advantage 是一回事吗？

**短答案：不是。**

| 量 | 问的问题 | 一个直觉 |
| --- | --- | --- |
| reward \(r_t\) | 此刻得到什么反馈？ | 这一步/整段得几分 |
| return \(G_t\) | 从现在起总共能得到多少？ | 后续折扣奖励之和 |
| value \(V(s_t)\) | 在这个状态通常能得多少？ | 赛前预期分数 |
| advantage \(A_t\) | 这次动作比通常水平好多少？ | 实际表现减预期 |

GAE 用 critic 的 value 构造 advantage；GRPO 用同一 prompt 的一组回答作为相对基线，通常不需要 critic。

## 6. actor、rollout、reference、critic、reward model 分别是什么？

**短答案：它们是职责，不一定永远对应五份独立模型或五组 GPU。**

- **actor/policy**：正在被优化的策略。
- **rollout engine**：用 actor 权重高吞吐地生成轨迹，常由 vLLM/SGLang 等推理后端实现。
- **reference policy**：冻结参照，用于限制策略偏离；启用相关 KL 设计时才需要。
- **critic**：估计 value；GAE/PPO 常用，GRPO 类方法通常可省略。
- **reward**：可能是规则函数、判别式/生成式奖励模型，或外部环境。

veRL 可以让一些角色共置、分离或共享权重来源，所以不要看到名字就假设它一定占一整张独立 GPU。

## 7. rollout 不就是 inference 吗？

**短答案：计算动作很像，目的和输出契约不同。** 普通推理只要把文字返回给用户；训练 rollout 还要保留 token ids、mask、生成策略概率、样本身份、奖励所需元数据等，使后续能重算概率并建立训练信号。

因此，“部署模型能回答”不代表“rollout 能训练”。后者还受组采样、权重版本、随机性、截断和训练数据契约约束。

## 8. 为什么已经生成过，还要重新算 `old_log_probs`？

**短答案：推理后端给出的概率与训练后端严格重算的概率可能存在差异，PPO 更新还需要一个稳定的旧策略参照。**

在本站固定的 V1 源码中，`_step_once()` 会在采样后调用 `_compute_old_log_prob()`。rollout 侧也可能保留 `rollout_log_probs`，用于检测/修正推理策略与训练策略的偏差。两者不要凭名字混为同一个张量。

## 9. on-policy 是不是要求生成完立刻只更新一次？

**短答案：不是这么机械。** 它要求用于梯度估计的数据与当前策略足够接近；PPO 会对同一批样本做多个 epoch，但用 ratio/clipping 控制偏离。

异步生成中，权重版本和样本 staleness 更重要。系统吞吐提高可能让样本更旧，所以“更异步”不是纯系统优化，还可能改变算法分布。

## 10. GRPO 为什么同一道题要生成多份回答？

**短答案：它需要组内比较来构造基线。** 如果同题四份回答得分为 `[1, 1, 0, 0]`，高于组均值的回答得到正优势，低于均值的得到负优势。

如果一组全对或全错，减去组均值后优势接近 0，几乎没有学习信号。这也是动态采样/过滤会出现的原因，但过滤规则会改变有效数据分布，必须记录。

## 11. PPO、GRPO、DAPO 是平级算法开关吗？

**短答案：不完全是。** PPO 常指带 clipped policy objective 的更新方法；GRPO 重点改变优势基线，仍可使用 PPO 风格的 ratio/clipping；DAPO 是 decoupled clipping、动态采样、token-level loss、overlong shaping 等技术组成的 recipe。

所以在 veRL 中不要期待 `adv_estimator=dapo` 就代表完整 DAPO。你需要分别核对采样、奖励、advantage 与 policy loss 配置。

## 12. Ray actor、veRL actor 和模型 actor 是不是同一个 actor？

**短答案：不是，这是最危险的同名词之一。**

- Ray actor：一个有状态的远程 Python worker 进程抽象；
- veRL 的 worker/role：训练系统中的职责与封装；
- RL actor/policy：被优化的策略模型。

一句话中出现 actor 时，先问它属于“分布式运行时、框架组织还是算法角色”。

## 13. `DataProto`、TensorDict、TransferQueue、`KVBatchMeta` 都叫 batch 吗？

**短答案：它们描述不同层次。** TensorDict 保存批字段及 batch 维；TransferQueue 按 key 存取字段；`KVBatchMeta` 主要携带 partition、keys、tags，是指向数据的轻量句柄；DataProto 仍在算法/兼容边界把 tensor 与 non-tensor 数据组合起来。

调试时先打印对象类型，再谈字段。把 `KVBatchMeta` 当成已经含完整 tensor 的 batch，会完全误判通信与内存。

## 14. 没有多卡或 24GB GPU，还能学吗？

**短答案：能完成大部分理解和许多关键验证。** 你可以：检查数据 schema、解析 Hydra 配置、单测奖励、手算优势、读调用链、为算法函数写 tiny tensor 测试。真正的端到端训练可在有资源时补做。

不要用生产规模作为第一步。第一次实验的目的，是让一个最小链路产生可信证据，而不是复现论文指标。

## 15. 报错、OOM 或一直卡住时，第一反应是什么？

**短答案：先确定最后一个成功阶段，而不是立刻改十个参数。**

按顺序问：配置是否成功解析？数据是否读入？Ray 角色是否创建？模型是否加载？rollout 是否返回？reward 是否有限？advantage 是否非零？actor 是否完成一步？权重是否同步？

每次只收集能区分两种假设的证据。具体决策树见[从现象定位故障](/verl/practice/debugging)。

## 把问题变成下一步

如果第 2–5 题仍模糊，进入[LLM 如何成为策略](/verl/fundamentals/llm)；如果主要卡在第 6–13 题，先跑[第一次可验证实验](/verl/practice/first-run)，再读[源码地图](./source-map)。不要继续收集术语——去观察一条真实样本，问题才会变得具体。
