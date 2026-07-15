---
title: 读懂 veRL 够用的数学工具
description: 用小数字和张量例子理解概率、对数、期望、标准化、梯度、重要性比率与 masked mean
lesson:
  stage: 01 · 地基
  order: 04
  time: 45–60 分钟
  level: 零基础补给
  question: PPO/GRPO 公式中哪些数学必须会，怎样把它们落到张量？
---

# 读懂 veRL 够用的数学工具

你不需要先证明每个定理，但必须能预测一项运算会把训练信号变大、变小还是反向。这一课只使用一个三选一策略和一组四条回答，把后面最常见的数学一次补齐。

## 1. 概率分布：一张总和为 1 的选择表

策略给三个 token 的概率为 `[0.7, 0.2, 0.1]`。它不是说下一次一定选第一个，而是说重复采样时约 70% 选第一个。

softmax 把任意 logits \(z_i\) 变成概率：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

给所有 logits 同时加 100，概率不变；放大 logits 之间的差异，分布会更尖。这是理解 temperature 的基础。

## 2. 对数：把乘法变加法

若一段三-token 回答的条件概率依次为 `0.5, 0.4, 0.2`，序列概率是：

$$
0.5\times0.4\times0.2=0.04.
$$

log-prob 则是：

$$
\log0.5+\log0.4+\log0.2=\log0.04.
$$

概率在 `(0,1]`，所以自然对数不大于 0。概率越接近 1，log-prob 越接近 0；极小概率对应很大的负数。

这解释了源码为什么用：

```python
ratio = torch.exp(log_prob - old_log_prob)
```

因为 \(\exp(\log p-\log q)=p/q\)，既稳定又避免先还原两个很小的序列概率。

## 3. 期望：按发生概率计算长期平均

动作 A 有 80% 得 1 分、20% 得 0 分，期望回报是 `0.8×1 + 0.2×0 = 0.8`。强化学习最大化的是策略分布下的期望回报，不保证每一次采样都更好。

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)].
$$

这里 `~` 表示轨迹由当前策略采样。改变策略，就同时改变会看到哪些轨迹；这是 RL 数据分布不断变化的原因。

## 4. 均值、方差与标准化：比较“相对好多少”

同一道题四个回答 reward 为 `[1, 1, 0, 0]`：

- 均值 \(\mu=0.5\)；
- 减均值后为 `[0.5, 0.5, -0.5, -0.5]`；
- 再除标准差，会改变尺度但不改变正负方向。

GRPO 的组内标准化让不同难度 prompt 的信号更可比，但组标准差也可能引入难度/尺度偏差。Dr.GRPO 形式只减均值，不除标准差。这里没有永远正确的按钮；要知道选择改变了什么统计量。

::: tip 先看正负，再看尺度
advantage 的正负决定提高还是降低已采样动作概率；绝对值决定这条样本影响多大。标准化通常不改变组内排序，却会改变不同组之间的相对权重。
:::

## 5. 导数与梯度：哪边能让目标上升

一维函数的导数告诉你参数稍微增加时函数怎样变化；多参数模型的梯度是每个方向的导数组成的向量。

训练框架通常最小化 loss。若我们想最大化 reward objective，就把它取负：

$$
\text{loss}=-\text{objective}.
$$

因此看到源码里 `pg_losses = -advantages * ratio` 不要误解为“优势越大越惩罚”。优化器最小化负目标，效果正好是最大化原目标。

## 6. log-prob 的梯度：不对离散 token 求导

采样出的 token ID 是离散整数，不能沿 token ID 求导。可求导的是模型参数对该 token log-prob 的影响：

$$
\nabla_\theta\log\pi_\theta(a\mid s).
$$

乘上 advantage 后：

- \(A>0\)：梯度更新倾向提高这次动作概率；
- \(A<0\)：倾向降低这次动作概率；
- \(A=0\)：该项不给策略梯度信号。

这座桥会在[策略梯度这座桥](/verl/algorithms/policy-gradient)中完整展开。

## 7. 重要性比率：新旧策略对同一动作的态度变化

同一个已采样 token，旧策略概率为 0.2，新策略概率为 0.3：

$$
r=\frac{0.3}{0.2}=1.5.
$$

`ratio > 1` 说明当前策略更偏爱这个动作，`ratio < 1` 说明更不偏爱。PPO clipping 不是裁剪 reward，也不是裁剪梯度数值本身，而是限制 ratio 超出区间后继续获得优化收益。

用 log-prob 计算：

$$
r=\exp(\log\pi_\theta-\log\pi_{old}).
$$

## 8. Shape 与 broadcasting：公式正确，维度也可能错

常见张量：

```text
responses       [B, R]
response_mask   [B, R]
sequence_reward [B]
advantages      [B, R]
```

GRPO 先得到每条回答一个标量 `[B]`，再 `unsqueeze(-1)` 成 `[B,1]`，与 `[B,R]` 的 mask 相乘，broadcast 成逐 token 优势。

```python
advantages = sequence_advantage.unsqueeze(-1) * response_mask
```

这不代表算法知道哪个 token 真正导致成功；只是把序列级信号广播给所有有效 response token。

## 9. Masked mean：分母是算法的一部分

假设两个回答的有效长度分别为 2 和 4，补齐后：

```text
loss = [[1, 1, 0, 0],
        [1, 1, 1, 1]]
mask = [[1, 1, 0, 0],
        [1, 1, 1, 1]]
```

token mean 为 \(6/6=1\)。若先对每条序列取平均、再对 batch 取平均，也是 1；但当两条序列 loss 不同时，两种聚合给长回答的权重不同。

通用 masked mean：

$$
\frac{\sum_{b,t}x_{b,t}m_{b,t}}
{\max(1,\sum_{b,t}m_{b,t})}.
$$

分母、跨 rank 归约顺序和零有效 token 防护都属于算法语义，不只是“代码细节”。

## 四道手算题

### 题 1：概率比

旧概率 0.5，新概率 0.4，ratio 是多少？它说明什么？

<details>
<summary>查看答案</summary>

ratio=`0.4/0.5=0.8`。当前策略给这个已采样动作的概率降低了 20%。这不直接说明更新好坏，还需看 advantage 正负。
</details>

### 题 2：更新方向

ratio=1.2，advantage=-0.5。忽略 clipping，最小化 `-ratio×advantage` 会倾向提高还是降低动作概率？

<details>
<summary>查看答案</summary>

降低。负 advantage 表示动作比基线差，策略梯度希望降低它的概率。不要只看 loss 数字正负，要看对 log-prob 的梯度方向。
</details>

### 题 3：组内信号

一组 reward `[1,1,1,1]` 减均值后是什么？这批 GRPO 样本还能提供什么策略梯度？

<details>
<summary>查看答案</summary>

全为 0，几乎没有组内相对策略信号。这就是全对/全错组常被动态采样关注的原因。
</details>

### 题 4：mask

`advantages=[[2,2,9]]`，`response_mask=[[1,1,0]]`。有效 token mean 是多少？

<details>
<summary>查看答案</summary>

2。最后的 9 位于无效 padding；若得到 13/3，说明聚合漏了 mask。
</details>

## 通关检查

你应能不查资料解释：为什么序列概率用 log 相加、为什么 ratio 用 log-prob 差的 exp、advantage 的正负与尺度各控制什么、`[B]` 如何变成 `[B,R]`、为什么 loss aggregation 会影响长度偏好。

下一步：[强化学习的完整闭环](./rl)。
