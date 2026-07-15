---
title: 'veRL 强化学习与源码学习实验室'
description: '从零理解强化学习，完成第一个 veRL 实验，再沿 V1 源码走到自定义算法与性能优化。'
pubDate: '2026-07-14'
tags: ['veRL', '强化学习', '源码阅读']
---

我把 veRL 学习内容重新整理成了一条面向初学者的六阶段路线：先补齐 LLM、概率与强化学习直觉，再理解 Policy Gradient、PPO、GRPO，随后完成一个可验证的训练 step，最后进入 veRL V1 源码、自定义算法与性能优化。

> [进入 veRL 学习实验室 →](https://cass998.github.io/losh_has_blog/verl-learning/)

课程包含六周学习计划、常见疑问、手算练习、检查点与实验产物清单。框架行为绑定 veRL 源码提交 `e5687fce`，并用架构图和时序图串起 prompt、rollout、reward、advantage、update 与权重同步的完整闭环。

如果你已经能跑通训练，建议直接进入两章系统设计主线：

- [Single Controller、HybridFlow 与模块化后端](https://cass998.github.io/losh_has_blog/verl-learning/internals/architecture)
- [Ray 基础、角色分工、代码位置与启动条件](https://cass998.github.io/losh_has_blog/verl-learning/internals/workers)
