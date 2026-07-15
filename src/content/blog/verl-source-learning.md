---
title: 'veRL 强化学习与源码学习实验室'
description: '从零理解强化学习，完成第一个 veRL 实验，再沿 V1 源码走到自定义算法与性能优化。'
pubDate: '2026-07-14'
tags: ['veRL', '强化学习', '源码阅读']
---

我把 veRL 学习内容重新整理成了一条面向初学者的六阶段路线：先补齐 LLM、概率与强化学习直觉，再理解 Policy Gradient、PPO、GRPO，随后完成一个可验证的训练 step，最后进入 veRL V1 源码、自定义算法与性能优化。

> [进入统一 LLM 系统实验室中的 veRL 路线 →](https://cass998.github.io/losh_has_blog/llm-systems/verl/)

课程包含六周学习计划、常见疑问、手算练习、检查点与实验产物清单。框架行为绑定 veRL 源码提交 `e5687fce`，并用架构图和时序图串起 prompt、rollout、reward、advantage、update 与权重同步的完整闭环。现在它也与同站的 SFT、vLLM、SGLang 和分布式训练课程互相链接，不再是一个孤立站点。

如果你已经能跑通训练，建议直接进入两章系统设计主线：

- [veRL 如何连接整个 LLM 系统](https://cass998.github.io/losh_has_blog/llm-systems/verl/guide/llm-systems-integration)
- [Single Controller、HybridFlow 与模块化后端](https://cass998.github.io/losh_has_blog/llm-systems/verl/internals/architecture)
- [Ray 基础、角色分工、代码位置与启动条件](https://cass998.github.io/losh_has_blog/llm-systems/verl/internals/workers)
- [Ray 固定提交：从 ray.init 追到 GCS、raylet、CoreWorker 与 veRL Worker](https://cass998.github.io/losh_has_blog/llm-systems/verl/internals/ray-source-runtime)
- [V1：从 Hydra 入口逐行走到一次参数更新](https://cass998.github.io/losh_has_blog/llm-systems/verl/internals/v1-source-walkthrough)

原来的 `/verl-learning/` 地址会继续保留，避免旧笔记中的链接失效；新增和持续校订的内容以统一站点为准。
