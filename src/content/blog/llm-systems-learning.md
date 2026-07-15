---
title: 'LLM 系统学习实验室：vLLM、SGLang、SFT、veRL 与分布式训练'
description: '五条从心智模型、最小实验走到固定版本源码与生产排障的中文学习路线。'
pubDate: '2026-07-16'
tags: ['LLM', 'vLLM', 'SGLang', 'SFT', 'veRL', '分布式训练']
---

我把大模型系统学习整理成了五条可以独立进入、又能互相衔接的路线。每条路线都从“这个系统究竟在解决什么”开始，经过可验证的小实验，再沿固定提交的真实调用链进入源码，最后落到性能指标、故障证据和生产决策。

> [进入 LLM 系统学习实验室 →](https://cass998.github.io/losh_has_blog/llm-systems/)

你也可以直接进入对应课程：

- [vLLM：推理循环、PagedAttention、Scheduler、Worker、Ray 与生产容量](https://cass998.github.io/losh_has_blog/llm-systems/vllm/)
- [SGLang：Runtime、RadixAttention、Scheduler、缓存池、PD 解耦与 HiCache](https://cass998.github.io/losh_has_blog/llm-systems/sglang/)
- [SFT：数据、Chat Template、Label Mask、LoRA/QLoRA、评估与源码](https://cass998.github.io/losh_has_blog/llm-systems/sft/)
- [veRL：强化学习、Single Controller、HybridFlow、Ray、rollout/训练后端与权重同步](https://cass998.github.io/losh_has_blog/llm-systems/verl/)
- [分布式训练：DDP、ZeRO、FSDP2、Megatron TP/PP/CP/EP、Checkpoint 与排障](https://cass998.github.io/losh_has_blog/llm-systems/distributed/)

关键行为绑定 vLLM、SGLang、TRL、Transformers、veRL、Ray、PyTorch、Megatron-LM、TorchTitan 与 DeepSpeed 的明确源码提交；课程不把“支持某功能”当作结论，而要求继续回答它切什么 tensor、使用哪个 process group、何时通信、如何验收数值与性能。

veRL 已经并入统一路线：[先看它怎样连接 SFT、vLLM/SGLang、Ray 与分布式训练](https://cass998.github.io/losh_has_blog/llm-systems/verl/guide/llm-systems-integration)，再沿一次真实 PPO/GRPO step 深入源码。
