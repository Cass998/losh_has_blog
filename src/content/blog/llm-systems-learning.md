---
title: 'LLM 系统学习实验室：vLLM、SGLang、SFT 与分布式训练'
description: '四条从心智模型、最小实验走到固定版本源码与生产排障的中文学习路线。'
pubDate: '2026-07-16'
tags: ['LLM', 'vLLM', 'SGLang', 'SFT', '分布式训练']
---

我把大模型系统学习整理成了四条可以独立进入、又能互相衔接的路线。每条路线都从“这个系统究竟在解决什么”开始，经过可验证的小实验，再沿固定提交的真实调用链进入源码，最后落到性能指标、故障证据和生产决策。

> [进入 LLM 系统学习实验室 →](https://cass998.github.io/losh_has_blog/llm-systems/)

你也可以直接进入对应课程：

- [vLLM：推理循环、PagedAttention、Scheduler、Worker、Ray 与生产容量](https://cass998.github.io/losh_has_blog/llm-systems/vllm/)
- [SGLang：Runtime、RadixAttention、Scheduler、缓存池、PD 解耦与 HiCache](https://cass998.github.io/losh_has_blog/llm-systems/sglang/)
- [SFT：数据、Chat Template、Label Mask、LoRA/QLoRA、评估与源码](https://cass998.github.io/losh_has_blog/llm-systems/sft/)
- [分布式训练：DDP、ZeRO、FSDP2、Megatron TP/PP/CP/EP、Checkpoint 与排障](https://cass998.github.io/losh_has_blog/llm-systems/distributed/)

整站目前包含 66 节课程和 132 张可渲染图。关键行为绑定 vLLM、SGLang、TRL、Transformers、Megatron-LM、TorchTitan 与 DeepSpeed 的明确源码提交；课程不把“支持某功能”当作结论，而要求继续回答它切什么 tensor、使用哪个 process group、何时通信、如何验收数值与性能。

如果你正在学习 veRL，可以先看原来的 [veRL 强化学习与源码实验室](https://cass998.github.io/losh_has_blog/verl-learning/)，再用这里的 vLLM/SGLang 与分布式训练课程补齐 rollout engine 和训练 backend。
