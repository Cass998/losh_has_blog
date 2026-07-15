---
title: SFT 源码、官方文档、论文与术语
description: 固定 TRL/Transformers 入口、PEFT 与训练文档、核心论文、术语表和版本核验清单
lesson:
  track: SFT
  stage: 04 · 索引
  time: 按需查阅
  level: 全阶段
  question: 当教程、当前文档和训练行为不一致时，应怎样回到可复现证据？
---

# SFT 源码、官方文档、论文与术语

本课程主线绑定 TRL [`f3adc50`](https://github.com/huggingface/trl/tree/f3adc504b93d634666c5628e7bdaa99ec8861028) 与 Transformers [`e52d0fd`](https://github.com/huggingface/transformers/tree/e52d0fd6fa9eb874f7c2da048198276b04c919b9)。为核对 adapter、数据容器与分布式调用，本课程还读取 PEFT [`cea8213`](https://github.com/huggingface/peft/tree/cea8213158c8b682acc0839405c2062d57fdf867)、Datasets [`41adfd0`](https://github.com/huggingface/datasets/tree/41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2) 和 Accelerate [`665444c`](https://github.com/huggingface/accelerate/tree/665444ceb62211f2b410d0d0fdb4bc013c5effdf)，并在 DeepSpeed [`53a2ac4`](https://github.com/deepspeedai/DeepSpeed/tree/53a2ac44fb664bea838df3981ba4366b91643070) 上核对 ZeRO 配置。训练行为还依赖 PyTorch、bitsandbytes、kernel、模型 revision 与 remote code；固定源码主线不等于完整环境已固定。

## 源码快照清单

| 项目 | 固定提交 | 本课程实际追踪的责任 |
| --- | --- | --- |
| TRL | [`f3adc504`](https://github.com/huggingface/trl/tree/f3adc504b93d634666c5628e7bdaa99ec8861028) | SFTConfig、trainer 初始化、dataset/mask/packing、SFT metrics/chunked NLL |
| Transformers | [`e52d0fd6`](https://github.com/huggingface/transformers/tree/e52d0fd6fa9eb874f7c2da048198276b04c919b9) | chat template、Trainer loop、causal LM loss、quantization config |
| PEFT | [`cea82131`](https://github.com/huggingface/peft/tree/cea8213158c8b682acc0839405c2062d57fdf867) | LoRA 注入/forward、k-bit preparation、adapter save/merge |
| Datasets | [`41adfd0f`](https://github.com/huggingface/datasets/tree/41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2) | `Dataset.from_list`、map/filter/cache 与 JSON/Arrow 数据容器 |
| Accelerate | [`665444ce`](https://github.com/huggingface/accelerate/tree/665444ceb62211f2b410d0d0fdb4bc013c5effdf) | prepare、accumulate/no-sync、backward、clip、state dict |
| DeepSpeed | [`53a2ac44`](https://github.com/deepspeedai/DeepSpeed/tree/53a2ac44fb664bea838df3981ba4366b91643070) | ZeRO stage、offload 与 engine 边界 |

这些提交是“阅读快照”，不是宣称五个仓库在任意安装方式下都兼容。实践时必须保存 `pip freeze`，并以实际安装源码重复定位。

## 证据优先级

1. 实际 run 的完整环境、最终 config、数据/template/model revisions、batch 与日志；
2. 实际安装版本的源码与测试；
3. 同 commit 官方文档；
4. 当前 stable docs/release notes；
5. 原始论文；
6. 第三方教程。

论文解释算法，不保证当前参数名；current docs 描述新版本，不保证旧 checkpoint 的行为；一个能跑的教程也可能用全序列 loss 而你以为是 assistant-only。

## 固定 TRL 源码地图

| 主题 | 入口 |
| --- | --- |
| `SFTConfig` 与默认值 | [`sft_config.py`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_config.py#L23) |
| `SFTTrainer` 构造 | [`sft_trainer.py`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L790) |
| Text collator | [`DataCollatorForLanguageModeling`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L394) |
| VLM collator | [`DataCollatorForVisionLanguageModeling`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L542) |
| Dataset preparation | [`_prepare_dataset`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L1374) |
| Packing | [`pack_dataset`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/data_utils.py#L843) |
| Chunked CE | [`_chunked_cross_entropy_loss`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L117) |
| Loss/metrics | [`SFTTrainer.compute_loss`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L1699) |
| Shared base trainer | [`base_trainer.py`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/base_trainer.py#L65) |
| Chat helpers/templates | [`chat_template_utils.py`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/chat_template_utils.py)、[`chat_templates/`](https://github.com/huggingface/trl/tree/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/chat_templates) |

## 固定 Transformers 源码地图

| 主题 | 入口 |
| --- | --- |
| Trainer | [`trainer.py`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L257) |
| Dataloader | [`get_train_dataloader`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L875) |
| Optimizer creation | [`create_optimizer`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L1156) |
| Training loop | [`_inner_training_loop`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L1444) |
| Training step | [`training_step`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L1880) |
| Loss | [`compute_loss`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L1953) |
| Save | [`save_model`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L3780) |
| Chat template API | [`apply_chat_template`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/tokenization_utils_base.py#L3002) |
| Generic LM collator | [`data_collator.py`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/data/data_collator.py#L619) |

## 固定 PEFT / Accelerate 源码地图

| 主题 | 固定入口 | 读源码时要回答 |
| --- | --- | --- |
| `get_peft_model` | [`mapping_func.py#L30`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/mapping_func.py#L30) | wrapper 是否原地改 model；base revision 怎样写入 config |
| `LoraConfig` | [`config.py#L373`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/tuners/lora/config.py#L373) | target 的匹配规则、rank/alpha、`modules_to_save` |
| LoRA forward | [`layer.py#L1030`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/tuners/lora/layer.py#L1030) | `base(x)+B(A(dropout(x)))*scale` 在何时执行 |
| k-bit preparation | [`other.py#L151`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/utils/other.py#L151) | base freeze、dtype、input grad、gradient checkpointing |
| adapter save | [`peft_model.py#L212`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/peft_model.py#L212) | 保存的是 adapter 还是完整 base |
| merge/unload | [`tuners_utils.py#L696`](https://github.com/huggingface/peft/blob/cea8213158c8b682acc0839405c2062d57fdf867/src/peft/tuners/tuners_utils.py#L696) | 返回值必须接住；何时能合并 |
| `Accelerator.prepare` | [`accelerator.py#L1414`](https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/src/accelerate/accelerator.py#L1414) | model/optimizer/dataloader 怎样按 backend 包装 |
| `Accelerator.backward` | [`accelerator.py#L2818`](https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/src/accelerate/accelerator.py#L2818) | 普通、DeepSpeed、scaler 分支怎样选 |
| distributed state dict | [`accelerator.py#L4002`](https://github.com/huggingface/accelerate/blob/665444ceb62211f2b410d0d0fdb4bc013c5effdf/src/accelerate/accelerator.py#L4002) | FSDP/DeepSpeed 保存何时 gather |

## 本地复核这些提交

```bash
git clone https://github.com/huggingface/trl.git
git -C trl checkout f3adc504b93d634666c5628e7bdaa99ec8861028
git clone https://github.com/huggingface/transformers.git
git -C transformers checkout e52d0fd6fa9eb874f7c2da048198276b04c919b9
git clone https://github.com/huggingface/peft.git
git -C peft checkout cea8213158c8b682acc0839405c2062d57fdf867
git clone https://github.com/huggingface/accelerate.git
git -C accelerate checkout 665444ceb62211f2b410d0d0fdb4bc013c5effdf
git clone https://github.com/huggingface/datasets.git
git -C datasets checkout 41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2

git -C trl status --short
git -C transformers status --short
git -C peft status --short
git -C accelerate status --short
git -C datasets status --short
```

五次 `status --short` 都应为空。若不能 checkout，先确认 commit 是否完整拉取；不要用当前 `main` 替代后仍沿用本站行号。

## 同提交官方文档

### TRL

- [SFT Trainer](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/docs/source/sft_trainer.md)
- [Dataset Formats](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/docs/source/dataset_formats.md)
- [Chat Templates](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/docs/source/chat_templates.md)
- [PEFT Integration](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/docs/source/peft_integration.md)
- [Reducing Memory Usage](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/docs/source/reducing_memory_usage.md)

### Transformers

- [Trainer](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/docs/source/en/trainer.md)
- [Chat Templates](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/docs/source/en/chat_templating.md)
- [Writing Chat Templates](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/docs/source/en/chat_templating_writing.md)
- [FSDP](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/docs/source/en/fsdp.md)
- [Bitsandbytes Quantization](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/docs/source/en/quantization/bitsandbytes.md)

## 核心论文

### Instruction/SFT 背景

- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — instruction tuning 与任务泛化。
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — SFT、reward modeling、RLHF pipeline；本课程只覆盖其中 SFT 主线。
- [Self-Instruct](https://arxiv.org/abs/2212.10560) — 合成 instruction 数据生成与过滤。

### Parameter-efficient fine-tuning

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — 冻结 base、低秩更新与实验。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — NF4、double quantization、paged optimizers 与 4-bit base 上的 LoRA。

论文中的硬件、模型、rank 与质量结论不是当前模型的默认配方。先复现实验假设，再映射到你的 workload。

## 术语对照

| 术语 | 本课程定义 |
| --- | --- |
| SFT | 用有监督 token targets 微调语言模型；不等于完整 RLHF |
| teacher forcing | 训练时每个位置的上下文使用数据中的真实历史 token |
| causal shift | 位置 $t$ 的 logits 对齐位置 $t+1$ 的 label |
| label mask | 用 `-100` 等 ignore index 排除某些 target 的 loss |
| attention mask | 控制 token 可见性/有效 padding；不等于 label mask |
| completion-only | prompt-completion 数据只对 completion 区间计算 loss |
| assistant-only | 对话数据只对 template 标记的 assistant spans 计算 loss |
| chat template | 把 structured messages 编码成模型 token 协议的 Jinja 模板 |
| generation marker | 模板中标出 assistant target span、用于生成 mask 的控制标记 |
| EOT / EOS | 对话轮结束/序列结束；可能同 id，也可能不同 |
| packing | 多样本装入固定长度 row，并保持样本 attention/position/loss 边界 |
| padding-free | flatten batch 以消除 padding，依赖正确 sequence metadata/backend |
| global batch | 每次 optimizer update 跨 data ranks 与 accumulation 的样本数 |
| supervised tokens/update | 一次 update 中所有非 `-100` shifted targets，更适合变长 SFT |
| LoRA | 冻结 base、训练低秩 adapter 更新 |
| QLoRA | 量化存储冻结 base，并训练通常为较高精度的 LoRA adapter |
| DDP | 每 data rank 复制模型并同步 gradients，主要扩吞吐 |
| FSDP/ZeRO | 分片参数/梯度/optimizer 状态以降低每卡驻留 |
| checkpoint | 可能是完整、adapter 或 distributed shards；必须说明类型 |

## 版本核验清单

```bash
python - <<'PY'
import inspect
import accelerate, datasets, peft, torch, transformers, trl
from trl import SFTConfig, SFTTrainer

for mod in (torch, transformers, trl, accelerate, datasets, peft):
    print(mod.__name__, getattr(mod, "__version__", "unknown"))
print(inspect.getsourcefile(SFTTrainer))
print(SFTConfig(output_dir="/tmp/check"))
PY
```

再保存 GPU/driver/CUDA、model/tokenizer revision、template hash、data manifest、resolved config、world size/topology、launcher command 和 git dirty state。

## 从课程毕业的五个作品

1. 一份 golden token audit：raw → rendered → ids → labels；
2. 一次 tiny-overfit 与 one-step gradient/parameter delta 证明；
3. 一份全参/LoRA/QLoRA 的质量、显存、吞吐与 checkpoint 对照；
4. 一条 `_prepare_dataset → collator → compute_loss → backward → step` 源码 trace；
5. 一份单卡到 DDP/FSDP 的 scaling report 与 OOM/NaN/hang runbook。

完成这些作品，才算从“会跑 SFT 命令”进阶到能解释、验证和维护训练系统。
