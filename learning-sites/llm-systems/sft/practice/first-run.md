---
title: 第一次可验证的 TRL SFT
description: 用小模型、内存数据集、token 审计和过拟合门禁完成一条可复现训练闭环
lesson:
  track: SFT
  stage: 02 · 实验
  time: 80–120 分钟
  level: 会运行 Python
  question: 怎样证明一次训练不是“能跑”，而是数据、loss、更新和生成都真的连通？
---

# 第一次可验证的 TRL SFT

第一次 run 的目标不是得到可上线模型，而是证明四件事：**样本被正确 token 化、目标 labels 正确、参数确实更新、模型能记住一小组监督。**因此使用小模型、内存构造的数据和短训练，不引入数据下载脚本、packing、LoRA 或多卡变量。

## 运行条件

推荐 Python 虚拟环境与一张可用 GPU；135M 级模型也可用 CPU 验证逻辑，但会慢。安装与你选定版本兼容的 PyTorch 后：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按 https://pytorch.org/get-started/locally/ 安装与你的 CUDA/CPU 匹配的 torch。
python -m pip install torch

# 课程阅读快照；直接 URL 会被 pip freeze 记录为 commit。
python -m pip install \
  "transformers @ git+https://github.com/huggingface/transformers.git@e52d0fd6fa9eb874f7c2da048198276b04c919b9" \
  "trl @ git+https://github.com/huggingface/trl.git@f3adc504b93d634666c5628e7bdaa99ec8861028" \
  "peft @ git+https://github.com/huggingface/peft.git@cea8213158c8b682acc0839405c2062d57fdf867" \
  "accelerate @ git+https://github.com/huggingface/accelerate.git@665444ceb62211f2b410d0d0fdb4bc013c5effdf" \
  "datasets @ git+https://github.com/huggingface/datasets.git@41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2"

python - <<'PY'
import accelerate, datasets, peft, torch, transformers, trl
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
print("cuda", torch.cuda.is_available())
PY
python -m pip freeze > runs-environment.txt
```

这些仓库快照不保证适配任意 PyTorch/CUDA；安装或 import 失败时保留完整 solver 输出，并按各仓库当时的 Python/PyTorch 要求建环境。网络受限环境应提前把 model/tokenizer 按固定 revision 放到本地并启用离线模式。不要在生产机直接运行未审查的 `trust_remote_code=True`。

## 最小训练脚本

下面刻意使用标准 `prompt`/`completion`，先绕开 chat template。模型 id 可替换为本地的 100M–600M causal LM；示例模型需要首次下载。

```python
import json
import os
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M")
REVISION = os.environ.get("REVISION", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
OUT = "runs/tiny-sft"
set_seed(42)

pairs = [
    ("代号 A17 对应什么颜色？", " 青绿色。"),
    ("代号 B04 对应什么动物？", " 雪豹。"),
    ("代号 C29 对应什么城市？", " 苏州。"),
    ("代号 D63 对应什么乐器？", " 大提琴。"),
]
rows = [
    {"prompt": f"问题：{q}\n答案：", "completion": a}
    for _ in range(16)
    for q, a in pairs
]
train_ds = Dataset.from_list(rows)

bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16 = torch.cuda.is_available() and not bf16
dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

args = SFTConfig(
    output_dir=OUT,
    model_init_kwargs={"revision": REVISION, "dtype": dtype},
    max_steps=120,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    max_length=128,
    completion_only_loss=True,
    packing=False,
    gradient_checkpointing=False,
    bf16=bf16,
    fp16=fp16,
    seed=42,
)

trainer = SFTTrainer(
    model=MODEL,
    args=args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

# 在第一步前审计 Trainer 实际构建的数据，不猜测 mask。
example = trainer.train_dataset[0]
tokenizer = trainer.processing_class
tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
audit = [
    {"pos": i, "token": tok, "id": tid, "label": lab}
    for i, (tok, tid, lab) in enumerate(
        zip(tokens, example["input_ids"], example["labels"], strict=True)
    )
]
print(json.dumps(audit, ensure_ascii=False, indent=2))
assert any(x["label"] != -100 for x in audit)
assert all(
    x["label"] == -100
    for x in audit[: next(i for i, x in enumerate(audit) if x["label"] != -100)]
)

# 选一个小的可训练参数完整保存，避免只看 loss 猜测 optimizer 是否生效。
probe_name, probe = next(
    (name, p) for name, p in trainer.model.named_parameters()
    if p.requires_grad and 0 < p.numel() <= 16384
)
before = probe.detach().float().cpu().clone()
result = trainer.train()
print(result.metrics)
print("global_step", trainer.state.global_step)
assert trainer.state.global_step == args.max_steps
after = dict(trainer.model.named_parameters())[probe_name].detach().float().cpu()
delta = (after - before).norm().item()
print("parameter_probe", probe_name, "delta_norm", delta)
assert delta > 0, "optimizer completed but probe parameter did not change"
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)

# 验收必须从磁盘 checkpoint 重载，不能复用 trainer.model 掩盖漏存文件。
del trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
reload_tokenizer = AutoTokenizer.from_pretrained(OUT)
reload_model = AutoModelForCausalLM.from_pretrained(OUT).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reload_model.to(device)


def normalize(text):
    return re.sub(r"\s+", "", text).strip("。.!！").casefold()


passed = 0
for q, expected in pairs:
    prompt = f"问题：{q}\n答案："
    inputs = reload_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = reload_model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=reload_tokenizer.eos_token_id,
            pad_token_id=reload_tokenizer.pad_token_id
            or reload_tokenizer.eos_token_id,
        )
    completion = reload_tokenizer.decode(
        ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    exact = normalize(completion) == normalize(expected)
    passed += int(exact)
    print(json.dumps({
        "greedy_question": q,
        "prediction": completion,
        "expected": expected,
        "exact": exact,
    }, ensure_ascii=False))

print("greedy_passed", passed, "of", len(pairs))
assert passed >= 3, "tiny-overfit gate failed: fewer than 3/4 mappings reproduced"
print("checkpoint_reload", type(reload_model).__name__, type(reload_tokenizer).__name__)
```

命令：

```bash
mkdir -p runs
set -o pipefail
python train_tiny.py 2>&1 | tee runs/tiny-sft.log
test "${PIPESTATUS[0]}" -eq 0
```

代码中的高学习率和重复数据只用于过拟合测试，不是生产配方。

## 第一批前必须看什么

把 audit 输出还原成人能读的表：

| 检查 | 通过条件 |
| --- | --- |
| prompt boundary | prompt 对应 labels 全为 `-100` |
| completion | 答案 token labels 为自身 token id |
| EOS | completion 末尾存在 EOS，且 label 有效 |
| 长度 | 未意外截断；`len(input_ids)==len(labels)` |
| 空监督 | 每条至少一个非 `-100` target |
| 特殊 token | 无重复 BOS/EOS 或乱码 token |

若这一步不通过，不运行 120 steps。loss 可能在错误 labels 上照样下降。

## 训练时的四项证据

```mermaid
flowchart LR
    A[token audit passes] --> B[initial loss finite]
    B --> C[loss falls on tiny set]
    C --> D[at least one trainable parameter changed]
    D --> E[greedy generation reproduces mappings]
```

### 1. Loss 有限且下降

NaN/Inf 先查 dtype、数据和 LR。小数据重复训练应明显下降；完全不动时先查有效 labels、trainable params 与 optimizer step，而不是先增加 epochs。

### 2. Global step 确实增加

`global_step` 是 optimizer update 数，不是 dataloader micro-batch 数。当前配置没有 accumulation，因此二者接近；改变 accumulation 后要重新计算。

### 3. 参数确实更新

脚本已保存一个小的 trainable parameter 并断言 `delta_norm>0`。全参训练应有大量参数 `requires_grad=True`；LoRA 则只允许 adapter/显式 modules 更新。若模型所有可训练参数都大于 16384 elements，改为按模型结构选择一个 norm/bias 参数，不能删掉这项验证。

### 4. Greedy 生成

完整脚本在保存后删除 `trainer`，从 `runs/tiny-sft` 重载模型和 tokenizer，再对四条训练映射做 greedy 生成。至少 3/4 normalized exact match 才退出 0；因此“只能由内存中的旧模型生成”不会假通过。这四条都是训练数据，目的是检查连通性，不是泛化指标。

## 预期输出与验收

不同硬件与依赖会让速度、绝对 loss 不同，不应硬编码单一数值。以下不变量必须全部出现：

```text
token audit: prompt labels 全 -100；completion/EOS 至少一个有效 label
train_runtime/train_loss: finite
parameter_probe ... delta_norm <正数>
global_step: 120（脚本会断言，未达到即失败）
greedy_passed: 至少 3 of 4，且每条 JSON 输出来自重载 checkpoint
checkpoint: config + model weights + tokenizer files 可重新加载
```

主脚本已经完成严格重载。还可在另一个 Python 进程做只读 smoke，验证目录不依赖训练进程状态：

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("runs/tiny-sft")
t = AutoTokenizer.from_pretrained("runs/tiny-sft")
print(type(m).__name__, type(t).__name__, m.config.name_or_path)
PY
```

预期退出码 0。若 loss 下降但参数 delta 断言失败，先查 optimizer param groups/overflow；若参数更新但不能生成映射，查训练步数、labels 和 generation protocol；若只能在原进程生成，查 checkpoint 实际保存内容。

该调用链对应固定源码：trainer 数据准备 [`1374–1624`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L1374)，collator [`462–507`](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L462)，训练 step [`1880–1951`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L1880)，optimizer update [`1766–1787`](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L1766)。

## 再加对话格式

基础 run 通过后，把相同事实改成：

```python
{"messages": [
    {"role": "user", "content": "代号 A17 对应什么颜色？"},
    {"role": "assistant", "content": "青绿色。"},
]}
```

并显式使用带 chat template 的 instruct tokenizer；如果启用 `assistant_only_loss=True`，先验证 generation markers 能产出 assistant mask。推理也必须：

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

不要同时保留自己手工拼的 `<|user|>` 文本和自动模板。

## 基线记录

```text
model id + revision:
tokenizer/template hash:
TRL/Transformers/PyTorch/CUDA:
dataset rows + hash:
input/nonmasked token distribution:
global batch + tokens/update:
first/final loss:
steps/sec + supervised tokens/sec:
peak GPU memory:
four greedy outputs:
checkpoint contents:
```

`nvidia-smi` 的瞬时数值不能替代 `torch.cuda.max_memory_allocated()`/reserved 与 profiler；先至少记录 peak allocated/reserved。

## 常见失败

| 现象 | 首查 |
| --- | --- |
| `No columns in dataset` | schema 是否为 `text/messages` 或 `prompt/completion` |
| loss=0/NaN | labels 是否全 `-100`、精度/LR |
| loss 降但生成续写 prompt | completion mask/template boundary |
| 不生成 EOS | EOS 是否出现在有效 labels |
| checkpoint 无完整权重 | 是否其实用了 PEFT adapter |
| OOM | sequence/batch、dtype、activation，不先上多卡 |

## 通关标准

你应能在一台机器上复现：token audit 通过、tiny set loss 下降、参数改变、greedy 记住四条映射。任何一项失败，都能沿 dataset → labels → forward → backward → optimizer 找到证据。

下一课比较[LoRA 与 QLoRA](./lora-qlora)。
