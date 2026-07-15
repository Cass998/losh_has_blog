---
title: SFT 评估、过拟合与数据诊断
description: 用 tiny-overfit、teacher-forced 指标、真实生成、任务切片与错误分类判断训练是否有效
lesson:
  track: SFT
  stage: 02 · 实验
  time: 90–125 分钟
  level: 已完成一次训练
  question: loss 降低后，怎样证明模型学到了目标能力而不是记忆、泄漏或模板？
---

# SFT 评估、过拟合与数据诊断

没有单一指标能证明 SFT 成功。应建立一座证据梯：**管线正确 → 能过拟合小样本 → held-out loss 改善 → 真实生成完成任务 → 关键切片与回归不过度退化。**越靠后越接近产品目标，越靠前越适合定位错误。

## 五层评估

```mermaid
flowchart BT
    A[1 token/data invariants] --> B[2 tiny-set overfit]
    B --> C[3 teacher-forced eval]
    C --> D[4 deterministic generation metrics]
    D --> E[5 human / task / safety slices]
```

| 层 | 回答 | 不能单独回答 |
| --- | --- | --- |
| token invariants | labels/template/截断是否正确 | 模型质量 |
| tiny overfit | forward-backward-update 是否连通 | 泛化 |
| eval loss/perplexity | held-out target token 概率是否提高 | 自由生成长期行为 |
| task metric | exact match、执行、JSON 等任务结果 | 未覆盖的开放质量 |
| 人工/安全/回归 | 帮助性、事实、风格、风险与旧能力 | 全量统计稳定性 |

## Tiny-overfit 是单元测试

从 16–64 条无歧义样本构造固定集，关闭复杂增强，训练到模型能 greedy 复现目标。失败的常见原因：

- labels 全/大部被 mask；
- answer 被 `max_length` 截断；
- optimizer 没 step 或 trainable params=0；
- chat template 与生成 prompt 不一致；
- LR/precision 导致数值问题；
- checkpoint/load 路径错误。

tiny overfit 通过只证明管线有能力记忆，不证明数据量、正则或泛化配置合理。

## Teacher-forced 指标怎样读

### Eval loss

必须在相同 tokenizer/template、mask policy、max length 和 loss normalization 下比较。assistant-only 与 full-sequence loss 不能直接横比；有效 labels 不同。

### Perplexity

$$
PPL=\exp(\mathcal L_{token})
$$

它要求 loss 是可比较的平均 NLL。不同词表/tokenization、mask 或特殊 token 会改变单位。PPL 更低表示 target token 概率更高，不保证回答更事实、更安全或遵循指令。

### Mean token accuracy

当前 TRL 可记录非 masked token 的 top-1 accuracy。它直观但偏向容易/重复 token，也不反映概率校准和长生成错误传播。

固定实现不是黑盒：[`SFTTrainer.compute_loss` 1751–1796](https://github.com/huggingface/trl/blob/f3adc504b93d634666c5628e7bdaa99ec8861028/trl/trainer/sft_trainer.py#L1751) 对普通 NLL 使用 `outputs.logits[..., :-1, :]` 与 `labels[..., 1:]`，只在 `label!=-100` 位置累计 entropy/correct/total，再跨 ranks gather；chunked NLL 则读取 patched forward 返回的原始计数。`mean_token_accuracy` 因而是 shifted、masked target 的 top-1，而不是 input token 重建率。

Transformers 的整体 eval 入口在 [`evaluate` 2542](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L2542)、聚合循环在 [`evaluation_loop` 2641](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L2641)、单 batch 在 [`prediction_step` 2909](https://github.com/huggingface/transformers/blob/e52d0fd6fa9eb874f7c2da048198276b04c919b9/src/transformers/trainer.py#L2909)。要排查 eval OOM，先确认是否缓存 full logits，而不是先降低训练 batch。

### Gradient 与优化指标

`learning_rate`、`grad_norm`、有效 tokens/update、step time 与 NaN/overflow 是解释 loss 的上下文。只看平滑后的 loss 会掩盖 batch slice 异常。

## 真实生成要固定协议

建立 versioned prompt suite，保存渲染后 ids 与生成参数：

```python
GEN = dict(
    do_sample=False,
    max_new_tokens=128,
    repetition_penalty=1.0,
)
```

先用 greedy 做版本回归，避免 sampling noise；再用生产采样参数评估体验。对话模型必须使用与训练一致的 chat template 和 generation prompt。

每条记录：example id、data slice、model/base/adapter revision、rendered input hash、output、latency、自动分数、错误标签。不要只保存聚合分数。

### 可运行的 deterministic 短答案评估器

上一课的 `SmolLM2-135M` 固定 tokenizer **没有 chat template**，训练契约是普通 `prompt`/`completion`。因此本节的默认输入每行是 `id/prompt/expected/slice`，且 `prompt` 必须与训练时前缀完全一致。评估器也支持 `messages`，但只有该 checkpoint 的 tokenizer 能通过 `get_chat_template()` 解析出模板时才允许走对话分支；两种 schema 不能混在同一行。

先从空目录生成与 tiny-overfit 对应的 smoke 数据。它用于验证保存/加载/生成链路，不冒充 held-out 泛化集：

```bash
mkdir -p data reports
cat > data/eval.jsonl <<'EOF'
{"id":"tiny-a17","prompt":"问题：代号 A17 对应什么颜色？\n答案：","expected":"青绿色。","slice":"tiny-overfit"}
{"id":"tiny-b04","prompt":"问题：代号 B04 对应什么动物？\n答案：","expected":"雪豹。","slice":"tiny-overfit"}
{"id":"tiny-c29","prompt":"问题：代号 C29 对应什么城市？\n答案：","expected":"苏州。","slice":"tiny-overfit"}
{"id":"tiny-d63","prompt":"问题：代号 D63 对应什么乐器？\n答案：","expected":"大提琴。","slice":"tiny-overfit"}
EOF
python -m json.tool --json-lines data/eval.jsonl >/dev/null
```

```python
# eval_generate.py
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, eval_path, out_path = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def norm(text):
    return re.sub(r"\s+", "", text).strip("。.!！").casefold()

rows = [
    json.loads(line)
    for line in Path(eval_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not rows:
    raise ValueError(f"{eval_path}: empty evaluation set")

records = []
for row in rows:
    has_plain = "prompt" in row
    has_chat = "messages" in row
    if has_plain == has_chat:
        raise ValueError(f"{row.get('id')}: require exactly one of prompt/messages")
    if has_plain:
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"{row.get('id')}: prompt must be non-empty text")
        protocol = "plain-prompt-completion"
        prompt = tok(row["prompt"], return_tensors="pt").to(device)
    else:
        # get_chat_template() 会在 tokenizer 无模板或多模板无默认值时明确报错。
        selected_template = tok.get_chat_template()
        protocol = "chat-template"
        prompt = tok.apply_chat_template(
            row["messages"],
            chat_template=selected_template,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    with torch.no_grad():
        output = model.generate(
            **prompt, do_sample=False, max_new_tokens=16,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    new_ids = output[0, prompt["input_ids"].shape[1]:]
    prediction = tok.decode(new_ids, skip_special_tokens=True)
    records.append({
        "id": row["id"], "slice": row.get("slice", "default"),
        "protocol": protocol,
        "input_sha256": hashlib.sha256(prompt["input_ids"].cpu().numpy().tobytes()).hexdigest(),
        "prediction": prediction, "expected": row["expected"],
        "exact": norm(prediction) == norm(row["expected"]),
    })

output_file = Path(out_path)
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8"
)
print(json.dumps({"n": len(records), "exact_match": sum(x["exact"] for x in records) / len(records)}))
```

```bash
mkdir -p reports
python eval_generate.py runs/tiny-sft data/eval.jsonl reports/eval-output.jsonl
test -s reports/eval-output.jsonl
```

预期：退出码 0、输出 `n/exact_match`，并保存每条原始预测、实际协议与 input hash。空集会在生成前明确失败；plain checkpoint 不再错误调用 chat template。模型/adapter 加载失败、对话数据的 template 缺失或输出切片错误都算评估失败，不能以手工复制几条回答代替。真正评估泛化时另建不与训练重复、按来源分组切分的 JSONL，仍复用相同 plain prompt 契约。

## 自动指标按任务选

| 任务 | 主指标 | 必须防的“假通过” |
| --- | --- | --- |
| 分类/短答案 | normalized exact match、F1 | 格式/标点造成假错或答案泄漏 |
| 数学 | final answer + 独立求解/校验 | 只匹配字符串、错误推理碰巧同答案 |
| 代码 | sandboxed unit tests | 不可信代码逃逸、测试太弱 |
| JSON/工具 | schema + business constraints + tool simulation | 语法合法但调用无权限/语义错 |
| 摘要 | factuality/coverage + 人审 | n-gram 高但事实幻觉 |
| 对话 | rubric pairwise + blind review | position/verbosity bias |

LLM judge 可以扩展抽样，但要固定 judge version/prompt，做人工校准并报告不确定/分歧。它不能成为唯一 ground truth。

## 切片比总分更有诊断力

至少按以下维度分桶：来源、语言、任务、难度、prompt/output 长度、多轮数、工具/推理、有无截断、训练中频次、安全类别。

```text
overall +4% 可能由：
  easy FAQ      +12%
  code           +1%
  long context   -9%
  safety         -6%
```

总体改善不应掩盖关键业务 slice 回退。发布门禁要明确“必须不退化”的切片，而不是训练后临时挑指标。

## 用曲线区分三类问题

```mermaid
flowchart TD
    A[train/eval evidence] --> B{train loss 能否下降?}
    B -->|不能| C[管线/优化/容量/欠拟合]
    B -->|能| D{eval 同步改善?}
    D -->|先改善后恶化| E[过拟合/数据重复/训练过久]
    D -->|从不改善| F[分布错位/泄漏式训练目标/坏标签]
    D -->|改善| G{生成任务改善?}
    G -->|否| H[template/decoding/metric错位/exposure gap]
    G -->|是| I[检查回归与安全切片]
```

### 欠拟合

train loss 高且仍下降：增加有效 steps/capacity、检查 LR；但先确认标签不是高噪声。train loss 不动：检查管线优先于“模型太小”。

### 过拟合

train 持续降、eval 反弹：early stopping/最佳 checkpoint、去重、增加多样性、调 LR/正则/LoRA rank。不能只通过扩大 eval 集掩盖。

### 目标错位

eval NLL 降但生成没改善：检查 mask 是否监督了模板/思维噪声，推理模板是否一致，任务 metric 是否真正反映产品目标。

## 数据归因实验

当新增数据后质量变化，用受控 ablation：

1. 固定 base、seed、tokens/update、总监督 token budget；
2. baseline 数据 A；
3. A+B；
4. 只在关键 slice 上比较；
5. 若变化显著，再按 B 的来源/质量等级拆分。

不能让 A+B 因总训练 token 更多而自然占优，再把全部增益归功于 B 的质量。

## Checkpoint 选择与测试集纪律

- validation 用于选 step/超参；test 只在方案冻结后少量运行；
- 保存 best 与 last，区分“最后一步”和“最佳验证”；
- 多次试验盯同一 test 会把 test 变成 validation；
- 报告多 seed 或至少说明单 seed 方差未知；
- adapter/merged checkpoint 都做加载后的独立进程 smoke。

## 最小报告模板

```text
Hypothesis:
Model/base/adapter revisions:
Data manifest and split grouping:
Template/tokenizer/mask policy:
Global batch and supervised tokens/update:
Train/eval curves:
Task metrics by slice + confidence/seed:
Representative wins/failures:
Regression and safety gates:
Peak memory / supervised tokens per second:
Selected checkpoint and why:
Known limitations / rollback:
```

## 通关练习

现象：train loss 0.4、eval loss 0.6，均比 base 好；但服务端回答总以 `<|assistant|>` 开头两次。最强假设不是过拟合，而是训练/服务发生模板重复或数据 content 已含 role marker。比较离线与服务最终 `input_ids`，以及清洗前 content。

## 通关标准

你应能设计 tiny-overfit 单测、解释 PPL 的可比条件、建立 deterministic generation suite、按 slice 找到总体分数掩盖的回退，并用曲线区分优化失败、过拟合和目标错位。

下一阶段沿源码看[TRL / Transformers 架构](../internals/architecture)。
