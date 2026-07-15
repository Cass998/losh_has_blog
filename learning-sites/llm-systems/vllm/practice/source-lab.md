---
title: vLLM 源码跟踪实验：从静态证据到运行时 trace
description: 固定源码、跑最小 CPU 单测、启动可观测服务、跟踪 request id、profile GPU 路径并形成可复查实验报告
lesson:
  track: vLLM
  stage: 03 · 源码实践
  time: 120–240 分钟
  level: 完成首次服务与源码主线
  question: 怎样证明自己读到的函数确实是这次请求运行的路径？
---

# vLLM 源码跟踪实验：从静态证据到运行时 trace

源码阅读给出“可能怎么走”，运行证据回答“这次配置实际怎么走”。本实验把二者闭环：先固定 commit 和配置，再用仓库已有测试验证纯控制逻辑，最后在本地服务上观察请求、调度、GPU 与输出。

::: danger 只在隔离开发环境做
Profiler、详细 iteration log 和开发端点会显著降低性能，甚至暴露危险控制接口。服务只绑定 `127.0.0.1`，不要在生产或公网开启。
:::

## 实验交付物

完成后应有：

```text
trace-lab/
├── version.txt
├── startup-command.txt
├── startup.log
├── request.json
├── response.sse
├── metrics-before.txt
├── metrics-during.txt
├── metrics-after.txt
├── cpu-tests.txt
├── profile/            # 有 GPU 且执行 profiler 时
└── report.md
```

不要求必须使用这个目录名；要求每项证据能对应同一次实验。

## 1. 获取固定源码和同版本官方文档

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm.git
cd vllm
git fetch origin 61141ed265bfef41a0ca19e992567ea980919b96
git switch --detach 61141ed265bfef41a0ca19e992567ea980919b96

mkdir -p trace-lab/profile
git show -s --format='commit=%H%ncommit_date=%ci%nsubject=%s' HEAD \
  | tee trace-lab/version.txt
test "$(git rev-parse HEAD)" = 61141ed265bfef41a0ca19e992567ea980919b96
```

验收：`test` 退出码为 0；commit、date 与 subject 已真实写入 `version.txt`。后续命令都在这个 `vllm/` 仓库根目录执行，因此从空目录开始也已经具备 `trace-lab/` 和 `profile/`。

本地仓库的 `docs/` 与源码同 commit，是判断这一版文档语义的第一选择：

```bash
sed -n '1,220p' docs/design/arch_overview.md
sed -n '1,220p' docs/design/prefix_caching.md
sed -n '1,220p' docs/serving/parallelism_scaling.md
sed -n '1,220p' docs/contributing/profiling.md
```

当前推荐安装/参数再看[官方 stable 文档](https://docs.vllm.ai/en/stable/)。必须在报告中标注“固定提交事实”还是“当前文档建议”。

### 常见失败

| 失败 | 原因 | 修正 |
| --- | --- | --- |
| `pathspec ... did not match` | shallow clone 没有目标 commit | `git fetch origin <SHA>` 后 detached switch |
| GitHub 行号和本地不同 | HEAD 不一致 | 重新核对完整 SHA，不接受只看前 7 位 |
| stable 文档参数本地不存在 | 文档比 commit 新 | 用本地 `--help` 和同 commit `docs/`，记录版本差异 |

## 2. 用最短搜索重建调用链

不要先打开 IDE 全局 call hierarchy。依次运行：

```bash
rg -n 'class ServeSubcommand|def run_server' \
  vllm/entrypoints/cli/serve.py vllm/entrypoints/openai/api_server.py

rg -n 'def create_chat_completion|def _create_chat_completion' \
  vllm/entrypoints/openai/chat_completion

rg -n 'class AsyncLLM|def add_request|def generate|def output_handler' \
  vllm/v1/engine/async_llm.py

rg -n 'def run_busy_loop|def step\(|def preprocess_add_request' \
  vllm/v1/engine/core.py

rg -n 'def schedule|def update_from_output|def finish_requests' \
  vllm/v1/core/sched/scheduler.py

rg -n 'def get_computed_blocks|def allocate_slots|def free' \
  vllm/v1/core/kv_cache_manager.py

rg -n 'def execute_model|def sample_tokens' \
  vllm/v1/executor vllm/v1/worker

rg -n 'class LlamaForCausalLM|class LlamaModel|class LlamaAttention' \
  vllm/model_executor/models/llama.py

rg -n 'class Sampler|def forward|def sample' vllm/v1/sample/sampler.py
```

把结果填进一张表：

| symbol | call condition | input → output | state mutation | next consumer |
| --- | --- | --- | --- | --- |
| `OpenAIServingChat._create_chat_completion` | chat request 渲染成功 | request → engine input/generator | request metadata | AsyncLLM |
| `Scheduler.schedule` | 有 waiting/running | Requests → SchedulerOutput | queues/block refs/budget | Executor |
| ... | ... | ... | ... | ... |

验收：至少覆盖[完整调用链](../internals/full-code-path)的 12 站；不能只填文件与类名。

## 3. 先跑不需要 GPU 的控制面单测

安装源码开发依赖的方式会随官方仓库变化，先读固定提交的 `CONTRIBUTING.md` 和安装文档。已有可用开发环境时运行：

```bash
set -o pipefail
pytest -q tests/v1/core/test_scheduler.py \
  -k 'test_schedule_order or test_preempt_during_execution' \
  2>&1 | tee trace-lab/cpu-tests.txt
test "${PIPESTATUS[0]}" -eq 0
```

这两个测试直接验证：

- chunked prefill 开/关怎样改变 waiting admission；
- KV block 满时，第二个请求怎样变成 `PREEMPTED`。

再跑最小请求状态测试：

```bash
set -o pipefail
pytest -q tests/v1/core/test_scheduler.py \
  -k 'test_add_requests or test_finish_request' \
  2>&1 | tee -a trace-lab/cpu-tests.txt
test "${PIPESTATUS[0]}" -eq 0
```

预期：测试被标记为 `cpu_test`，无需模型权重。若 import 阶段失败，先按官方开发安装补齐同 commit 依赖；不要因为测试没运行就写“通过”。

### 从测试反推源码

对 `test_preempt_during_execution` 做三栏记录：

```text
test setup            source branch              asserted state
num_blocks=11         allocate_slots() None       request.status=PREEMPTED
```

验收：能解释为什么有效 block 是 10（一个 null block 被保留），以及抢占发生在 Scheduler 而非 CUDA allocator。

## 4. 启动一台“可观测但不公开”的服务

先用当前安装版确认参数：

```bash
vllm --version
vllm serve --help | rg 'enable-logging-iteration|profiler|enforce-eager'
```

固定提交支持 iteration detail 配置；启动参数的精确 CLI 拼写以 `--help` 为准。一个本地示例：

```bash
export MODEL=Qwen/Qwen3-0.6B
export MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p trace-lab/profile

cat > trace-lab/startup-command.txt <<EOF
MODEL=$MODEL
MODEL_REVISION=$MODEL_REVISION
vllm serve "$MODEL" --revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION" --host 127.0.0.1 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.80 --generation-config vllm
EOF

set -o pipefail
vllm serve "$MODEL" \
  --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --generation-config vllm \
  2>&1 | tee trace-lab/startup.log
```

在另一个终端记录：

```bash
rg 'V2 Model Runner|attention backend|KV cache|CUDA graph|compile|world size' \
  trace-lab/startup.log

curl -s http://127.0.0.1:8000/metrics > trace-lab/metrics-before.txt
```

验收：报告中明确 Runner V1/V2、Executor backend、attention backend、TP/PP/DP、KV capacity 和 graph/compile 模式。启动日志缺某项时写“未观察到”，不要自行补齐。

## 5. 发送一条可追踪、有限长度的流式请求

在第二个终端回到同一个仓库根目录，先生成请求文件。这里故意让第一条请求生成足够多 token，以便指标轮询确实覆盖请求运行期；`ignore_eos` 只用于本次系统 trace，不是产品生成配置。

```bash
export MODEL=Qwen/Qwen3-0.6B
mkdir -p trace-lab/profile

cat > trace-lab/request.json <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "列出二十个要点，逐项解释 KV Cache 的生命周期。"}
  ],
  "temperature": 0,
  "max_tokens": 512,
  "ignore_eos": true,
  "stream": true
}
EOF

cat > trace-lab/long-request.json <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "连续编号写两百条分布式推理排障检查项。"}
  ],
  "temperature": 0,
  "max_tokens": 2048,
  "ignore_eos": true,
  "stream": true
}
EOF

python -m json.tool trace-lab/request.json >/dev/null
python -m json.tool trace-lab/long-request.json >/dev/null
```

发送并保存原始 SSE。请求必须在后台运行，否则下一条命令只能采到“结束后”的指标：

```bash
curl -fN http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: source-trace-001' \
  --data @trace-lab/request.json \
  > trace-lab/response.sse \
  2> trace-lab/response.stderr &
REQUEST_PID=$!

: > trace-lab/metrics-during.txt
for sample in $(seq 1 200); do
  {
    printf 'sample=%s time=%s\n' "$sample" "$(date --iso-8601=ns)"
    curl -fsS http://127.0.0.1:8000/metrics \
      | rg 'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|request_success)' \
      || true
  } >> trace-lab/metrics-during.txt
  kill -0 "$REQUEST_PID" 2>/dev/null || break
  sleep 0.05
done

wait "$REQUEST_PID"
REQUEST_STATUS=$?
curl -fsS http://127.0.0.1:8000/metrics > trace-lab/metrics-after.txt
test "$REQUEST_STATUS" -eq 0
rg -q '^data: ' trace-lab/response.sse
rg -q '\[DONE\]' trace-lab/response.sse
```

若你的服务版本不采用该 header 作为内部 request id，以响应/服务日志中的实际 id 为准。不要仅因自定义 header 存在就假设 Core 使用它。

`metrics-before.txt` 已在发送前采集，循环文件才是运行中证据。验证至少捕获过一个非零 running 样本；若这个断言失败，不得把“during”写成通过，应增大 `max_tokens` 或并发数后重跑：

```bash
rg 'vllm:num_requests_running.* [1-9][0-9]*(\.[0-9]+)?$' \
  trace-lab/metrics-during.txt
```

### 应观察什么

| 证据 | 预期 | 失败含义 |
| --- | --- | --- |
| SSE 首 chunk | role/first delta 到达 | API/renderer/Core/first sample 至少跑通 |
| 后续 chunks | 增量文本，不重复全量 | OutputProcessor/stream generator 正常 |
| finish reason | `stop` 或 `length` | 与 EOS/max_tokens 一致 |
| running/waiting | 请求期间变化、结束后回落 | 若不回落，检查流结束/abort/metrics 采样 |
| startup Runner log | 明确 V1/V2 | 决定该读哪个 ModelRunner 文件 |

返回 200 只覆盖 HTTP 层；上表五项共同通过才是本实验的服务验收。

## 6. 做一次取消实验，验证反向控制链

上一步已创建 `long-request.json`。另发一条长输出，在客户端收到若干 token 后由 `timeout` 中断，并分别保存取消前后指标：

```bash
curl -fsS http://127.0.0.1:8000/metrics > trace-lab/metrics-before-cancel.txt
timeout 2s curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data @trace-lab/long-request.json \
  > trace-lab/cancelled-response.sse || test "$?" -eq 124

for attempt in $(seq 1 100); do
  curl -fsS http://127.0.0.1:8000/metrics \
    > trace-lab/metrics-after-cancel.txt
  if rg -q 'vllm:num_requests_running.* 0(\.0+)?$' \
      trace-lab/metrics-after-cancel.txt \
    && rg -q 'vllm:num_requests_waiting.* 0(\.0+)?$' \
      trace-lab/metrics-after-cancel.txt; then
    break
  fi
  sleep 0.1
done
rg -q 'vllm:num_requests_running.* 0(\.0+)?$' \
  trace-lab/metrics-after-cancel.txt
rg -q 'vllm:num_requests_waiting.* 0(\.0+)?$' \
  trace-lab/metrics-after-cancel.txt
```

预期控制链：

```text
HTTP disconnect
→ AsyncLLM.generate CancelledError/GeneratorExit
→ AsyncLLM.abort
→ EngineCore ABORT
→ Scheduler.finish_requests
→ KV blocks free
```

源码证据：[`generate()` cancel handler](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/engine/async_llm.py#L588-L635) 与 [`Scheduler.finish_requests()`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/vllm/v1/core/sched/scheduler.py#L2075-L2169)。

验收：中断后 waiting/running/KV 回落且日志没有继续为该请求生成。`timeout` 返回非零是预期，不是服务失败。

## 7. 可选：只 profile 两个请求

固定提交官方 profiler 文档是 [`docs/contributing/profiling.md`](https://github.com/vllm-project/vllm/blob/61141ed265bfef41a0ca19e992567ea980919b96/docs/contributing/profiling.md)。PyTorch profiler 示例：

```bash
vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "./trace-lab/profile",
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true
  }'
```

另一个终端：

```bash
curl -X POST http://127.0.0.1:8000/start_profile

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data @trace-lab/request.json > /dev/null

curl -X POST http://127.0.0.1:8000/stop_profile
```

预期：停止 profiler 可能等待 trace flush；生成的 trace 可在 Perfetto 打开。只采少量请求，profile 数据不能拿来报告无 profiler 的真实性能。

至少标记这些区间：

```text
runner state/input prep
model forward
attention/KV update
linear/MLP
sampling
D2H/output packaging
```

### 安全与失败模式

- `/start_profile`、`/stop_profile` 只在启用 profiler 时存在；404 先查启动参数。
- profile 会显著拖慢服务，不把 TTFT/ITL 与无 profile baseline 比。
- trace 为空时，确认 start → requests → stop 顺序和目录权限。
- profiler/development endpoints 不能暴露公网；它们是管理能力，不是普通 inference API。

## 8. 可选：eager 与默认路径做因果对照

用完全相同的模型、请求和环境分别启动默认与 `--enforce-eager`：

| 项目 | 默认 | eager |
| --- | --- | --- |
| cold ready time |  |  |
| first request |  |  |
| warm TTFT/ITL |  |  |
| output token ids |  |  |
| peak memory |  |  |
| compile/graph log |  |  |

验收：语义输出在你的确定性容差内一致；性能差异只用于说明 compile/graph 的贡献。若 eager 修复 crash，结论是“问题缩到默认执行优化路径”，不是“永久关掉优化已根治”。

## 9. 报告模板

先生成交付文件，再逐项把占位内容替换成真实证据：

```bash
cat > trace-lab/report.md <<'EOF'
# vLLM source trace report

- Question:
- Fixed commit and package version:
- Hardware / driver / torch / CUDA:
- Model + revision:
- Complete startup command: see startup-command.txt
- Resolved Runner / Executor / attention / compile mode:

## Static source chain

symbol → condition → input/output → mutation → next

## Runtime observations

- startup:
- request/SSE:
- scheduler/KV metrics:
- cancel:
- profiler (optional):

## Conclusion

- Expected vs actual:
- Strongest explanation:
- Alternative explanations ruled out:
- Failure modes encountered:
- Acceptance checklist:
- Next smallest experiment:
EOF
test -s trace-lab/report.md
```

```text
Question:
Fixed commit and package version:
Hardware / driver / torch / CUDA:
Model + revision:
Complete startup command:
Resolved Runner / Executor / attention / compile mode:

Static source chain:
  symbol → condition → input/output → mutation → next

Runtime observations:
  startup:
  request/SSE:
  scheduler/KV metrics:
  cancel:
  profiler (optional):

Expected vs actual:
Strongest explanation:
Alternative explanations ruled out:
Failure modes encountered:
Acceptance checklist:
Next smallest experiment:
```

## 最终验收

- [ ] commit、模型 revision 和环境完整可复查；
- [ ] 至少两条 CPU Scheduler 测试真实运行并保存输出；
- [ ] 静态调用链每站有条件、输入输出、状态变化；
- [ ] 确定性非流式或流式请求完成并有 finish reason；
- [ ] 指标显示请求进入、运行、结束后的状态变化；
- [ ] 取消实验能证明 abort 与 KV 回收；
- [ ] 若做 profiler，只采隔离请求并不拿 trace 性能冒充 baseline；
- [ ] 所有失败都保留原始错误，没有把未运行写成通过。

上一课是[模型 forward 与采样源码](../internals/model-forward-sampling)。完成后进入[分布式与 Ray 实验](./distributed-lab)，把单进程证据扩展到 rank、actor 和 collective。
