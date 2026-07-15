import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const commits = {
  vllm: '61141ed265bfef41a0ca19e992567ea980919b96',
  sglang: 'c879f3da5ceaaef3cb197c4e59ce683d420ce96c',
  trl: 'f3adc504b93d634666c5628e7bdaa99ec8861028',
  transformers: 'e52d0fd6fa9eb874f7c2da048198276b04c919b9',
  peft: 'cea8213158c8b682acc0839405c2062d57fdf867',
  accelerate: '665444ceb62211f2b410d0d0fdb4bc013c5effdf',
  datasets: '41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2',
  verl: 'e5687fce0516d31e1fdc4580499074a9bd94c751',
  ray: '98c9eb75e44207266cc35f0cd94e1ecd58e3f77b',
  pytorch: 'e11b512fef37205cc3b83872eabd92c3cdf05a28',
  megatron: '82e9dc69c9e6f8c27681f2cb6856a188187edf6b',
  torchtitan: 'fec3e196a4ceb87bfc87fb4f1a36a538d7e98ee4',
  deepspeed: '53a2ac44fb664bea838df3981ba4366b91643070'
}

const basePath = process.env.BASE_PATH || '/'

const vllmSidebar = [
  {
    text: '00 · 先找到主线',
    items: [
      { text: 'vLLM 学习入口', link: '/vllm/' },
      { text: '学习地图与版本边界', link: '/vllm/guide/learning-path' },
      { text: '30 日逐日源码计划', link: '/vllm/guide/study-plan' },
      { text: '源码地图', link: '/vllm/guide/source-map' }
    ]
  },
  {
    text: '01 · 推理系统地基',
    items: [
      { text: '一次生成到底做什么', link: '/vllm/fundamentals/inference-loop' },
      { text: 'KV Cache 与 PagedAttention', link: '/vllm/fundamentals/kv-cache' },
      { text: '批处理、延迟与吞吐', link: '/vllm/fundamentals/performance' }
    ]
  },
  {
    text: '02 · 跑通并测量',
    items: [
      { text: '启动第一台服务', link: '/vllm/practice/first-server' },
      { text: '基准测试与调参', link: '/vllm/practice/benchmark' },
      { text: '源码跟踪实验', link: '/vllm/practice/source-lab' }
    ]
  },
  {
    text: '03 · 沿请求读源码',
    items: [
      { text: 'V1 多进程架构', link: '/vllm/internals/architecture' },
      { text: '一条请求的生命周期', link: '/vllm/internals/request-lifecycle' },
      { text: 'HTTP 到 SSE 完整调用链', link: '/vllm/internals/full-code-path' },
      { text: 'Scheduler 与 KV Cache', link: '/vllm/internals/scheduler-kv' },
      { text: 'Worker、Runner 与模型执行', link: '/vllm/internals/model-execution' },
      { text: '模型 forward、attention 与采样', link: '/vllm/internals/model-forward-sampling' },
      { text: 'TP、PP、DP 与多节点', link: '/vllm/internals/distributed' }
    ]
  },
  {
    text: '04 · 进入生产与优化',
    items: [
      { text: 'Prefix、Chunk 与 Speculative', link: '/vllm/advanced/features' },
      { text: '生产诊断与容量规划', link: '/vllm/advanced/production' },
      { text: 'TP/PP/DP/Ray 分布式实验', link: '/vllm/practice/distributed-lab' },
      { text: '源码、论文与术语', link: '/vllm/appendix/references' }
    ]
  }
]

const sglangSidebar = [
  {
    text: '00 · 先找到主线',
    items: [
      { text: 'SGLang 学习入口', link: '/sglang/' },
      { text: '学习地图与版本边界', link: '/sglang/guide/learning-path' },
      { text: '六周逐周源码计划', link: '/sglang/guide/study-plan' },
      { text: '源码地图', link: '/sglang/guide/source-map' }
    ]
  },
  {
    text: '01 · Runtime 地基',
    items: [
      { text: '推理循环与进程边界', link: '/sglang/fundamentals/runtime' },
      { text: 'RadixAttention', link: '/sglang/fundamentals/radix-attention' },
      { text: '调度、缓存与性能', link: '/sglang/fundamentals/performance' }
    ]
  },
  {
    text: '02 · 跑通并测量',
    items: [
      { text: '启动第一台服务', link: '/sglang/practice/first-server' },
      { text: '基准、指标与调参', link: '/sglang/practice/benchmark' },
      { text: '八组递进实验手册', link: '/sglang/practice/lab-workbook' }
    ]
  },
  {
    text: '03 · 沿请求读源码',
    items: [
      { text: '进程与通信架构', link: '/sglang/internals/architecture' },
      { text: '一条请求的生命周期', link: '/sglang/internals/request-lifecycle' },
      { text: 'HTTP 到 Detokenizer 消息链', link: '/sglang/internals/message-flow' },
      { text: 'Scheduler 与 ScheduleBatch', link: '/sglang/internals/scheduler' },
      { text: 'Chunked Prefill 逐源码状态机', link: '/sglang/internals/chunked-prefill' },
      { text: 'RadixCache 与内存池', link: '/sglang/internals/cache-pools' },
      { text: 'ModelRunner 与执行后端', link: '/sglang/internals/model-execution' }
    ]
  },
  {
    text: '04 · 扩展与生产',
    items: [
      { text: '并行、PD 与 HiCache', link: '/sglang/advanced/distributed' },
      { text: 'PD 与 HiCache 完整数据路径', link: '/sglang/advanced/pd-hicache' },
      { text: '结构化输出与 RL 接入', link: '/sglang/advanced/features' },
      { text: 'RL 换权生命周期与 veRL', link: '/sglang/advanced/rl-lifecycle' },
      { text: '生产诊断与容量规划', link: '/sglang/advanced/production' },
      { text: '源码、论文与术语', link: '/sglang/appendix/references' }
    ]
  }
]

const sftSidebar = [
  {
    text: '00 · 定义训练目标',
    items: [
      { text: 'SFT 学习入口', link: '/sft/' },
      { text: '14 天学习计划与工具边界', link: '/sft/guide/learning-path' }
    ]
  },
  {
    text: '01 · 从文本到 Loss',
    items: [
      { text: 'Teacher Forcing 与交叉熵', link: '/sft/fundamentals/objective' },
      { text: '数据格式与质量', link: '/sft/fundamentals/data' },
      { text: 'Chat Template 与特殊 token', link: '/sft/fundamentals/chat-template' },
      { text: 'Label Mask、截断与 Packing', link: '/sft/fundamentals/masking-packing' }
    ]
  },
  {
    text: '02 · 完成可验证训练',
    items: [
      { text: '第一次 TRL SFT', link: '/sft/practice/first-run' },
      { text: 'LoRA 与 QLoRA', link: '/sft/practice/lora-qlora' },
      { text: '评估、过拟合与数据诊断', link: '/sft/practice/evaluation' }
    ]
  },
  {
    text: '03 · 沿样本读源码',
    items: [
      { text: 'TRL / Transformers 架构', link: '/sft/internals/architecture' },
      { text: 'Dataset 到 Batch', link: '/sft/internals/data-pipeline' },
      { text: 'Labels 到 Loss 与更新', link: '/sft/internals/loss-update' },
      { text: 'SFTTrainer 到 optimizer 逐源码主线', link: '/sft/internals/source-walkthrough' }
    ]
  },
  {
    text: '04 · 扩规模与排错',
    items: [
      { text: '显存、吞吐与分布式', link: '/sft/systems/scaling' },
      { text: '失败模式与排障', link: '/sft/systems/debugging' },
      { text: '源码、论文与术语', link: '/sft/appendix/references' }
    ]
  }
]

const distributedSidebar = [
  {
    text: '00 · 先选对切分维度',
    items: [
      { text: '分布式训练入口', link: '/distributed/' },
      { text: '学习地图与版本边界', link: '/distributed/guide/learning-path' },
      { text: '42 天逐日源码计划', link: '/distributed/guide/source-study-plan' },
      { text: '框架选择地图', link: '/distributed/guide/decision-map' }
    ]
  },
  {
    text: '01 · 共同地基',
    items: [
      { text: '进程、拓扑与集合通信', link: '/distributed/fundamentals/collectives' },
      { text: '训练显存账本', link: '/distributed/fundamentals/memory' }
    ]
  },
  {
    text: '02 · 数据并行与切状态',
    items: [
      { text: 'DDP 与 DeepSpeed ZeRO', link: '/distributed/data-parallel/ddp-zero' },
      { text: 'FSDP2、DTensor 与 DeviceMesh', link: '/distributed/data-parallel/fsdp2' }
    ]
  },
  {
    text: '03 · 切模型与序列',
    items: [
      { text: 'Megatron Core 总体设计', link: '/distributed/model-parallel/megatron' },
      { text: 'Tensor / Sequence Parallel', link: '/distributed/model-parallel/tensor-sequence' },
      { text: 'Pipeline Parallel', link: '/distributed/model-parallel/pipeline' },
      { text: 'Context / Expert Parallel', link: '/distributed/model-parallel/context-expert' },
      { text: '3D/4D/5D 并行组合', link: '/distributed/model-parallel/multidimensional' }
    ]
  },
  {
    text: '04 · 实验、源码与生产',
    items: [
      { text: 'FSDP2 与 Megatron 实验', link: '/distributed/practice/first-runs' },
      { text: '可下载源码实验手册', link: '/distributed/practice/source-labs' },
      { text: 'PyTorch DDP Reducer 主线', link: '/distributed/internals/pytorch-ddp-runtime' },
      { text: 'FSDP2 参数状态机', link: '/distributed/internals/fsdp2-source' },
      { text: 'TorchTitan 源码主线', link: '/distributed/internals/torchtitan' },
      { text: 'Megatron 源码主线', link: '/distributed/internals/megatron-flow' },
      { text: 'DeepSpeed ZeRO 源码主线', link: '/distributed/internals/deepspeed-zero-flow' },
      { text: '分布式 Checkpoint', link: '/distributed/practice/checkpointing' },
      { text: 'Hang、OOM 与性能排障', link: '/distributed/practice/debugging' },
      { text: '源码、论文与术语', link: '/distributed/appendix/references' }
    ]
  }
]

const verlSidebar = [
  {
    text: '00 · 先找到起点',
    items: [
      { text: 'veRL 学习入口', link: '/verl/' },
      { text: '六周学习计划', link: '/verl/guide/learning-plan' },
      { text: '21 天 Ray 源码计划', link: '/verl/guide/ray-source-study-plan' },
      { text: '小白常见问题', link: '/verl/guide/common-questions' },
      { text: '连接 SFT、推理与分布式', link: '/verl/guide/llm-systems-integration' },
      { text: 'V1 版本边界', link: '/verl/guide/version-boundary' }
    ]
  },
  {
    text: '01 · 建立算法地基',
    items: [
      { text: 'LLM 如何成为策略', link: '/verl/fundamentals/llm' },
      { text: '够用的数学工具', link: '/verl/fundamentals/math-toolkit' },
      { text: '强化学习完整闭环', link: '/verl/fundamentals/rl' },
      { text: '策略梯度', link: '/verl/algorithms/policy-gradient' },
      { text: 'PPO 与 GAE', link: '/verl/algorithms/ppo' },
      { text: 'GRPO、Dr.GRPO 与 DAPO', link: '/verl/algorithms/grpo-family' }
    ]
  },
  {
    text: '02 · 跑通并验证',
    items: [
      { text: '第一次可验证实验', link: '/verl/practice/first-run' },
      { text: '读懂 Hydra 配置', link: '/verl/practice/configuration' },
      { text: '可靠奖励函数', link: '/verl/practice/reward-function' },
      { text: 'Ray 观测实验', link: '/verl/practice/ray-observation-lab' },
      { text: '故障定位', link: '/verl/practice/debugging' },
      { text: '性能采集', link: '/verl/practice/profiling' }
    ]
  },
  {
    text: '03 · 逐源码理解系统',
    items: [
      { text: '源码地图', link: '/verl/guide/source-map' },
      { text: 'Single Controller 与 HybridFlow', link: '/verl/internals/architecture' },
      { text: '入口与初始化', link: '/verl/internals/entry-and-init' },
      { text: 'V1 逐源码主线', link: '/verl/internals/v1-source-walkthrough' },
      { text: 'Rollout 到 Update', link: '/verl/internals/rollout-to-update' },
      { text: '数据结构与 TransferQueue', link: '/verl/internals/data-structures' },
      { text: 'Ray、角色与资源编排', link: '/verl/internals/workers' },
      { text: 'Ray 运行时逐源码主线', link: '/verl/internals/ray-source-runtime' },
      { text: '训练/推理后端契约', link: '/verl/internals/backend-contracts' }
    ]
  },
  {
    text: '04 · 扩展与优化',
    items: [
      { text: '扩展点地图', link: '/verl/customization/extension-map' },
      { text: '实现自定义算法', link: '/verl/customization/custom-algorithm' },
      { text: '优化方法论', link: '/verl/customization/optimization-playbook' },
      { text: '源码、论文与术语', link: '/verl/appendix/references' }
    ]
  }
]

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'LLM 系统学习实验室',
    titleTemplate: ':title · LLM 系统学习实验室',
    description: '从心智模型、最小实验到源码与生产诊断的 vLLM、SGLang、SFT、veRL 和分布式训练中文课程',
    base: basePath,
    outDir: process.env.OUT_DIR || '.vitepress/dist',
    cleanUrls: true,
    lastUpdated: true,
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: `${basePath}lab-mark.svg` }],
      ['meta', { name: 'theme-color', content: '#173e46' }],
      ['meta', { name: 'referrer', content: 'strict-origin-when-cross-origin' }],
      ['meta', { name: 'color-scheme', content: 'light dark' }]
    ],
    markdown: {
      math: true,
      image: { lazyLoading: true }
    },
    mermaid: {
      theme: 'neutral',
      flowchart: { htmlLabels: true, curve: 'basis' }
    },
    themeConfig: {
      logo: '/lab-mark.svg',
      siteTitle: 'LLM 系统实验室',
      nav: [
        { text: '总览', link: '/' },
        { text: 'vLLM', link: '/vllm/' },
        { text: 'SGLang', link: '/sglang/' },
        { text: 'SFT', link: '/sft/' },
        { text: 'veRL', link: '/verl/' },
        { text: '分布式训练', link: '/distributed/' },
        {
          text: '固定源码',
          items: [
            { text: `vLLM ${commits.vllm.slice(0, 7)}`, link: `https://github.com/vllm-project/vllm/tree/${commits.vllm}` },
            { text: `SGLang ${commits.sglang.slice(0, 7)}`, link: `https://github.com/sgl-project/sglang/tree/${commits.sglang}` },
            { text: `TRL ${commits.trl.slice(0, 7)}`, link: `https://github.com/huggingface/trl/tree/${commits.trl}` },
            { text: `Transformers ${commits.transformers.slice(0, 7)}`, link: `https://github.com/huggingface/transformers/tree/${commits.transformers}` },
            { text: `PEFT ${commits.peft.slice(0, 7)}`, link: `https://github.com/huggingface/peft/tree/${commits.peft}` },
            { text: `Accelerate ${commits.accelerate.slice(0, 7)}`, link: `https://github.com/huggingface/accelerate/tree/${commits.accelerate}` },
            { text: `Datasets ${commits.datasets.slice(0, 7)}`, link: `https://github.com/huggingface/datasets/tree/${commits.datasets}` },
            { text: `veRL ${commits.verl.slice(0, 7)}`, link: `https://github.com/verl-project/verl/tree/${commits.verl}` },
            { text: `Ray ${commits.ray.slice(0, 7)}`, link: `https://github.com/ray-project/ray/tree/${commits.ray}` },
            { text: `PyTorch ${commits.pytorch.slice(0, 7)}`, link: `https://github.com/pytorch/pytorch/tree/${commits.pytorch}` },
            { text: `Megatron ${commits.megatron.slice(0, 7)}`, link: `https://github.com/NVIDIA/Megatron-LM/tree/${commits.megatron}` },
            { text: `TorchTitan ${commits.torchtitan.slice(0, 7)}`, link: `https://github.com/pytorch/torchtitan/tree/${commits.torchtitan}` },
            { text: `DeepSpeed ${commits.deepspeed.slice(0, 7)}`, link: `https://github.com/deepspeedai/DeepSpeed/tree/${commits.deepspeed}` }
          ]
        }
      ],
      sidebar: {
        '/vllm/': vllmSidebar,
        '/sglang/': sglangSidebar,
        '/sft/': sftSidebar,
        '/verl/': verlSidebar,
        '/distributed/': distributedSidebar
      },
      search: {
        provider: 'local',
        options: {
          translations: {
            button: { buttonText: '搜索', buttonAriaLabel: '搜索全部课程' },
            modal: {
              noResultsText: '没有找到相关内容',
              resetButtonTitle: '清除查询',
              footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
            }
          }
        }
      },
      outline: { level: [2, 3], label: '这一课讲什么' },
      docFooter: { prev: '上一节', next: '下一节' },
      lastUpdated: { text: '最近更新' },
      socialLinks: [{ icon: 'github', link: 'https://github.com/Cass998/losh_has_blog' }],
      footer: {
        message: '先建立可计算的心智模型，再以固定版本源码和实验校准。',
        copyright: '原创中文学习材料；各框架源码版权归其项目贡献者。'
      }
    }
  })
)
