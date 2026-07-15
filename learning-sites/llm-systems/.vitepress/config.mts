import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const commits = {
  vllm: '61141ed265bfef41a0ca19e992567ea980919b96',
  sglang: 'c879f3da5ceaaef3cb197c4e59ce683d420ce96c',
  trl: 'f3adc504b93d634666c5628e7bdaa99ec8861028',
  transformers: 'e52d0fd6fa9eb874f7c2da048198276b04c919b9',
  megatron: '82e9dc69c9e6f8c27681f2cb6856a188187edf6b',
  torchtitan: 'fec3e196a4ceb87bfc87fb4f1a36a538d7e98ee4',
  deepspeed: '53a2ac44fb664bea838df3981ba4366b91643070'
}

const vllmSidebar = [
  {
    text: '00 · 先找到主线',
    items: [
      { text: 'vLLM 学习入口', link: '/vllm/' },
      { text: '学习地图与版本边界', link: '/vllm/guide/learning-path' },
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
      { text: '基准测试与调参', link: '/vllm/practice/benchmark' }
    ]
  },
  {
    text: '03 · 沿请求读源码',
    items: [
      { text: 'V1 多进程架构', link: '/vllm/internals/architecture' },
      { text: '一条请求的生命周期', link: '/vllm/internals/request-lifecycle' },
      { text: 'Scheduler 与 KV Cache', link: '/vllm/internals/scheduler-kv' },
      { text: 'Worker、Runner 与模型执行', link: '/vllm/internals/model-execution' },
      { text: 'TP、PP、DP 与多节点', link: '/vllm/internals/distributed' }
    ]
  },
  {
    text: '04 · 进入生产与优化',
    items: [
      { text: 'Prefix、Chunk 与 Speculative', link: '/vllm/advanced/features' },
      { text: '生产诊断与容量规划', link: '/vllm/advanced/production' },
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
      { text: '基准、指标与调参', link: '/sglang/practice/benchmark' }
    ]
  },
  {
    text: '03 · 沿请求读源码',
    items: [
      { text: '进程与通信架构', link: '/sglang/internals/architecture' },
      { text: '一条请求的生命周期', link: '/sglang/internals/request-lifecycle' },
      { text: 'Scheduler 与 ScheduleBatch', link: '/sglang/internals/scheduler' },
      { text: 'RadixCache 与内存池', link: '/sglang/internals/cache-pools' },
      { text: 'ModelRunner 与执行后端', link: '/sglang/internals/model-execution' }
    ]
  },
  {
    text: '04 · 扩展与生产',
    items: [
      { text: '并行、PD 与 HiCache', link: '/sglang/advanced/distributed' },
      { text: '结构化输出与 RL 接入', link: '/sglang/advanced/features' },
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
      { text: '学习地图与工具边界', link: '/sft/guide/learning-path' }
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
      { text: 'Labels 到 Loss 与更新', link: '/sft/internals/loss-update' }
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
      { text: 'TorchTitan 源码主线', link: '/distributed/internals/torchtitan' },
      { text: 'Megatron 源码主线', link: '/distributed/internals/megatron-flow' },
      { text: '分布式 Checkpoint', link: '/distributed/practice/checkpointing' },
      { text: 'Hang、OOM 与性能排障', link: '/distributed/practice/debugging' },
      { text: '源码、论文与术语', link: '/distributed/appendix/references' }
    ]
  }
]

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'LLM 系统学习实验室',
    titleTemplate: ':title · LLM 系统学习实验室',
    description: '从心智模型、最小实验到源码与生产诊断的 vLLM、SGLang、SFT 和分布式训练中文课程',
    base: process.env.BASE_PATH || '/',
    outDir: process.env.OUT_DIR || '.vitepress/dist',
    cleanUrls: true,
    lastUpdated: true,
    head: [
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
        { text: '分布式训练', link: '/distributed/' },
        {
          text: '固定源码',
          items: [
            { text: `vLLM ${commits.vllm.slice(0, 7)}`, link: `https://github.com/vllm-project/vllm/tree/${commits.vllm}` },
            { text: `SGLang ${commits.sglang.slice(0, 7)}`, link: `https://github.com/sgl-project/sglang/tree/${commits.sglang}` },
            { text: `TRL ${commits.trl.slice(0, 7)}`, link: `https://github.com/huggingface/trl/tree/${commits.trl}` },
            { text: `Transformers ${commits.transformers.slice(0, 7)}`, link: `https://github.com/huggingface/transformers/tree/${commits.transformers}` },
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
