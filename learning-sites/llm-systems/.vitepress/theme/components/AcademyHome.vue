<script setup lang="ts">
import { withBase } from 'vitepress'

const tracks = [
  {
    no: '01',
    code: 'SERVE',
    title: 'vLLM',
    question: '请求如何经过调度、KV Cache 和 GPU 执行，持续产出 token？',
    detail: 'PagedAttention · continuous batching · V1 engine · distributed serving',
    link: '/vllm/'
  },
  {
    no: '02',
    code: 'RUNTIME',
    title: 'SGLang',
    question: '共享前缀如何进入 Radix Tree，并被调度器转成真实吞吐收益？',
    detail: 'RadixAttention · Scheduler · memory pools · PD disaggregation',
    link: '/sglang/'
  },
  {
    no: '03',
    code: 'TRAIN',
    title: 'SFT',
    question: '一条对话怎样变成 input_ids、labels、loss 和一次可靠更新？',
    detail: 'chat template · label mask · packing · TRL · LoRA',
    link: '/sft/'
  },
  {
    no: '04',
    code: 'SCALE',
    title: '分布式训练',
    question: '模型、状态、序列和专家应该切在哪个维度，通信代价是什么？',
    detail: 'FSDP2 · Megatron · TP/PP/CP/EP · checkpoint',
    link: '/distributed/'
  },
  {
    no: '05',
    code: 'ALIGN',
    title: 'veRL',
    question: 'SFT 权重如何经 vLLM/SGLang 采样、Ray 编排和 PPO/GRPO 更新，形成可验证的强化学习闭环？',
    detail: 'Single Controller · HybridFlow · Ray · rollout backend · weight sync',
    link: '/verl/'
  }
]
</script>

<template>
  <main class="academy-home">
    <section class="academy-hero" aria-labelledby="academy-title">
      <div>
        <p class="academy-kicker"><span>SYSTEMS COURSE</span> 固定源码 · 实验驱动</p>
        <h1 id="academy-title">不要背参数，<br><em>追踪系统。</em></h1>
        <p class="academy-lead">五条面向 LLM 工程师的中文学习路线。每个概念先用可计算的直觉解释，再落到张量、进程、通信、源码与可复现实验。</p>
        <a class="academy-button" :href="withBase('/vllm/')">
          从 vLLM 开始
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5" /></svg>
        </a>
      </div>
      <aside class="academy-board" aria-label="共同学习方法">
        <div class="academy-board__top"><span>TRACE, MEASURE, EXPLAIN</span><i></i><i></i><i></i></div>
        <ol>
          <li><span>01</span><p><strong>心智模型</strong>先能预测系统下一步会发生什么。</p></li>
          <li><span>02</span><p><strong>最小实验</strong>留下命令、输入、日志和指标。</p></li>
          <li><span>03</span><p><strong>源码主线</strong>沿一个请求或样本走完整调用链。</p></li>
          <li><span>04</span><p><strong>生产判断</strong>用瓶颈证据决定参数和并行策略。</p></li>
        </ol>
      </aside>
    </section>

    <section class="academy-section" aria-labelledby="tracks-title">
      <p class="academy-eyebrow">FIVE LEARNING TRACKS</p>
      <div class="academy-heading">
        <h2 id="tracks-title">五个入口，一条端到端闭环</h2>
        <p>SFT 产出初始策略，veRL 借助 vLLM/SGLang 收集轨迹并更新权重；训练和 rollout 的规模上限又由并行、显存与通信共同决定。</p>
      </div>
      <ol class="track-grid">
        <li v-for="track in tracks" :key="track.no">
          <a :href="withBase(track.link)">
            <span class="track-no">{{ track.no }}</span>
            <span class="track-code">{{ track.code }}</span>
            <h3>{{ track.title }}</h3>
            <p>{{ track.question }}</p>
            <small>{{ track.detail }}</small>
            <span class="track-arrow" aria-hidden="true">→</span>
          </a>
        </li>
      </ol>
    </section>

    <section class="academy-section academy-loop" aria-labelledby="loop-title">
      <div>
        <p class="academy-eyebrow">ONE END-TO-END LOOP</p>
        <h2 id="loop-title">把五门课接成一条链</h2>
        <p>数据先完成 SFT 冷启动；veRL 用推理后端生成轨迹、计算奖励并更新策略；FSDP 或 Megatron 承担跨 GPU 训练；新权重进入服务后，线上指标再反馈给下一轮数据与训练。</p>
      </div>
      <div class="loop-track" role="img" aria-label="数据经过 SFT 冷启动，由 veRL 使用推理后端采样并强化学习更新，分布式训练负责扩展计算，新权重进入服务后产生指标反馈">
        <div><span>DATA</span><strong>样本与模板</strong></div><i></i>
        <div><span>SFT</span><strong>监督冷启动</strong></div><i></i>
        <div><span>ALIGN</span><strong>veRL 采样与更新</strong></div><i></i>
        <div><span>SCALE</span><strong>并行通信</strong></div><i></i>
        <div><span>SERVE</span><strong>推理调度</strong></div><i></i>
        <div><span>OBSERVE</span><strong>指标反馈</strong></div>
      </div>
    </section>
  </main>
</template>
