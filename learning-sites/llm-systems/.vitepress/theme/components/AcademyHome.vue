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
  }
]
</script>

<template>
  <main class="academy-home">
    <section class="academy-hero" aria-labelledby="academy-title">
      <div>
        <p class="academy-kicker"><span>SYSTEMS COURSE</span> 固定源码 · 实验驱动</p>
        <h1 id="academy-title">不要背参数，<br><em>追踪系统。</em></h1>
        <p class="academy-lead">四条面向 LLM 工程师的中文学习路线。每个概念先用可计算的直觉解释，再落到张量、进程、通信、源码与可复现实验。</p>
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
      <p class="academy-eyebrow">FOUR LEARNING TRACKS</p>
      <div class="academy-heading">
        <h2 id="tracks-title">四个入口，一套系统视角</h2>
        <p>推理、训练和分布式不是孤岛：SFT 产出的权重进入 serving，引擎的吞吐又受并行、显存与通信边界约束。</p>
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
        <h2 id="loop-title">把四门课接成一条链</h2>
        <p>数据经过 SFT 改变权重；FSDP 或 Megatron 让更新跨 GPU 发生；权重进入 vLLM 或 SGLang；线上请求和指标再暴露数据、模型与系统的新问题。</p>
      </div>
      <div class="loop-track" role="img" aria-label="数据经过 SFT 和分布式训练生成权重，再由推理引擎提供服务并产生指标反馈">
        <div><span>DATA</span><strong>样本与模板</strong></div><i></i>
        <div><span>TRAIN</span><strong>SFT 更新</strong></div><i></i>
        <div><span>SCALE</span><strong>并行通信</strong></div><i></i>
        <div><span>SERVE</span><strong>推理调度</strong></div><i></i>
        <div><span>OBSERVE</span><strong>指标反馈</strong></div>
      </div>
    </section>
  </main>
</template>
