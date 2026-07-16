# LLM 系统学习实验室

一条共同基础与五条互相连接的中文课程：先从 Transformer 和 KV Cache 建立张量、生成与缓存心智，再进入 vLLM、SGLang、SFT、veRL，以及以 FSDP2、Megatron Core 为主线的分布式训练。veRL 不是孤立附录：课程会把 SFT 初始权重、vLLM/SGLang rollout、Ray 控制面和 FSDP/Megatron 训练后端接成一条链。

## 本地运行

分别安装博客和课程依赖：

```bash
npm install
npm install --prefix learning-sites/llm-systems
```

然后运行：

```bash
npm --prefix learning-sites/llm-systems run dev
```

生产校验：

```bash
npm --prefix learning-sites/llm-systems run check
npm --prefix learning-sites/llm-systems run build
```

`check` 同时解析所有课程的站内链接、GitHub 固定提交引用和 Mermaid；不存在的页面目标或浮动 `main/master` 源码链接都会阻止部署。最终发布仍需用真实浏览器逐项点击首页、导航、侧栏和正文链接。

博客根目录的 `npm run build` 会先把课程构建到 `public/llm-systems/`，再执行 Astro 构建。
