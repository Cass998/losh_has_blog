# LLM 系统学习实验室

四条独立但互相连接的中文课程：vLLM、SGLang、SFT，以及以 FSDP2、Megatron Core 为主线的分布式训练。

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
npm --prefix learning-sites/llm-systems run check:diagrams
npm --prefix learning-sites/llm-systems run build
```

博客根目录的 `npm run build` 会先把课程构建到 `public/llm-systems/`，再执行 Astro 构建。
