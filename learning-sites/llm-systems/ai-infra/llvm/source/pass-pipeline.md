---
title: Pass 与优化流水线
description: 理解新 Pass Manager、PassBuilder、分析缓存和逐 pass 定位方法
lesson:
  track: LLVM
  stage: 02 · 源码
  time: 35 分钟
  level: 入门进阶
  question: 一个优化 pass 如何进入流水线，怎样证明是哪一步首次改变了 IR？
---

# Pass 与优化流水线

## Pass 不只是“优化算法”

Pass 可以分析、转换或验证不同粒度的 IR：Module、CGSCC、Function、Loop。新 Pass Manager 通过 AnalysisManager 缓存分析结果；转换后必须正确声明哪些分析仍然有效。

## 关键源码入口

| 文件 | 作用 |
| --- | --- |
| `llvm/lib/Passes/PassBuilder.cpp` | 组装 O0/O1/O2/O3 pipeline |
| `llvm/include/llvm/IR/PassManager.h` | pass/analysis 基础接口 |
| `llvm/tools/opt/opt.cpp` | `opt` 工具入口 |
| `llvm/lib/Transforms/` | 各类 IR transforms |

## 用 `opt` 做单变量实验

```bash
clang -O0 -Xclang -disable-O0-optnone -S -emit-llvm test.c -o test.ll
opt -passes=mem2reg -S test.ll -o mem2reg.ll
opt -passes='default<O2>' -S test.ll -o o2.ll
opt -passes='print<domtree>' -disable-output test.ll
```

`optnone` 会阻止许多 function pass；所以 O0 生成的 IR 用于 pass 实验时常需要显式禁用它。

## 找“第一个变化”

```bash
opt -passes='default<O2>' \
  -print-before-all -print-after-all \
  -filter-print-funcs=target_function \
  -disable-output test.ll 2> passes.log
```

定位报告至少包含：

```text
最后正常：After <pass A>
第一个异常：After <pass B>
变化对象：某条 instruction / type / metadata / CFG edge
决策输入：pass B 读取的分析与条件
```

关闭某个 pass 后结果变化，只说明它是充分触发条件之一，不自动证明它创建了最早错误状态。

