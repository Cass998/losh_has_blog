---
title: LLVM 六周学习路线
description: 从编译阶段、IR、后端、工具到独立定位问题的六周源码学习安排
lesson:
  track: LLVM
  stage: 00 · 路线
  time: 20 分钟
  level: 初学者
  question: LLVM 内容太多，应该按什么顺序学习并留下哪些验收证据？
---

# LLVM 六周学习路线

目标不是六周“看完 LLVM”，而是建立一条可以反复复用的定位链：**阶段 → 表示 → 核心对象 → 创建者 → 消费者 → 运行证据。**

## 第 1 周：切开编译阶段

- 阅读：[编译器完整流水线](../architecture/compiler-pipeline)
- 实验：对同一个 `add.c` 生成 `.i`、`.ll`、`.s`、`.o` 和可执行文件。
- 工具：`clang -###`、`clang -ccc-print-phases`、`file`。
- 验收：能解释预处理、编译、汇编、链接分别消费什么、产生什么。

## 第 2 周：理解中间表示

- 阅读：[IR、SelectionDAG 与 MIR](../architecture/ir-dag-mir)
- 实验：比较 `-O0` 与 `-O2` IR；用 `opt -passes=...` 单独运行 pass。
- 工具：`llvm-as`、`llvm-dis`、`opt`。
- 验收：能区分 SSA value、basic block、phi、target opcode 和物理寄存器。

## 第 3 周：进入 Clang 与 Pass

- 阅读：[Clang 前端](../source/clang-frontend)、[Pass 流水线](../source/pass-pipeline)
- 实验：AST dump、IR dump、`-print-before-all/-print-after-all`。
- 验收：能找到一个 C 表达式对应的 AST 节点和 IR 指令。

## 第 4 周：进入后端

- 阅读：[Target 后端](../architecture/target-backend)、[指令选择](../source/instruction-selection)
- 实验：选择 AArch64 或 RISC-V 的一个简单 `add` pattern，从 `.td` 追到汇编。
- 验收：能解释 TableGen 构建期生成与用户程序编译期运行的边界。

## 第 5 周：寄存器与目标文件

- 阅读：[寄存器分配](../source/register-allocation)、[MC 与目标文件](../source/mc-object)
- 实验：保存 MIR、强制产生 spill、查看 `.text`、symbol 和 relocation。
- 工具：`llc`、`llvm-mca`、`llvm-objdump`、`readelf`、`nm`。
- 验收：能区分虚拟寄存器、物理寄存器、stack slot、section 和 relocation。

## 第 6 周：调试一条真实链路

- 阅读：[GDB 调试 LLVM](../tools/gdb)、[测试工具](../tools/testing)、[Q&A](../qa/)
- 实验：为一个公开 builtin 建立 AST → IR → DAG pattern → 指令证据链。
- 验收：写一份“最后正常状态 / 第一个异常状态 / 决策函数 / 最小复现”的定位报告。

## 每周固定产物

```text
notes/week-N/
├── commands.sh       # 真正执行过的命令
├── source-map.md     # 文件、函数、对象
├── artifacts/        # .ll/.mir/.s/.o/log
└── conclusion.md     # 已证实、推断、待验证、已排除
```

::: warning 不要用“读了多少文件”衡量进度
源码学习的有效证据是：你能否从输出回到生成它的函数，并用一个实验改变输入，预测输出如何变化。
:::

