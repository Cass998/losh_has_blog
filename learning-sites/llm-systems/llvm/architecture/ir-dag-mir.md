---
title: LLVM IR、SelectionDAG 与 MIR
description: 理解 LLVM 编译中三种核心表示的职责、边界和可观测方法
lesson:
  track: LLVM
  stage: 01 · 架构
  time: 35 分钟
  level: 入门进阶
  question: IR、DAG、MIR 为什么同时存在，错误应该在哪一层观察？
---

# LLVM IR、SelectionDAG 与 MIR

## 三种表示解决三类问题

| 表示 | 核心关注 | 典型内容 |
| --- | --- | --- |
| LLVM IR | 与目标大体无关的程序语义 | SSA、basic block、load/store、call |
| SelectionDAG | 指令选择期间的值/副作用依赖 | value、chain、glue、合法化节点 |
| MIR | 目标相关机器函数 | target opcode、寄存器类、frame index |

## LLVM IR：优化器共同语言

IR 使用 SSA：多数值只定义一次。控制流合流处通过 `phi` 选择来自不同前驱边的值。

```llvm
merge:
  %v = phi i32 [ %a, %left ], [ %b, %right ]
```

查看和验证：

```bash
clang -O0 -S -emit-llvm test.c -o test.ll
llvm-as test.ll -o test.bc
llvm-dis test.bc -o -
opt -passes=verify test.ll -disable-output
```

## SelectionDAG：把语义压到目标能执行的形状

DAG 中：

- value edge 表示数据依赖；
- chain 约束 load/store/call 等副作用顺序；
- glue 约束必须粘在一起选择/调度的节点；
- legalization 把目标不支持的类型或操作扩展、拆分或自定义降低。

因此 `Cannot select` 不一定只是“缺一条 Pat”，也可能是类型尚未合法化、谓词不满足或节点本应 custom lower。

## MIR：机器层的可保存检查点

```text
%0:gpr = COPY $x0
%1:gpr = ADDI %0, 1
$x10 = COPY %1
PseudoRET
```

这里开始出现：

- 虚拟/物理寄存器；
- register class；
- target opcode；
- stack slot、kill/dead、implicit operand；
- MachineFunction 与 MachineBasicBlock。

常用命令：

```bash
llc -mtriple=riscv64 -stop-after=finalize-isel test.ll -o test.mir
llc -run-pass=machineverifier test.mir -o /dev/null
llc -print-before-all -print-after-all test.ll -o /dev/null 2> passes.log
```

::: tip 定位原则
IR 正确不代表 MIR 正确；最终汇编错误也不证明错误由最后一个 pass 创建。寻找“最后正常表示”和“第一个异常表示”。
:::

