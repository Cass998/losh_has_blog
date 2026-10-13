---
title: lit、FileCheck 与 MIR 测试
description: 为前端诊断、LLVM IR、汇编和 Machine IR 编写稳定的回归测试
lesson:
  track: LLVM
  stage: 03 · 工具
  time: 35 分钟
  level: 入门进阶
  question: 修复编译器问题后，如何写一个既精确又不过拟合的回归测试？
---

# lit、FileCheck 与 MIR 测试

## LLVM 测试的三层

| 测试层 | 常见目录 | 检查什么 |
| --- | --- | --- |
| Clang Sema/CodeGen | `clang/test/` | 诊断、AST、LLVM IR |
| LLVM transforms/CodeGen | `llvm/test/` | IR pass、MIR、汇编 |
| 工具/链接器 | `llvm/test/tools/`、`lld/test/` | CLI、object、link behavior |

## FileCheck 基本模型

```c
// RUN: %clang_cc1 -triple aarch64 -emit-llvm -o - %s | FileCheck %s
// CHECK-LABEL: define{{.*}} @test
// CHECK: call void @llvm.aarch64.dmb(i32 1)
```

常用 directive：

- `CHECK-LABEL`：建立函数/区块边界；
- `CHECK-NEXT`：必须紧邻；
- `CHECK-NOT`：两个正检查之间不能出现；
- `CHECK-DAG`：顺序不固定的一组匹配；
- `[[NAME:regex]]` / `[[NAME]]`：捕获并复用值。

## MIR 测试适合什么

当问题发生在寄存器分配、spill、pseudo expansion 或特定 MachineInstr pass，MIR 可以绕开前端和 IR 选择，保存更小、更稳定的机器层输入。

```bash
llc -stop-before=greedy input.ll -o case.mir
llc -run-pass=greedy -verify-machineinstrs case.mir -o -
```

## 一个好回归测试应包含

- 正例：新支持路径产生正确 IR/MIR/汇编；
- 负例：非法立即数或 feature 缺失仍被拒绝；
- 类型边界：32/64 位、vector/scalar、subregister；
- O0/O2：如果问题与 pipeline 有关，两者都覆盖；
- 运行验证：语义 bug 不能只检查“编译不崩溃”。

::: tip
测试应锁定合同，而不是锁定无关的寄存器编号、临时 SSA 名字或整个函数的完整输出。
:::

