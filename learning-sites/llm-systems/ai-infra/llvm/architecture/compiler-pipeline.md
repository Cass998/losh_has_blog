---
title: 编译器完整流水线
description: 从 clang driver 到前端、LLVM 优化、后端、汇编器与链接器的完整阶段图
lesson:
  track: LLVM
  stage: 01 · 架构
  time: 35 分钟
  level: 初学者
  question: clang、cc1、LLVM、汇编器和链接器各自负责什么？
---

# 编译器完整流水线

## `clang` 首先是 driver

你在 shell 中执行的 `clang` 首先解析命令行、目标 triple、输入类型和工具链，然后构造多个 job。查看而不执行：

```bash
clang -ccc-print-phases hello.c
clang -### hello.c -o hello
```

典型阶段：

```text
source → preprocessor → compiler → backend → assembler → linker
```

`clang -cc1` 才是前端编译 job 的内部入口；链接通常由系统 linker 或 LLD 完成。

## 用产物切开阶段

```bash
clang -E hello.c -o hello.i                 # 仅预处理
clang -S -emit-llvm hello.c -o hello.ll     # AST/CodeGen → LLVM IR
clang -S hello.c -o hello.s                 # 生成汇编
clang -c hello.c -o hello.o                 # 生成目标文件，不链接
clang hello.o -o hello                      # 链接
```

| 产物 | 仍然包含什么 | 不再包含什么 |
| --- | --- | --- |
| `.i` | 展开后的 C/C++ token | 宏定义结构、include 边界 |
| `.ll` | typed SSA、CFG、metadata | 完整 C++ 类型与 AST |
| `.s` | 目标汇编指令与 directive | LLVM SSA |
| `.o` | section、symbol、relocation、机器码 | 未解析的源码语义 |
| executable | 布局完成的可加载映像 | 多数未使用 section/symbol |

## 前端与后端的合同

Clang CodeGen 负责把语言语义翻译成合法 LLVM IR；LLVM 中后端只相信 IR 合同。例如：

```c
int add(int a, int b) { return a + b; }
```

可能产生：

```llvm
define i32 @add(i32 %a, i32 %b) {
  %sum = add nsw i32 %a, %b
  ret i32 %sum
}
```

`nsw` 不是“运行时检查”，而是优化合同：有符号溢出时结果是 poison。后端据此可以进行更激进变换。

## 根据错误先判断阶段

| 错误 | 已经走到哪里 | 不应先查什么 |
| --- | --- | --- |
| unknown type / no member | Parser/Sema | 寄存器分配 |
| verifier rejected IR | Clang CodeGen 或 IR pass | 汇编编码 |
| `Cannot select` | LLVM IR 已进入后端 | C 预处理 |
| MachineVerifier error | 已产生 MIR | 链接脚本 |
| undefined reference | 编译通常已完成，链接失败 | Instruction Selection |
| illegal instruction at runtime | 可执行文件已生成 | 不能只凭编译成功判断 CPU feature |

