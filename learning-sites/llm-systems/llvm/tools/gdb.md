---
title: 用 GDB 调试 Clang 与 LLVM
description: 从确认二进制、断点、对象 dump 到定位第一个异常状态的 GDB 实战方法
lesson:
  track: LLVM
  stage: 03 · 工具
  time: 40 分钟
  level: 入门进阶
  question: 如何用 GDB 调试真正执行的 Clang/llc，而不是停在错误的二进制或错误的栈帧？
---

# 用 GDB 调试 Clang 与 LLVM

## 第一步永远是确认二进制

```bash
type -a clang
readlink -f "$(command -v clang)"
file "$(command -v clang)"
clang --version
```

进入 GDB 后再次确认：

```gdb
set pagination off
set print pretty on
show args
start
info files
```

`Starting program:` 必须是你刚构建、带调试信息的二进制。

## 调试 driver 还是 cc1

Clang driver 可能创建子进程，也可能在同一进程进入 cc1。先用：

```bash
clang -### -O0 -S -emit-llvm test.c -o test.ll
```

复制实际 `-cc1` 参数，可减少 driver 噪声。常用设置：

```gdb
set follow-fork-mode child
set detach-on-fork off
catch exec
```

## LLVM 对象优先使用 `dump()`

```gdb
call E->dump()          # Clang Expr
call V->dump()          # llvm::Value
call N->dump(CurDAG)    # SDNode，签名依版本而定
call MI->dump()         # MachineInstr
call MF->dump()         # MachineFunction
```

如果当前 frame 没有变量 `E`，用 `bt`、`up`、`down` 找真正持有它的调用者，不要在错误栈帧反复修改表达式。

## 可复用定位漏斗

1. 在最终断言/报错处停住；
2. dump 当前失败对象；
3. 找它的 ID、type、opcode、operand、register class；
4. 在创建/改写该对象的函数设断点；
5. 找到最后正常状态和第一次异常改写；
6. 用单变量实验验证决策条件。

## 保存会话证据

```gdb
set logging file /tmp/llvm-gdb.log
set logging overwrite on
set logging enabled on
bt
call MI->getParent()->dump()
set logging enabled off
```

::: warning
Debug 构建最易调试但体积和内存成本高。资源有限时可用 `RelWithDebInfo`，并只构建 `clang`、`llc`、`opt` 等目标。
:::

