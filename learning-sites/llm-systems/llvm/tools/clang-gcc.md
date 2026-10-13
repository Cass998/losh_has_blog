---
title: Clang、GCC 与编译阶段工具箱
description: 用 Clang、GCC、opt 和 llc 分阶段生成、比较并验证编译产物
lesson:
  track: LLVM
  stage: 03 · 工具
  time: 35 分钟
  level: 初学者
  question: Clang 和 GCC 应该怎样配合使用，如何避免被一条大命令隐藏阶段？
---

# Clang、GCC 与编译阶段工具箱

## Clang 与 GCC 的关系

Clang 和 GCC 都可以作为 C/C++ compiler driver，但内部 IR、优化器和后端不同。两者生成相同结果不证明中间路径相同；结果不同也不自动证明其中一个编译器有 bug，先排除未定义行为、ABI、链接库和编译选项差异。

## 必会命令

```bash
# 查看版本、目标与安装信息
clang --version
gcc --version
clang -print-target-triple

# 查看阶段和真实子命令
clang -ccc-print-phases test.c
clang -### -O2 test.c -o test
gcc -v -O2 test.c -o test

# 保留中间产物
clang -save-temps=obj -O2 test.c -o test
gcc -save-temps=obj -O2 test.c -o test
```

## 把 Clang 拆成 LLVM 工具

```bash
clang -O0 -S -emit-llvm test.c -o test.ll
opt -passes='default<O2>' -S test.ll -o test.opt.ll
llc -O2 test.opt.ll -o test.s
llvm-mc -triple=x86_64 -filetype=obj test.s -o test.o
clang test.o -o test
```

这条拆分链适合定位“前端生成错、IR pass 改错、后端选错还是链接错”。实际 Clang pipeline 的参数会更多，实验时要记录 triple、cpu、features、relocation model 和 optimization level。

## 比较编译器时的最小纪律

```bash
clang -O2 -Wall -Wextra test.c -o clang.out
gcc   -O2 -Wall -Wextra test.c -o gcc.out
./clang.out; echo $?
./gcc.out; echo $?
```

- 使用同一源码和输入；
- 清除旧 `.o`，避免混合对象；
- 保存完整命令、stderr 和 exit status；
- 不用“编译成功”替代运行结果与产物验证。

