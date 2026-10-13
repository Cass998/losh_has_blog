---
title: objdump、readelf 与 nm
description: 使用 LLVM/GNU 二进制工具检查汇编、ELF header、section、symbol 和 relocation
lesson:
  track: LLVM
  stage: 03 · 工具
  time: 35 分钟
  level: 初学者
  question: 可执行文件生成后，怎样确认它是什么架构、包含哪些符号和重定位？
---

# objdump、readelf 与 nm

用户常说的 “odbjump” 通常指 `objdump`。LLVM 对应工具是 `llvm-objdump`，GNU binutils 对应 `objdump`。

## 工具职责速查

| 工具 | 最适合回答的问题 |
| --- | --- |
| `file` | 文件格式和大致架构是什么？ |
| `readelf -h` | ELF class、endianness、machine、entry 是什么？ |
| `readelf -S` | 有哪些 section、地址、offset、flags？ |
| `readelf -s` / `nm` | 符号定义在哪里、是否未定义？ |
| `readelf -r` | 哪些位置需要链接器重定位？ |
| `llvm-objdump -d` | 机器码反汇编成了什么指令？ |
| `llvm-objdump -dr` | 指令旁有哪些 relocation？ |
| `llvm-readobj` | 结构化检查 object metadata |

## 一组可复制命令

```bash
file test.o test
readelf -h test.o
readelf -S -W test.o
readelf -s -W test.o
readelf -r -W test.o
nm -C -n test.o
llvm-objdump -d --no-show-raw-insn test.o
llvm-objdump -dr test.o
```

## 自定义目标反汇编失败时

“后端已注册”不保证工具能从文件 header 自动推断 triple。先检查：

```bash
readelf -h a.out
llvm-objdump --version
llvm-objdump -d --triple=<registered-triple> a.out
```

确认实际运行的 `llvm-objdump` 与生成目标文件的工具链来自同一 build/version。

## 链接错误的最短证据链

```bash
nm -C caller.o | rg ' U '
nm -C provider.o | rg ' T | D | B '
readelf -r caller.o
clang caller.o provider.o -Wl,--trace -o app
```

`undefined reference` 是链接输入中缺少可用定义，和后端是否能选出指令属于不同阶段。

