---
title: LLVM Q&A
description: 按编译阶段组织的 LLVM 常见问题、源码证据与调试步骤
lesson:
  track: LLVM
  stage: 04 · Q&A
  time: 按需查阅
  level: 全阶段
  question: 遇到具体 LLVM 问题时，怎样快速找到所属阶段和下一步证据？
---

# LLVM Q&A

这里收录“一个具体问题 → 一条完整证据链”的课程。每篇回答都遵循同一结构：

```text
一句话结论
→ 所属阶段
→ 完整流程图
→ 真实开源源码入口
→ 最小例子
→ 查找失败后的 fallback
→ 可复制调试命令
```

## 前端与 IR

| 问题 | 你会学到什么 |
| --- | --- |
| [builtin 如何一步步变成机器指令？](./builtin-to-instruction) | builtin、intrinsic、TableGen、SelectionDAG 与 fallback |

## 后端定位速查

| 现象 | 先去哪里 |
| --- | --- |
| builtin 名称无法识别 | Clang builtin 注册与目标 feature |
| unsupported/unhandled builtin | `CGBuiltin.cpp` 自动映射与目标 emitter |
| `Cannot select: intrinsic` | intrinsic 签名、legalize、Pat、手写 `Select()` |
| MachineVerifier 失败 | 第一条非法 MIR 与创建它的 pass |
| `Impossible reg-to-reg copy` | pre-RA class/subreg 与 post-RA 物理 COPY |
| `undefined reference` | object symbol、链接输入与顺序 |

后续问答可以直接新增在 `llvm/qa/`，并在这里按阶段登记。

