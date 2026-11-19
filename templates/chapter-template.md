# Chapter Template

> 复制本文件到目标 Part 目录，重命名为 `chNN-kebab-case-title.md`，再填写内容。

```markdown
---
chapter: 0
part: 0
title: Chapter Title
level: tbd
status: outline
prerequisites: []
lab:
---

# Chapter 0. Chapter Title

## 0.1 Problem Statement

明确待解决的问题。

## 0.2 Background

必要的数学和硬件背景。

## 0.3 Fundamental Principles

核心原理。

## 0.4 Algorithm

算法及数学推导。

## 0.5 Data Structures

关键数据结构和内存布局。

## 0.6 Implementation

C++ / CUDA 实现。

## 0.7 Performance Analysis

计算、访存、通信和延迟分析。

## 0.8 Trade-offs

适用条件和局限。

## 0.9 Real-world Systems

开源项目源码分析。

## 0.10 Experiments

可重复的 Benchmark。

## 0.11 Further Reading

论文、源码和技术文档。

## 本章小结
```

## 说明

- 章节正文章节号使用 `NN.M`，与全章编号一致。
- 不是所有章节都平均分配篇幅；按内容类型调整各节权重。
- 每个重要算子尽量提供三个层次：Reference、Optimized、Production。
- 实验必须记录环境、命令、原始结果和性能波动。
