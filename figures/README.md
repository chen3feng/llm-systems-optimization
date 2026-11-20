# Figures

全书插图集中存放，正文使用相对路径引用。

## 命名约定

```text
figures/
├── part-01/
│   └── ch03-roofline-model.svg
├── part-04/
│   └── ch20-gemm-tiling.svg
└── part-05/
    └── ch26-kv-cache-layout.svg
```

- 目录：`part-NN/`
- 文件：`chNN-<short-name>.svg`
- 优先使用 SVG；需要位图时使用 PNG，并同时保留源文件。

## 正文引用

图号按章编号，`图 3-1` 表示第 3 章第 1 幅。图片下一行给出说明：

```markdown
![图 3-1](../../figures/part-01/ch03-roofline-model.svg)

图 3-1　Roofline 模型：横轴是算术强度，纵轴是可达性能
```

文件名与正文图号一一对应：`ch03-roofline-model.svg` 就是第 3 章的图 3-1。
章内插图按 `chNN-<主题>.svg` 命名，不出现连续编号，增删图时只需改正文的图号。
