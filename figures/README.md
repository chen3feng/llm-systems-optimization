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
