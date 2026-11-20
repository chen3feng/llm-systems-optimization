# Appendix G. C++/CUDA Development

本书的实践代码以 C++17 和 CUDA 12 为基准。本附录给出构建、调试和 CI 的最小配置。

## G.1 构建

```cmake
cmake_minimum_required(VERSION 3.24)
project(llm_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80;90)

add_library(kernels STATIC kernels.cu)
target_compile_options(kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
```

关键选项：

| 选项 | 作用 |
| --- | --- |
| `-O3` | 主机端优化 |
| `--use_fast_math` | 启用快速数学函数，精度略降 |
| `-lineinfo` | 保留行号信息，供 profiler 使用 |
| `--ptxas-options=-v` | 打印寄存器用量和 shared memory 占用 |
| `-Xptxas -dlcm=cg` | 全局加载绕过 L1（按需） |
| `-arch=sm_90a` | 启用 Hopper 专属指令 |

`--use_fast_math` 会打开 FTZ 和近似的 `exp`、`log`，在数值敏感的场景要慎用。

## G.2 调试

| 工具 | 用途 |
| --- | --- |
| `compute-sanitizer` | 检测越界、未初始化读、竞态 |
| `cuda-gdb` | 断点、查看寄存器、单步 |
| `nsys profile` | 时间线、kernel 间隙、API 耗时 |
| `ncu` | 单 kernel 的占用率、访存、停顿 |

```bash
compute-sanitizer --tool memcheck ./app
compute-sanitizer --tool racecheck ./app
nsys profile -o out --trace=cuda,nvtx ./app
ncu --set full -k my_kernel ./app
```

`compute-sanitizer` 会让程序慢 10–100 倍，只在小输入上跑。`racecheck` 能抓到 `__syncthreads` 缺失和 shared memory 竞争，这类 bug 在正常运行时表现为偶发的错误结果。

## G.3 错误处理

CUDA 的错误是异步的：kernel 内的越界访问不会在 launch 时返回，而是在下一次同步时以“unspecified launch failure”的形式出现，且不告诉你哪个 kernel 出错。

两个习惯：

- 每个 API 调用后检查返回值（见附录 C 的 `CUDA_CHECK`）；
- 在可疑的 kernel 后加 `cudaDeviceSynchronize()`，缩小错误范围。

调试构建里可以打开 `CUDA_LAUNCH_BLOCKING=1`，让每次 launch 同步返回，代价是性能大幅下降。

## G.4 性能验证

改动 kernel 后至少验证三件事：

1. **正确性**：和 CPU reference 对比，用相对误差而不是绝对误差；
2. **数值稳定性**：极端输入（全零、极大值、NaN）不应产生非法结果；
3. **性能**：和基线对比，注意附录 F 里的预热和锁频。

一个常见的错误是只测了大形状。小形状（decode 的 `M=1`）往往才是真实场景，而它更容易被 tile 大小和边界处理拖慢。

## G.5 CI

GPU 相关的 CI 分两层：

- **CPU 层**：编译检查、静态分析、CPU reference 的单元测试，每次提交都跑；
- **GPU 层**：真实 kernel 的正确性和性能测试，按需触发或 nightly。

性能测试进 CI 要小心：共享 GPU 上的噪声会让结果不可比。做法是固定机器、锁频、跑多次取中位数，并设置较宽的阈值（比如回退超过 10% 才报警）。

```yaml
# 示例：nightly 性能测试
- name: build
  run: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build build -j
- name: correctness
  run: ./build/tests
- name: benchmark
  run: nvidia-smi -lgc 1980 && ./build/bench --iters 100
```

## G.6 工程约定

- **头文件里不放 kernel**：`.cu` 编译单元分开，避免重复实例化；
- **模板 kernel 显式实例化**：减少编译时间，避免链接错误；
- **RAII 管理 CUDA 资源**：stream、event、内存用包装类，异常路径也能释放；
- **避免在热路径分配**：预分配 workspace，用内存池；
- **命名空间隔离**：`llm_lab::cuda`、`llm_lab::cpu`，便于 CPU/GPU 双实现对比。
