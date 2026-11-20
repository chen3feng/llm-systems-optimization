# Appendix D. Distributed Communication

## D.1 集合通信操作

设参与通信的 rank 数为 `N`，每个 rank 持有的数据量为 `S` 字节。

| 操作 | 语义 | 每 rank 发送量 | 每 rank 接收量 | 典型用途 |
| --- | --- | --- | --- | --- |
| Broadcast | root 的数据发给所有 rank | `S` | `S` | 权重加载 |
| All-reduce | 所有 rank 求和，结果给所有 rank | `S` | `S` | 梯度同步、TP |
| Reduce | 求和，结果只给 root | `S` | `S`（仅 root） | 指标汇总 |
| All-gather | 每个 rank 的片段拼给所有 rank | `S` | `N·S` | ZeRO-3、FSDP |
| Reduce-scatter | 求和后按 rank 切分 | `S` | `S` | ZeRO、FSDP 反向 |
| All-to-all | 每个 rank 给其他每个 rank 发不同数据 | `(N−1)·S/N` | `(N−1)·S/N` | MoE、EP |
| Send/Recv | 点对点 | `S` | `S` | 流水并行 |

## D.2 通信量估算

环状 all-reduce 的通信量是 `2(N−1)/N × S`，即接近 `2S`；树状是 `2S·log N` 的延迟但带宽更省。实际实现会根据消息大小选择算法：

| 消息大小 | 算法 | 特点 |
| --- | --- | --- |
| < 1 MB | 树 / 递归倍增 | 延迟低 |
| > 1 MB | 环 | 带宽利用率高 |

估算带宽时，用“每卡总流量 / 单向带宽”。例如 70B 的梯度是 140 GB，8 卡 all-reduce 每卡发送 `2 × 7/8 × 140 = 245 GB`；NVLink 单向 0.45 TB/s 下约 0.54 秒。

## D.3 并行方式与通信模式

| 并行 | 通信操作 | 频次 | 通信量 |
| --- | --- | --- | --- |
| DP（DDP） | all-reduce 梯度 | 每步 | `2S` |
| ZeRO-1/2 | reduce-scatter + all-gather | 每步 | `2S` |
| ZeRO-3 | all-gather 参数 + reduce-scatter 梯度 | 每层 2 次 | `3S` |
| TP | all-reduce 激活 | 每层 2 次 | `2·B·S·h` |
| PP | send/recv 激活 | 每层边界 | `B·S·h` |
| EP | all-to-all | 每层 2 次 | `2·B·S·h·(N−1)/N` |

TP 的通信量小但频次高，适合放在 NVLink 域内；DP 的通信量大但每步一次，可以跨节点。这是“TP 内、DP 外”布局的由来。

## D.4 NCCL 环境变量

| 变量 | 作用 | 常用值 |
| --- | --- | --- |
| `NCCL_DEBUG` | 日志级别 | `INFO`、`WARN` |
| `NCCL_IB_HCA` | 指定 IB 网卡 | `mlx5_0,mlx5_1` |
| `NCCL_SOCKET_IFNAME` | 指定 socket 网卡 | `eth0` |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 等级 | `2`–`5` |
| `NCCL_ALGO` | 强制算法 | `Ring`、`Tree` |
| `NCCL_PROTO` | 强制协议 | `LL`、`LL128`、`Simple` |
| `NCCL_BUFFSIZE` | 通信缓冲大小 | `4194304` |

`LL128` 协议在小消息上延迟最低，但要求数据对齐；`Simple` 适合大消息。

## D.5 拓扑与带宽

| 链路 | 单向带宽 | 典型场景 |
| --- | --- | --- |
| NVLink 4（H100） | 450 GB/s | 单机 8 卡 |
| PCIe Gen5 x16 | 50 GB/s | 跨卡、CPU-GPU |
| InfiniBand NDR | 50 GB/s | 跨节点 |
| RoCEv2 200G | 25 GB/s | 跨节点（以太网） |

单机内 8 卡通过 NVLink 全互联，任意两卡之间是一跳；跨节点要走网卡，延迟从约 1 µs 涨到 5–10 µs。这决定了 TP 组不应该跨节点。

## D.6 常见问题

- **死锁**：不同 rank 的集合通信顺序不一致。所有 rank 必须以相同顺序调用相同的集合操作；
- **慢 rank**：一个 rank 的延迟会拖住整个集合通信。定位方法是看每个 rank 的耗时分布，而不是平均值；
- **带宽不达标**：检查是否走了 socket 而不是 RDMA（`NCCL_DEBUG=INFO` 会打印选用的传输层），以及消息大小是否落在低效区间；
- **hang 在初始化**：通常是 `MASTER_ADDR`、端口或防火墙问题，先用 `nc -vz` 验证连通性。
