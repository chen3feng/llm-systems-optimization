# 符号与术语约定

<!-- TODO: 全书符号、单位、缩写和中英文术语对照。 -->

## 数学符号

| 符号 | 含义 |
| --- | --- |
| $B$ | Batch size |
| $S$ | Sequence length |
| $H$ | Hidden dimension |
| $L$ | Number of layers |

## 单位

| 单位 | 含义 |
| --- | --- |
| FLOPs | 浮点运算次数 |
| MACs | 乘加运算次数 |
| GB / GiB | 存储容量（注意十进制与二进制区别） |

## 术语

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| Prefill | 预填充 | 处理输入 prompt 的阶段 |
| Decode | 解码 | 逐 token 生成的阶段 |
| KV Cache | KV 缓存 | 缓存历史 Key/Value，避免重复计算 |
