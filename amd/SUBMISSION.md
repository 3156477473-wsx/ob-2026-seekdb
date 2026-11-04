# 提交说明 · AMD 赛题

## 题目

> 本题要求结合 AMD Radeon GPU 与 hipVS，优化 seekdb 的 HNSW 索引构建链路，并验证实际加速效果。

## 交付清单

| 要求 | 交付物 | 位置 |
|---|---|---|
| 优化方案 | 技术报告：方案、调用入口、链路分析 | [`docs/REPORT.md`](docs/REPORT.md) |
| 加速效果数据 | 完整数据表（5 个批量档位 + 单条 + 召回） | [`docs/RESULTS.md`](docs/RESULTS.md) · [`evidence/summary.csv`](evidence/summary.csv) |
| 实验环境 | GPU 型号、ROCm 版本、seekdb 编译后端 | [`evidence/01-环境自检.png`](evidence/01-环境自检.png) |
| 实测截图 | 环境、启动、单条对比、批量提速、总结记分牌 | [`evidence/`](evidence/) |
| 可复现方式 | 基准 SQL 与执行说明 | [`bench/`](bench/README.md) |

## 核心结果

### 批量检索：CPU（VSAG）vs GPU（cuVS）

| N | CPU (ms) | GPU (ms) | 加速比 |
|---|---|---|---|
| 100 | 375.2 | 524.8 | 0.71× |
| 500 | 1541.9 | 564.2 | 2.73× |
| 1000 | 2984.1 | 902.9 | 3.30× |
| 2000 | 5872.9 | 794.7 | 7.39× |
| **4000** | **10035.8** | **837.5** | **11.98×** |

### 其他指标

| 项目 | CPU | GPU |
|---|---|---|
| 单条查询延迟 | 7.96 ms | 12.96 ms |
| 吞吐（N=4000） | 399 probes/s | 4776 probes/s |
| recall@10 | 0.640 | 0.868 |

## 结论

1. **批量越大，GPU 优势越明显**：N=4000 时加速 11.98 倍，吞吐提升约 12 倍。
2. **存在交叉点**：N<500 时 GPU 因固定调用开销反而更慢，N≈500 起反超。
3. **单条查询不适合 GPU**：CPU HNSW 已是毫秒级，GPU 调用开销无法摊薄。
4. **GPU 同时提升召回**：cuVS CAGRA 在同等批量下 recall@10 高出 36%。

## 实验环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon Graphics（gfx1100 / RDNA3） |
| 显存 | 49136 MB |
| ROCm | 7.2.4 |
| seekdb | 官方 `seekdb-hipvs` 模板二进制，已编译 GPU 后端（libcuvs 链接正常） |
| 向量数 | 10000 |
| 平台 | AMD Radeon Cloud |

## 复现步骤

1. AMD Radeon Cloud 注册并兑换算力：<https://developer.amd.com.cn/login>
2. Gallery 搜索 `seekdb` → 启动 `seekdb-hipvs` 模板
3. 按 Notebook 顺序执行，`sk.sweep()` 自动产出 `results/summary.csv`
4. 或按 [`bench/bench_batch_knn.sql`](bench/bench_batch_knn.sql) 手工复现

## 数据来源说明

所有数字来自 AMD 官方 `seekdb-hipvs` 模板的 Notebook 实测输出，原始 CSV（`evidence/summary.csv`）与逐页截图（`evidence/*.png`）均已随仓库提交，未做人工修改。