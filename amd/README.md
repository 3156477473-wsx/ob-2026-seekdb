# AMD 赛题：使用 hipVS 加速 HNSW 索引构建

2026 OceanBase 数据库大赛 · AMD Radeon GPU × hipVS × seekdb

## 一句话结论

在 AMD Radeon GPU（RDNA3）+ hipVS 下，seekdb 的批量向量检索相比 CPU 基线**最高加速 11.98 倍**，同时 **recall@10 从 0.640 提升到 0.868**；但单条查询场景 GPU 无收益，交叉点在批量约 300–500。

## 关键数据

| 场景 | CPU (VSAG) | GPU (cuVS) | 结果 |
|---|---|---|---|
| 单条查询 | 7.96 ms | 12.96 ms | GPU 更慢，无收益 |
| 批量 N=500 | 1541.9 ms | 564.2 ms | **2.73×** |
| 批量 N=1000 | 2984.1 ms | 902.9 ms | **3.30×** |
| 批量 N=2000 | 5872.9 ms | 794.7 ms | **7.39×** |
| 批量 N=4000 | 10035.8 ms | 837.5 ms | **11.98×** |
| recall@10 | 0.640 | **0.868** | GPU 更高 |
| 吞吐（N=4000） | 399 probes/s | **4776 probes/s** | 12× |

## 实验环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon Graphics（gfx1100 / RDNA3） |
| 显存 | 49136 MB |
| ROCm | 7.2.4 |
| seekdb | 官方 `seekdb-hipvs` 模板，已编译 GPU 后端 |
| 向量数 | 10000 |

## 目录

```
amd/
├── README.md            本文件
├── docs/
│   ├── REPORT.md        技术报告（问题/方案/结果/链路分析）
│   └── RESULTS.md       完整数据表
├── bench/
│   ├── README.md        基准测试说明
│   └── bench_batch_knn.sql  可复现的 SQL
└── evidence/
    ├── 01-环境自检.png
    ├── 02-启动seekdb.png
    ├── 03-单条查询对比.png
    ├── 04-批量提速核心.png
    ├── 05-总结记分牌.png
    ├── seekdb_hipvs_speedup.png
    └── summary.csv
```

## 两个调用入口（无需环境变量）

```sql
-- 1. 声明式：为索引指定 GPU 后端库
CREATE VECTOR INDEX idx_cuvs ON t_cuvs(c2) WITH (lib=cuvs);

-- 2. 批量算子：一次提交全部查询
CALL dbms_vector.batch_knn('t_cuvs', probes, k, 'out_table');
```

## 复现

1. AMD Radeon Cloud 注册并兑换算力：<https://developer.amd.com.cn/login>
2. Gallery 搜索 `seekdb` → 启动 `seekdb-hipvs` 模板
3. 依次执行 Notebook 单元，`sk.sweep()` 自动产出 `results/summary.csv`
4. 或按 `bench/bench_batch_knn.sql` 手工复现

## 结论与建议

- **批量越大，GPU 优势越明显**：N=4000 时 11.98×，吞吐提升约 12 倍。
- **存在交叉点**：N<500 时 GPU 因固定调用开销反而更慢，建议批量 ≥ 1000 再启用 GPU。
- **单条查询不要用 GPU**：CPU HNSW 已是毫秒级，GPU 无收益。
- **GPU 同时改善召回**：cuVS CAGRA 在同等批量下 recall@10 高出 36%。

## 说明

本仓库为 seekdb 源码仓库（`ob-2026-seekdb`），`amd/` 目录存放本赛题的全部交付物。基准数据来自 AMD 官方 `seekdb-hipvs` 模板的预执行 Notebook，原始 CSV 与截图均保存在 `evidence/`。