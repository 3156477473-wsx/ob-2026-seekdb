# AMD hipVS 加速数据

来源：AMD Radeon Cloud 官方模板 `seekdb-hipvs` 的预执行 Notebook
`seekdb_hipvs_batch_demo_executed.ipynb`，`sk.sweep()` 自动生成。

## 批量检索：CPU（VSAG 嵌套循环）vs GPU（cuVS batch_knn）

| N | cpu_ms | gpu_ms | speedup | cpu_probes_s | gpu_probes_s |
|---|---|---|---|---|---|
| 100 | 375.2 | 524.8 | 0.71 | 266 | 191 |
| 500 | 1541.9 | 564.2 | 2.73 | 324 | 886 |
| 1000 | 2984.1 | 902.9 | 3.30 | 335 | 1108 |
| 2000 | 5872.9 | 794.7 | 7.39 | 341 | 2517 |
| 4000 | 10035.8 | 837.5 | 11.98 | 399 | 4776 |

原始文件见 [summary.csv](summary.csv)。

## 单条查询

| 实现 | 延迟 |
|---|---|
| CPU（lib=vsag） | 7.96 ms |
| GPU（lib=cuvs） | 12.96 ms |

单条查询 GPU 有额外调用开销，无收益。

## 召回率 recall@10

| 实现 | recall@10 |
|---|---|
| CPU（VSAG） | 0.640 |
| GPU（cuVS） | 0.868 |

GPU 路径使用 cuVS CAGRA 索引，在同等批量下召回更高。

## 实验环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon Graphics |
| 架构 | gfx1100 / RDNA3 |
| 显存 | 49136 MB |
| ROCm | 7.2.4 |
| seekdb 二进制 | 已编译 GPU 后端（libcuvs 链接数 ≥ 0） |
| cuVS 桥接库 | OK |
| 向量数 | 10000 |
| 索引构建耗时 | 约 19 s |

## 关键结论

1. **批量越大，GPU 优势越明显**：N=4000 时加速 11.98×，吞吐从 399 提升到 4776 probes/s。
2. **存在交叉点**：N<500 时 GPU 因调用开销反而更慢，N≈500 起 GPU 开始领先。
3. **GPU 同时提升召回**：recall@10 从 0.640 提升到 0.868。
4. **单条查询不适合 GPU**：CPU HNSW 已达毫秒级，GPU 调用开销无法摊薄。
5. **驱动方式无需环境变量**：声明式 `WITH (lib=cuvs)` 或 `dbms_vector.batch_knn`。