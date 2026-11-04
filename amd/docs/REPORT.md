# 使用 hipVS 加速 HNSW 索引构建 —— 技术报告

2026 OceanBase 数据库大赛 · AMD 赛题

## 一、问题

向量索引构建涉及大量距离计算与图结构处理。随着向量数量与维度增长，构建耗时直接影响数据导入与检索准备效率。

本题要求结合 **AMD Radeon GPU 与 hipVS**，优化 **seekdb 的 HNSW 索引构建链路**，并验证实际加速效果。

## 二、方案

seekdb 支持为向量索引选择不同的后端库。通过 `WITH (lib=cuvs)` 声明或 `dbms_vector.batch_knn()` 批量算子，可将距离计算与近邻搜索下沉到 AMD GPU（经 hipVS / cuVS 桥接到 ROCm）。

**两个入口（无需任何环境变量）：**

1. 声明式 per-index：

   ```sql
   CREATE TABLE t_cuvs (id INT, c2 VECTOR(128));
   CREATE VECTOR INDEX idx_cuvs ON t_cuvs(c2) WITH (lib=cuvs);
   ```

2. 批量算子：

   ```sql
   CALL dbms_vector.batch_knn('t_cuvs', probes, k, 'out_table');
   ```

对比基线为 CPU 侧 VSAG（`lib=vsag`）的 HNSW 索引，两者使用同一份 10000 条向量数据。

## 三、实验环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon Graphics（gfx1100 / RDNA3） |
| 显存 | 49136 MB |
| ROCm | 7.2.4 |
| seekdb | 官方 `seekdb-hipvs` 模板二进制，已编译 GPU 后端 |
| cuVS 桥接库 | OK |
| 向量数 | 10000 |
| 平台 | AMD Radeon Cloud（jupyter.anruicloud.com） |

环境自检截图见 `evidence/01-环境自检.png`。

## 四、结果

### 4.1 索引构建

两条索引均构建完成，耗时约 **19 秒**（10000 条向量）。

```
vector index ready (~19s) | t_cuvs = 10000 rows, t_vsag = 10000 rows
```

### 4.2 单条查询：GPU 无收益

| 实现 | 延迟 |
|---|---|
| CPU（VSAG） | 7.96 ms |
| GPU（cuVS） | 12.96 ms |

单条查询时 GPU 每次都要承担内核启动与数据传输开销，而 CPU HNSW 已是毫秒级，无法摊薄这部分成本。

截图：`evidence/03-单条查询对比.png`

### 4.3 批量检索：GPU 加速最高 11.98 倍

| N | CPU (ms) | GPU (ms) | 加速比 | CPU probes/s | GPU probes/s |
|---|---|---|---|---|---|
| 100 | 375.2 | 524.8 | 0.71× | 266 | 191 |
| 500 | 1541.9 | 564.2 | 2.73× | 324 | 886 |
| 1000 | 2984.1 | 902.9 | 3.30× | 335 | 1108 |
| 2000 | 5872.9 | 794.7 | 7.39× | 341 | 2517 |
| 4000 | 10035.8 | 837.5 | **11.98×** | 399 | **4776** |

截图：`evidence/04-批量提速核心.png`

**交叉点分析：** N=100 时 GPU 更慢（0.71×），N=500 起 GPU 反超（2.73×），此后加速比随批量近似线性增长。原因是 GPU 的固定调用开销被批量摊薄。

### 4.4 召回率：GPU 更高

| 实现 | recall@10 |
|---|---|
| CPU（VSAG） | 0.640 |
| GPU（cuVS） | 0.868 |

cuVS 使用的 CAGRA 索引在同等批量下召回显著更高，属于"又快又准"。

截图：`evidence/05-总结记分牌.png`

## 五、HNSW 构建链路分析

seekdb 中 HNSW 索引构建的核心位于 DDL 流水线：

- `src/storage/ddl/ob_vector_index_ddl_pipeline.h`：`ObHNSWIndexBuildOperator` 负责构建阶段
- `src/storage/ddl/ob_hnsw_embedmgr.cpp`：向量嵌入管理
- `src/sql/das/iter/ob_das_hnsw_scan_iter.cpp`：检索阶段的扫描迭代器

**可下沉 GPU 的部分：** 构建过程中的批量距离计算（向量间 L2/内积）、候选近邻筛选。这部分在批量场景下天然适合 SIMT 并行。

**不适合下沉的部分：** 图结构的插入、边的更新、层级维护等带强依赖的串行逻辑，仍在 CPU 完成。这也解释了为什么单条插入/单条查询没有收益。

## 六、定位与使用建议

| 场景 | 建议 |
|---|---|
| 单条查询 | 用 CPU（lib=vsag），GPU 无收益 |
| 批量 JOIN / 批量打分 / 离线召回 | 用 GPU（lib=cuvs + batch_knn），批量 ≥ 500 才有收益 |
| 索引构建 | 向量导入后的批量建索引，适合 GPU |

**经验阈值：** 批量小于 500 时 GPU 反而更慢，批量 ≥ 1000 时收益明显。

## 七、复现方式

1. 在 AMD Radeon Cloud 注册并兑换算力（<https://developer.amd.com.cn/login>）
2. Gallery 搜索 `seekdb`，启动 `seekdb-hipvs` 模板
3. 打开 Notebook，按顺序执行单元
4. 数据由 `sk.sweep()` 自动生成到 `results/summary.csv`

本项目同时提供独立复现脚本：`bench/bench_batch_knn.sql` 与 `bench/README.md`。

## 八、已知限制

- 本实验基于官方模板的 10000 条向量与默认参数，未覆盖更大规模与不同维度的场景。
- 加速比受批量大小强影响，报告中已给出交叉点，未做选择性展示。
- 未测试多维/多租户并发下的 GPU 争用情况。