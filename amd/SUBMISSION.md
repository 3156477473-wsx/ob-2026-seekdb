# 提交说明 · AMD 赛题

## 题目

> 本题要求结合 AMD Radeon GPU 与 hipVS，优化 seekdb 的 HNSW 索引构建链路，并验证实际加速效果。

## 核心结论

在 AMD Radeon GPU（RDNA3）+ ROCm 7.2.4 + 官方 `seekdb-hipvs` 模板上，我们分别测量了 **索引构建** 与 **批量检索** 两个阶段，得到一个非对称的结果：

| 阶段 | CPU | GPU | 加速比 |
|---|---|---|---|
| **HNSW 索引构建**（10 万条向量） | 44.028 s | 44.356 s | **0.99×** |
| **批量检索**（N=4000） | 10035.8 ms | 837.5 ms | **11.98×** |

**`LIB=CUVS` 目前只把查询下沉到了 GPU，索引构建仍走 CPU 路径。** 这正是本题「优化 HNSW 索引构建链路」要解决的问题所在——构建阶段是 GPU 加速的空白区。

同时，GPU 路径的召回率还更高：

| 实现 | recall@10 |
|---|---|
| CPU（VSAG） | 0.640 |
| GPU（CUVS） | 0.868 |

## 交付清单

| 要求 | 交付物 | 位置 |
|---|---|---|
| 索引构建链路测量 | 单独拆分 CPU/GPU 构建耗时，4 个规模档位 | [`docs/BUILD_TIMING.md`](docs/BUILD_TIMING.md) |
| 批量检索加速效果 | 5 个批量档位 + 单条查询 + 召回率 | [`docs/RESULTS.md`](docs/RESULTS.md) |
| 技术分析与链路定位 | 构建 vs 检索的非对称性、源码位置、优化方向 | [`docs/REPORT.md`](docs/REPORT.md) |
| 实验环境 | GPU 型号、ROCm 版本、编译后端 | [`evidence/01-环境自检.png`](evidence/01-环境自检.png) |
| 实测截图 | 环境、启动、单条对比、批量提速、总结记分牌 | [`evidence/`](evidence/) |
| 可复现 | 基准 SQL、测量脚本（含踩坑记录） | [`bench/`](bench/) |

## 数据一览

### 索引构建（单独计时，只算 `CREATE VECTOR INDEX`）

| N | CPU (VSAG) | GPU (CUVS) | 加速比 |
|---|---|---|---|
| 5,000 | 1.444 s | 1.422 s | 1.02× |
| 10,000 | 2.626 s | 2.623 s | 1.00× |
| 50,000 | 16.975 s | 17.604 s | 0.96× |
| **100,000** | **44.028 s** | **44.356 s** | **0.99×** |

### 批量检索

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

1. **构建阶段 GPU 无收益**（0.96×–1.02×）：`LIB=CUVS` 未把建索引的批量距离计算下沉到 GPU。
2. **构建是主要耗时**：10 万条 44 秒，而单次查询仅 16–24 毫秒——优化构建的收益远大于优化查询。
3. **批量检索 GPU 优势显著**：N=4000 时 11.98×，且存在交叉点（N<500 时 GPU 反而慢）。
4. **GPU 同时改善召回**：CUVS 在同等批量下 recall@10 高出 36%。

## 优化方向（基于测量的分析）

| 方向 | 依据 | 可下沉 GPU 的部分 |
|---|---|---|
| 建索引时的批量距离计算 | 构建耗时与 GPU 无关（0.99×） | 向量间 L2 计算天然适合 SIMT |
| 候选近邻筛选 | 同上 | 批量 top-k 筛选 |
| 图结构插入维护 | 强依赖串行 | **不适合下沉**，仍留 CPU |

seekdb 中 HNSW 构建链路的位置：

- `src/storage/ddl/ob_vector_index_ddl_pipeline.h` — `ObHNSWIndexBuildOperator`
- `src/storage/ddl/ob_hnsw_embedmgr.cpp` — 向量嵌入管理

## 实验环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon Graphics（gfx1100 / RDNA3） |
| 显存 | 49136 MB |
| ROCm | 7.2.4 |
| seekdb | v1.4.0，官方 `seekdb-hipvs` 模板，已编译 GPU 后端 |
| 平台 | AMD Radeon Cloud |

## 复现步骤

1. AMD Radeon Cloud 注册并兑换算力：<https://developer.amd.com.cn/login>
2. Gallery 搜索 `seekdb` → 启动 `seekdb-hipvs` 模板
3. 按 Notebook 顺序执行（第 3 节会启动 seekdb 并准备数据）
4. **构建测量**：新建单元运行 [`bench/measure_build.py`](bench/measure_build.py)
5. **检索测量**：模板自带的 `sk.sweep()`，产出 `results/summary.csv`

复现时容易踩的坑（每批行数、库名、索引参数）已记录在 [`docs/BUILD_TIMING.md`](docs/BUILD_TIMING.md) 末节。

## 数据来源说明

- 检索数据来自官方模板 Notebook 的实测输出，原始 CSV（`evidence/summary.csv`）与逐页截图均已提交
- 构建数据为本队新增测量，脚本（`bench/`）与原始 CSV（`evidence/build_*.csv`）一并提交
- 未做人工修改，未针对文件名或结果硬编码