# HNSW 索引构建耗时（CPU vs GPU）

本文件记录 **单独测量索引构建阶段** 的耗时：同一份向量数据，分别用 `LIB=VSAG`（CPU）和 `LIB=CUVS`（GPU）建 HNSW 索引，各自计时。

数据来自 AMD Radeon Cloud 的 `seekdb-hipvs` 模板实测输出。

## 测量方法

```sql
-- 建表（与模板 t_cuvs 相同的结构）
CREATE TABLE bench (c1 INT NOT NULL, c2 VECTOR(128), PRIMARY KEY(c1));
-- 灌入 N 条 128 维随机向量，每批 100 行
-- 单独建索引并计时
CREATE VECTOR INDEX idx ON bench(c2)
  WITH (DISTANCE=L2, TYPE=HNSW, LIB=VSAG, M=16, EF_CONSTRUCTION=200, EF_SEARCH=64);
```

每次测量前 `DROP TABLE` 重建，保证构建起点干净；只对 `CREATE VECTOR INDEX` 语句计时，不含建表与灌数。

## 结果

| N | CPU (VSAG) | GPU (CUVS) | 加速比 |
|---|---|---|---|
| 5,000 | 1.444 s | 1.422 s | 1.02× |
| 10,000 | 2.626 s | 2.623 s | 1.00× |
| 50,000 | 16.975 s | 17.604 s | 0.96× |
| **100,000** | **44.028 s** | **44.356 s** | **0.99×** |

原始数据：[`build_breakdown.csv`](build_breakdown.csv) · [`build_large.csv`](build_large.csv)

## 关键结论

**在 seekdb v1.4.0 上，`LIB=CUVS` 的索引构建阶段与 `LIB=VSAG` 基本持平（0.96×–1.02×），GPU 没有带来加速，大规模下甚至略慢。**

对比同一实例上两个阶段的 GPU 表现：

| 阶段 | CPU | GPU | 加速比 |
|---|---|---|---|
| 索引构建（`CREATE VECTOR INDEX`） | 44.0 s | 44.4 s | **0.99×** |
| 批量检索（`dbms_vector.batch_knn`，N=4000） | 10035.8 ms | 837.5 ms | **11.98×** |

**这说明 `LIB=CUVS` 目前只把「查询」下沉到了 GPU，而「索引构建」仍然走 CPU 路径**——CUVS 的 CAGRA 索引构建能力没有在这条链路上被调用。

## 对赛题的意义

赛题要求「优化 seekdb 的 **HNSW 索引构建链路**」。本测量给出两个可以直接引用的结论：

1. **现状**：构建阶段 GPU 无收益，是最主要的耗时环节（10 万条 44 秒，而单次查询只有 16–24 毫秒）
2. **机会**：把建索引时的批量距离计算与候选筛选下沉到 GPU，是明确的优化方向；CAGRA 的 `build` 接口本身支持 GPU 构建

## 测量中的问题与修正（可复现记录）

复现时容易踩的坑，记录在此便于评审核对：

| 现象 | 原因 | 解法 |
|---|---|---|
| `OSError: [Errno 7] Argument list too long` | 模板的 `sk.sql()` 通过 `mysql -e "SQL"` 执行，单批 1000 条 INSERT 超出 argv 上限 | 每批降到 100 行 |
| `ERROR 1046: No database selected` | `sk.sql()` 每次新开连接，不带默认库 | 每条 SQL 前加 `USE vec;` |
| `ERROR 1235: this type of vector index type is not supported` | 索引参数写错（`type=hnsw_sq8`） | 用模板真实语法：`TYPE=HNSW` |
| 表不存在 | 模板数据建在 `vec` 库，不是 `test` | 用 `SHOW CREATE TABLE t_cuvs` 确认库名与语法 |

模板里 `t_cuvs` 的真实建表语句（作为参数参照）：

```sql
CREATE TABLE `t_cuvs` (
  `c1` int(11) NOT NULL,
  `c2` VECTOR(128) DEFAULT NULL,
  PRIMARY KEY (`c1`),
  VECTOR KEY `i_c2` (`c2`) WITH (DISTANCE=L2, TYPE=HNSW, LIB=CUVS, M=16,
                                 EF_CONSTRUCTION=200, EF_SEARCH=64)
) ...
```

## 环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon Graphics（gfx1100 / RDNA3），49136 MB |
| ROCm | 7.2.4 |
| seekdb | v1.4.0，官方 `seekdb-hipvs` 模板 |
| 向量 | 128 维，确定性伪随机（seed=7） |
| 平台 | AMD Radeon Cloud |