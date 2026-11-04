# 基准测试说明

## 目的

复现「CPU（VSAG）vs GPU（cuVS）」在 seekdb 批量向量检索与索引构建上的耗时对比。

## 环境

- 平台：AMD Radeon Cloud，模板 `seekdb-hipvs`
- GPU：AMD Radeon Graphics（gfx1100 / RDNA3），49136 MB 显存
- ROCm：7.2.4
- seekdb：模板预置二进制，已编译 GPU 后端

## 前置

1. 在 AMD Radeon Cloud 启动 `seekdb-hipvs` 模板
2. 执行 Notebook 第 3 节，确认输出：
   ```
   data loaded: t_cuvs (lib=cuvs, GPU) + t_vsag (lib=vsag, CPU), 10000 vectors each
   vector index ready (~19s)
   ```

## 执行

按 `bench_batch_knn.sql` 中的顺序执行。核心是第 3 节：对 N ∈ {100, 500, 1000, 2000, 4000} 分别计时。

## 记录格式

| N | cpu_ms | gpu_ms | speedup |
|---|---|---|---|
| 100 | 375.2 | 524.8 | 0.71 |
| 500 | 1541.9 | 564.2 | 2.73 |
| 1000 | 2984.1 | 902.9 | 3.30 |
| 2000 | 5872.9 | 794.7 | 7.39 |
| 4000 | 10035.8 | 837.5 | 11.98 |

`speedup = cpu_ms / gpu_ms`。

## 判定

- `speedup > 1` 表示 GPU 更快
- 交叉点出现在 N 约 300–500 之间
- N=100 时 speedup < 1，说明 GPU 调用开销未摊薄，属预期结果

## 说明

本目录只包含复现用的 SQL 与说明。测量脚本与图表由模板 Notebook 的 `sk.sweep()` 生成，原始数据见 `../evidence/summary.csv`。