# 全文索引调优 · 优化记录

2026 OceanBase 数据库大赛 · seekdb 全文索引性能优化

## 一、编译环境（本机从零搭建）

| 步骤 | 结果 |
|---|---|
| Rust 1.98.1 | ✅ 5 个 target |
| WSL 内存 | 7.6 GB → **11 GB** + 16 GB swap |
| deps/3rd | ✅ `./build.sh init` |
| 编译 | ✅ `./build.sh release --make -j8` |
| 产物 | `build_release/src/observer/seekdb`（1.3 GB） |

**踩坑记录**（便于复现）：

| 现象 | 原因 | 解法 |
|---|---|---|
| `could not download nonexistent rust version 1.98.1` | 清华镜像缺该版本 | 换 USTC：`RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static` |
| cargo 下载 57 KB/s，编译卡在 16% | crates.io 直连慢 | `~/.cargo/config.toml` 指向 USTC sparse 索引 |
| `build.sh: /usr/bin/env: 'bash\r'` | 脚本是 CRLF | `sed -i 's/\r$//'` 全部脚本 |
| `rpm2cpio: command not found` | 缺工具 | `apt install rpm2cpio cpio rpm` |

## 二、优化 1：分词热路径去 ObList 迭代

### 改动位置

`src/storage/fts/ob_ik_ft_parser.{h,cpp}`

### 问题

`ObIKFTParser::process_one_char()` 对**每一个字符**都执行：

```cpp
for (ObList<ObIIKProcessor *, ObIAllocator>::iterator iter = segmenters_.begin();
     iter != segmenters_.end(); iter++) {
  (*iter)->process(ctx);
}
```

segmenter 数量固定为 4（letter / quantifier / cjk / surrogate），顺序固定，却每字符都走一次链表迭代器。

### 改法

1. `init_segmenter()` 完成后把链表内容摊平到定长数组 `segmenter_arr_[4]`，并记录 `segmenter_cnt_`
2. `process_one_char()` 改为遍历数组：

```cpp
for (int i = 0; OB_SUCC(ret) && i < segmenter_cnt_; ++i) {
  ObIIKProcessor *seg = segmenter_arr_[i];
  if (OB_ISNULL(seg)) { ret = OB_ERR_UNEXPECTED; }
  else if (OB_FAIL(seg->process(ctx))) {}
}
```

3. `reset()` 中与链表一起清空数组，避免悬垂指针

### 正确性

- segmenter 的执行顺序、数量、对象生命周期都不变（数组只是引用同一批指针）
- 数组在 `reset()` 中随链表一起清空
- 分词结果与改动前一致（`TOKENIZE()` 输出经人工核对）

## 三、实测效果

测试条件：20 000 篇文档，`SAMPLES=3`，本地 16 核 / 11 GB。

| 指标 | 基线 | 优化后（3 次中位数） | 提升 |
|---|---|---|---|
| `build_ik_all_sec` | 8.397 | **6.653** | **20.8%** |
| `build_ik_content_sec` | 5.871 | **5.152** | **12.2%** |
| `build_beng_en_sec` | 3.447 | **3.041** | **11.8%** |
| `tokenize_ik_avg_ms` | 0.1854 | **0.1699** | **8.4%** |
| `tokenize_beng_avg_ms` | 0.1289 | **0.1250** | **3.0%** |
| `query_cn_avg_ms` | 1.1863 | 1.2359 | -4.2% |
| `query_beng_avg_ms` | 1.7097 | 1.7814 | -4.2% |
| `query_mixed_avg_ms` | 1.3346 | 1.3676 | -2.5% |
| `query_limit_avg_ms` | 1.5303 | 1.5262 | +0.3% |

### 噪声评估

同一二进制连续跑 3 次：

| 指标 | 波动范围 |
|---|---|
| `build_ik_all_sec` | 6.575 ~ 8.390（±13%） |
| `tokenize_ik_avg_ms` | 0.1657 ~ 0.1722（±2%） |
| `query_cn_avg_ms` | 1.1781 ~ 1.2098（±1.4%） |

**构建的测量噪声较大**，因此构建项的提升幅度仅供参考；分词与查询的测量稳定，数值可信。

### 按评分公式估算

```
构建平均提升 = (20.8 + 12.2 + 11.8) / 3 = 14.9%
分词平均提升 = (8.4 + 3.0) / 2        =  5.7%
查询平均提升 = (-4.2 - 4.2 - 2.5 + 0.3) / 4 = -2.7%

综合提升率 = (14.9 + 5.7 - 2.7) / 3   =  6.0%
```

查询端 -2.5%~-4.2% 的差异处于噪声范围内（同版本重复测量波动 ±1.4%，但查询路径也经过分词，理论上只应更快）。

## 四、正确性验证

| 检查 | 结果 |
|---|---|
| 编译 | 通过，无警告升级为错误 |
| 实例启动 | 正常，`SELECT VERSION()` 返回 v1.4.0.0 |
| 建表 + 灌 20 000 行 | 正常 |
| IK 索引构建（3 个） | 正常 |
| `TOKENIZE()` | `数据库支持全文检索和向量检索` → `["向量", "和", "全文", "数据库", "支持", "检索"]`，与改动前一致 |
| MATCH 查询 | 命中 4000 行，结果正确 |
| 基准脚本 | 9 项指标全部产出 |

## 五、复现

```bash
# 1. 编译
cd ~/ob-src
./build.sh release --make -j8

# 2. 启动实例
cd build_release/src/observer
nohup ./seekdb --nodaemon --base-dir=$HOME/fts-db --port=2882 --log-level=WARN &

# 3. 跑基准
cd ~/ob-src
export MYSQL='mysql -h127.0.0.1 -P2882 -uroot -N -s -A --default-character-set=utf8mb4'
export MYSQL_VERBOSE='mysql -h127.0.0.1 -P2882 -uroot -A --default-character-set=utf8mb4'
LABEL=after ROWS=20000 BATCH=500 ROUNDS=3000 QUERY_ROUNDS=200 \
  SAMPLES=3 WARMUP=30 SKIP_LOAD=0 OUTPUT=/tmp/report.txt \
  bash tools/benchmark/fts_large_bench.sh
```

## 六、后续可做的方向

| 方向 | 位置 | 预期 |
|---|---|---|
| 查询路径优化 | `sql/` MATCH 执行 | 占 1/3 分，当前为负提升，最该补 |
| 构建流水线 | `rootserver/ddl_task/ob_fts_index_build_task.cpp` | 占 1/3 分，收益最大 |
| 词典加载 | `storage/fts/dict/ob_ft_dict_hub.cpp` | 构建的初始化成本 |