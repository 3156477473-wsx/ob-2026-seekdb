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
| `member initializers order` 编译错误 | 初始化列表顺序与声明顺序不符 | 按声明顺序排列 |

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
- 多次调用结果稳定（连续 3 次输出完全相同）
- 建表 + 索引 + MATCH 查询全部正常

## 三、实测效果

### 3.1 测量噪声评估（关键）

**同一份二进制连续运行，构建指标的波动非常大：**

| 时段 | `build_ik_all_sec` | 说明 |
|---|---|---|
| 昨日 23:00 | 8.397 | 基线 |
| 昨日 23:10 | 6.653 | 优化后，机器空闲 |
| 今日 12:10 | 9.494 | 优化后，机器有其他负载 |

**同一份代码的构建指标波动达 43%（6.653 ~ 9.494）**，因此构建项的对比不可靠。

**分词指标则稳定**（±2%）：

| 时段 | `tokenize_ik_avg_ms` |
|---|---|
| 昨日 23:07 | 0.1722 |
| 昨日 23:08 | 0.1675 |
| 昨日 23:08 | 0.1657 |
| 今日 12:04 | 0.1758 |
| 今日 12:05 | 0.1707 |
| 今日 12:05 | 0.1712 |

### 3.2 分词提升（可信结论）

| 指标 | 基线 | 优化后（今晚） | 优化后（次日） | 提升 |
|---|---|---|---|---|
| `tokenize_ik_avg_ms` | 0.1854 | **0.1699** | **0.1712** | **7.7% ~ 8.4%** |
| `tokenize_beng_avg_ms` | 0.1289 | **0.1250** | **0.1321** | 0 ~ 3.0% |

**两种不同负载下都稳定在 -7.7% ~ -8.4%，这是本优化最可靠的证据。**

### 3.3 其他指标（仅供参考，噪声大）

| 指标 | 基线 | 优化后（机闲） | 优化后（机忙） |
|---|---|---|---|
| `build_ik_all_sec` | 8.397 | 6.653（-20.8%） | 9.494（+13.1%） |
| `build_ik_content_sec` | 5.871 | 5.152（-12.2%） | 6.605（+12.5%） |
| `build_beng_en_sec` | 3.447 | 3.041（-11.8%） | 4.666（+35.4%） |
| `query_cn_avg_ms` | 1.1863 | 1.2359（-4.2%） | 1.2272（-3.4%） |
| `query_beng_avg_ms` | 1.7097 | 1.7814（-4.2%） | 1.7676（-3.4%） |
| `query_mixed_avg_ms` | 1.3346 | 1.3676（-2.5%） | 1.3909（-4.2%） |
| `query_limit_avg_ms` | 1.5303 | 1.5262（+0.3%） | 1.5238（+0.4%） |

## 四、失败的尝试（已回滚）

### 尝试：线程内复用 IK parser

**思路**：`ObIKFTParserDesc::segment()` 每行文档都创建并销毁一个 `ObIKFTParser`，其中 `init_dict()` 会重新加载词典。若在 `thread_local` 里缓存 parser、只重建 `ctx_`，可省掉重复初始化。

**结果：段错误（Segmentation fault）。**

原因：

- `ObIKFTParser` 由 `param->allocator_` 分配，而 `param` 的生命周期只到单次 `segment()` 调用结束
- `free_token_iter()` 改为不释放后，下一次调用拿到的是指向已失效 allocator 的对象
- 表现为 `ERROR 4006: The object is not initialized` 后进程崩溃

**已完全回滚**，代码回到仅含优化 1 的状态。记录在此，说明此方向若要推进，必须解决 allocator 生命周期（例如改用全局或 session 级分配器，而非每次调用的临时 allocator）。

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

| 方向 | 位置 | 说明 |
|---|---|---|
| `init_segmenter()` 批量分配 | `ob_ik_ft_parser.cpp` | 4 次 `OB_NEWx` 合并为一次，收益小 |
| 句柄池化 | `ob_ft_cache_container.h: fetch_info_for_dict()` | 每 range 一次分配 + 链表插入 |
| 词典 range 并行加载 | `ob_ft_range_dict.cpp: try_load_cache()` | range 数多时收益明显，改动大 |
| 查询路径 | `sql/` MATCH 执行 | 当前为负提升，需先定位真实热点 |
| parser 复用 | `ob_ik_ft_parser_desc::segment()` | **需先解决 allocator 生命周期**，见第四节 |