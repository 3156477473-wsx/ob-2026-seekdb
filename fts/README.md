# 全文索引调优 · 实验记录

2026 OceanBase 数据库大赛 · seekdb 全文索引性能优化

## 一、编译环境（本机从零搭建）

| 步骤 | 结果 |
|---|---|
| Rust 1.98.1 | ✅ 5 个 target（linux-gnu / windows-gnu / android / ios / ios-sim） |
| WSL 内存 | 7.6 GB → **11 GB** + 16 GB swap（`.wslconfig`） |
| deps/3rd | ✅ `./build.sh init` |
| 编译 | ✅ `./build.sh release --make -j8`，1.3 GB 可执行 |
| 产物 | `build_release/src/observer/seekdb` |

编译耗时：约 20 分钟（16 核 / 11 GB）。

**踩坑记录**（便于复现）：

| 现象 | 原因 | 解法 |
|---|---|---|
| `rustup: could not download nonexistent rust version 1.98.1` | 清华镜像缺该版本 | 换 USTC 镜像 `https://mirrors.ustc.edu.cn/rust-static` |
| `cargo` 下载极慢（57 KB/s），编译卡在 16% | crates.io 直连慢 | `~/.cargo/config.toml` 指向 USTC sparse 索引，3 分钟从 16% → 54% |
| `build.sh: /usr/bin/env: 'bash\r'` | 仓库脚本是 CRLF | `sed -i 's/\r$//'` 全部脚本 |
| `rpm2cpio: command not found` | 缺工具 | `apt install rpm2cpio cpio rpm` |

## 二、基线数据（本机实测）

运行：

```bash
export MYSQL='mysql -h127.0.0.1 -P2882 -uroot -N -s -A --default-character-set=utf8mb4'
export MYSQL_VERBOSE='mysql -h127.0.0.1 -P2882 -uroot -A --default-character-set=utf8mb4'
LABEL=baseline ROWS=20000 BATCH=500 ROUNDS=3000 QUERY_ROUNDS=200 \
  SAMPLES=3 WARMUP=30 SKIP_LOAD=0 OUTPUT=/tmp/fts-baseline-report.txt \
  bash tools/benchmark/fts_large_bench.sh
```

结果（commit `2ad0f37`）：

| 类别 | 指标 | 本机基线 | CI 基线 | 本机/CI |
|---|---|---|---|---|
| 构建 | `build_ik_all_sec` | **8.397** | 35.284 | 0.24× |
| 构建 | `build_ik_content_sec` | **5.871** | 28.376 | 0.21× |
| 构建 | `build_beng_en_sec` | **3.447** | 14.758 | 0.23× |
| 构建 | `build_total_sec` | **17.724** | 78.434 | 0.23× |
| 分词 | `tokenize_ik_avg_ms` | **0.1854** | 0.7648 | 0.24× |
| 分词 | `tokenize_beng_avg_ms` | **0.1289** | 0.4226 | 0.31× |
| 查询 | `query_cn_avg_ms` | **1.1863** | 16.663 | 0.07× |
| 查询 | `query_beng_avg_ms` | **1.7097** | 24.304 | 0.07× |
| 查询 | `query_mixed_avg_ms` | **1.3346** | 17.559 | 0.08× |
| 查询 | `query_limit_avg_ms` | **1.5303** | 16.233 | 0.09× |

**本机比 CI 快 3–14 倍**（CI 核数少）。因此：

- **绝对数值不能直接换算分数**（评分用 CI 固定基线）
- **相对变化（改前 / 改后）是有效指标**

## 三、瓶颈分析

### 3.1 耗时分布

| 类别 | 单项量级 | 备注 |
|---|---|---|
| 构建 | **秒级**（17.7 s 总计） | 最大头 |
| 分词 | 亚毫秒级（0.13–0.19 ms） | 基数小 |
| 查询 | 毫秒级（1.2–1.7 ms） | 中等 |

### 3.2 评分模型

```
单项提升率 = (基线耗时 − 提交耗时) ÷ 基线耗时
综合提升率 = (构建平均 + 分词平均 + 查询平均) ÷ 3
性能得分   = 100 × clamp(综合提升率, 0, 0.5) ÷ 0.5
```

**三类等权**：只优化一类最多得 33 分；三类各提升 30% 可得 60 分。

### 3.3 候选优化点

| 方向 | 位置 | 说明 |
|---|---|---|
| 分词热路径 | `ob_ik_ft_parser.cpp: process_one_char()` | 每字符遍历 segmenter 链表 + 虚函数调用 |
| 分词器初始化 | `ob_ik_ft_parser.cpp: init_dict()` | 每次实例化都加载词典 |
| 构建任务 | `rootserver/ddl_task/ob_fts_index_build_task.cpp` | 构建主流程，耗时的真正来源 |
| 查询路径 | `sql/engine/...` | MATCH 执行 |

## 四、下一轮实验计划

1. 读 `ob_fts_index_build_task.cpp`，定位构建热点
2. 选一个改动点，改代码
3. **增量编译**（只改 1-2 个文件，几分钟）
4. 重启实例，再跑同一基准，对比
5. 保持结果正确（跑回归测试）

## 五、复现命令

```bash
# 启动编译出的实例
cd ~/ob-src/build_release/src/observer
nohup ./seekdb --nodaemon --base-dir=$HOME/fts-db --port=2882 --log-level=WARN &

# 跑基准
cd ~/ob-src
export MYSQL='mysql -h127.0.0.1 -P2882 -uroot -N -s -A --default-character-set=utf8mb4'
LABEL=after ROWS=20000 BATCH=500 ROUNDS=3000 QUERY_ROUNDS=200 \
  SAMPLES=3 WARMUP=30 SKIP_LOAD=0 OUTPUT=/tmp/fts-after.txt \
  bash tools/benchmark/fts_large_bench.sh
```