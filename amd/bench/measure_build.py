# ============================================================================
# seekdb × hipVS：单独测量 HNSW 索引构建耗时（CPU VSAG vs GPU CUVS）
#
# 用法：在 AMD Radeon Cloud 的 seekdb-hipvs 模板里，跑完 Notebook 前几节后，
#       新建一个代码单元，整段粘贴运行。
#
# 产出：/seekdb_workspace/results/build_breakdown.csv
#
# 注意（踩过的坑，已在代码里规避）：
#   1. 模板的 sk.sql() 走 `mysql -e "SQL"`，单批行数过大会触发
#      OSError: [Errno 7] Argument list too long —— 所以每批只放 100 行
#   2. sk.sql() 每次新开连接，不带默认库 —— 每条 SQL 前都加 USE
#   3. 索引参数必须写全 DISTANCE / TYPE / HNSW，否则报
#      ERROR 1235: this type of vector index type is not supported
#   4. 模板数据建在 vec 库（不是 test）
# ============================================================================

import csv
import os
import random
import time

DB = "vec"          # 模板里 t_cuvs / t_vsag 所在的库
DIM = 128
BATCH = 100         # 每批行数：太大触发 argv 上限
SIZES = [5000, 10000]   # 想跑更大规模就加 50000 / 100000（会更慢）

os.makedirs("/seekdb_workspace/results", exist_ok=True)


def make_rows(n, seed=7):
    """确定性伪随机向量，保证可复现。"""
    rng = random.Random(seed)
    return [
        (i + 1, "[" + ",".join(f"{rng.uniform(-1, 1):.6f}" for _ in range(DIM)) + "]")
        for i in range(n)
    ]


def ex(q):
    """所有语句带上 USE，规避 No database selected。"""
    return sk.sql(f"USE {DB}; {q}")


def setup_table(name, rows):
    """重建空表并分批灌入，返回成功写入行数。"""
    ex(f"DROP TABLE IF EXISTS {name}")
    rc, out, err = ex(
        f"CREATE TABLE {name} (c1 INT NOT NULL, c2 VECTOR({DIM}), PRIMARY KEY(c1))"
    )
    if rc != 0:
        print(f"  建表失败: {(err or '')[:200]}")
        return 0

    total = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        values = ",".join(f"({vid},'{vec}')" for vid, vec in chunk)
        rc, out, err = ex(f"INSERT INTO {name} (c1, c2) VALUES {values}")
        if rc != 0:
            print(f"  插入失败: {(err or '')[:150]}")
            break
        total += len(chunk)
    return total


def build(name, idx, lib):
    """单独建索引并计时，只计 CREATE VECTOR INDEX 这条语句。"""
    sql = (
        f"CREATE VECTOR INDEX {idx} ON {name}(c2) "
        f"WITH (DISTANCE=L2, TYPE=HNSW, LIB={lib}, M=16, "
        f"EF_CONSTRUCTION=200, EF_SEARCH=64)"
    )
    t0 = time.perf_counter()
    rc, out, err = ex(sql)
    dt = time.perf_counter() - t0
    if rc != 0:
        print(f"  建索引失败({lib}): {(err or '')[:250]}")
        return None
    return dt


print("开始测量 HNSW 索引构建耗时\n")
rows_out = []

for n in SIZES:
    print(f"------ N = {n} ------")
    data = make_rows(n)

    got = setup_table("bench_cpu", data)
    print(f"  [CPU] 灌入 {got} 行")
    cpu = build("bench_cpu", "idx_cpu", "VSAG")
    if cpu:
        print(f"  [CPU] VSAG 建索引: {cpu:.3f}s")
    ex("DROP TABLE IF EXISTS bench_cpu")

    got = setup_table("bench_gpu", data)
    print(f"  [GPU] 灌入 {got} 行")
    gpu = build("bench_gpu", "idx_gpu", "CUVS")
    if gpu:
        print(f"  [GPU] CUVS 建索引: {gpu:.3f}s")
    ex("DROP TABLE IF EXISTS bench_gpu")

    if cpu and gpu:
        rows_out.append(
            {
                "n": n,
                "cpu_sec": round(cpu, 3),
                "gpu_sec": round(gpu, 3),
                "speedup": round(cpu / gpu, 2),
            }
        )
        print(f"  ==> 加速比: {cpu / gpu:.2f}x")
    print()

if rows_out:
    print("=" * 52)
    print(f"{'N':>8} {'CPU(s)':>10} {'GPU(s)':>10} {'加速比':>8}")
    for r in rows_out:
        print(
            f"{r['n']:>8} {r['cpu_sec']:>10.3f} {r['gpu_sec']:>10.3f} {r['speedup']:>7.2f}x"
        )

    path = "/seekdb_workspace/results/build_breakdown.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "cpu_sec", "gpu_sec", "speedup"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n已写入 {path}")
else:
    print("\n两次建索引都失败，请检查库名与索引参数")