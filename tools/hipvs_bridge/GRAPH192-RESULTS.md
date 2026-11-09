# CAGRA intermediate graph degree 192

## Controlled-pair correction

The original 2.901x screen below did not control the reference process's OpenMP
settings and must not be attributed to graph degree alone. The reference's
actual runtime thread count was not captured. A later probe found the notebook's
exported OMP_NUM_THREADS unset, despite a shell variable displaying 1; the SDK
runtime defaulted to 128 threads in that probe. This does not establish the old
process's exact runtime count. The statement that it inherited OMP_NUM_THREADS=1
is withdrawn. Historical raw records are retained.

Both processes in these replacement pairs explicitly used OMP_NUM_THREADS=16,
OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, 16-CPU quota and 8 GiB seekdb memory:

| Controlled measurement | 128 dimensions | 256 dimensions |
| --- | ---: | ---: |
| f677 DDL plus first legal query | 20.905244 s | 26.216285 s |
| graph192 DDL plus first legal query | 17.903806 s | 20.189203 s |
| Build speedup relative to f677 | 1.167642x | 1.298530x |
| Top-10 QPS ratio | 1.096072x | 1.028505x |
| Top-100 QPS ratio | 1.036249x | 1.044284x |
| Recall@10 delta | +0.001025 | +0.000171 |
| Recall@100 delta | +0.002695 | +0.003420 |

Both pairs used 250K seeded Gaussian FP32 vectors, 4096 queries per K, the same
SQL settings, fresh database directories, and exhaustive float64 ground truth
outside timing. Graph192 was 5f14b7d9df2cfcdd9f0e80994aa0b2dc1d1cf148. These
measurements do not test the later CPU-budget policy. IDs/counts/uniqueness and
eight ordered-result statements after normal reopen passed. The 256-D driver's
final stop monitor encountered an exiting-process cmdline race; its original
failure is retained, and a separate read-only reconciliation verified all its
recorded processes absent or zombies. The corrected 128-D driver completed all
normal stops. Neither pair is official SIFT/Hidden or an official score.

This change builds on accepted candidate
`f677a5d8ea2ea49aa8cdc354dfe8048e3fb50f16`, retaining its device-input lifetime
and persistent query worker improvements. It changes only the CAGRA
intermediate graph degree from 256 to 192. Final graph degree 128, NN-descent
iteration cap 100, FP32/L2 input, original IDs and search parameters are retained.
The fixed hipVS library is not modified.

The smaller intermediate graph reduces NN-descent host graph maintenance and
CAGRA sorting/pruning work. This is a static construction parameter, applied to
every build; there is no dataset, benchmark-phase or environment selector.

## Historical parameter screen, thread conditions not controlled

Environment: AMD gfx1100, ROCm 7.2.4, 16 CPU quota, seekdb memory budget 8 GiB.
Both runs used the same full seekdb build and independently compiled bridge,
`-O2 -gline-tables-only -DNDEBUG`, identical SQL/index options, and fresh isolated
database directories. The main binary SHA256 was
`bdd64abacff6045b566fa21bffacda89696e013f2602814f79670c5e70d99d08`.

Input: reproducible NumPy PCG64 seed 20261029, 250,000 FP32 Gaussian vectors of
dimension 128 and 4,096 independent queries. Ground truth is exhaustive CPU
float64 L2, outside timings. This is **synthetic development data**, not the
official SIFT250K or Hidden250K data. Each Top-10/100 group runs 4,096 separate
SQL queries; no batch-query shortcut is used.

| Measurement | Accepted f677 | Intermediate degree 192 |
| --- | ---: | ---: |
| Offline CREATE VECTOR INDEX plus first legal query | 49.809 s | 17.171 s |
| Build speedup relative to f677 | 1.000x | 2.901x |
| Top-10 QPS relative to f677 | 1.000x | 1.245x |
| Top-100 QPS relative to f677 | 1.000x | 1.230x |
| Recall@10 change | reference 0.990601 | +0.001147 |
| Recall@100 change | reference 0.966926 | +0.002876 |

All imported original IDs were verified, and all queries returned the requested
number of distinct IDs from that set. After normal stop/reopen, four fixed
queries at each K returned the same ordered original IDs. This sampled reopen
check does not mean full Recall was measured again after restart.

Raw timing/ID/Recall arrays, instance identities, bridge source/binary hashes and
GPU utilization samples are retained in the owned development experiment.
Construction uses the same SQL options:
`distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64`.

The earlier f677 OJ result was user-reported as 7.817, build=1.153x,
query=1.779x, gates=pass. It is a result for f677, not this change. Synthetic
speedups cannot be multiplied into a promised official score. Official paired
SIFT/Hidden evaluation and its Recall, GPU-path, throughput and memory gates
remain authoritative. Public SIFT and repeated measurements are still pending.

Two other screened choices were rejected: final/intermediate degrees 64/128,
and the unchanged graph with NN-descent cap 20. Both ran faster on these inputs
but lost more than 0.005 Recall relative to f677; neither is included here.
