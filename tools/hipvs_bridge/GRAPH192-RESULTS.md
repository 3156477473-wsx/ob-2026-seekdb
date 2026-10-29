# CAGRA intermediate graph degree 192

This change builds on accepted candidate
`f677a5d8ea2ea49aa8cdc354dfe8048e3fb50f16`, retaining its device-input lifetime
and persistent query worker improvements. It changes only the CAGRA
intermediate graph degree from 256 to 192. Final graph degree 128, NN-descent
iteration cap 100, FP32/L2 input, original IDs and search parameters are retained.
The fixed hipVS library is not modified.

The smaller intermediate graph reduces NN-descent host graph maintenance and
CAGRA sorting/pruning work. This is a static construction parameter, applied to
every build; there is no dataset, benchmark-phase or environment selector.

## Real AMD development measurement

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
