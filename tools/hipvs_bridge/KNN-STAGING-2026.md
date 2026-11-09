# Batched kNN staging candidate

Status: implementation candidate. CPU byte-layout checks passed. HIP object
compilation is recorded separately for each immutable source version; actual GPU
peak, Recall and end-to-end performance remain unverified for the current
quality-recovery configuration. Do not treat allocation arithmetic or object
compilation as a passed runtime gate. See `RECALL-RECOVERY-2026.md`.

## Motivation

The prior full-FP32 pooled-search build completed public SIFT/Gaussian SQL tests
with build speedups about3.40x/2.65x but exceeded the frozen-relative physical
VRAM limit. Its complete intermediate kNN graph overlapped the search pool.
At250,000 rows and192 intermediate edges, lossless u24 storage uses144,000,000
bytes. Refinement writes one batch at a time; it does not read earlier rows.

## Change

Keep one1024-row output batch on the GPU. The initial192-edge configuration used
589,824 bytes for u24 IDs; the current256-edge recovery uses786,432 bytes. After GPU
FP32 refinement, copy the output to pinned host storage on the same HIP stream.
Packed storage is a low16 plane followed by a high8 plane: copy each plane
separately into its position in the complete host matrix, including a partial
last batch. The uint32 fallback uses an ordinary contiguous slice.

After all batch copies complete, synchronize and destroy pooled-search resources.
Then allocate and upload the complete kNN graph for the existing GPU pruning and
reverse-edge kernels. Release the pinned host buffer after that upload completes.
The complete GPU kNN allocation and the pooled-search lifetime no longer overlap.

Host staging performs byte transport and address calculation only. IVF-PQ,
FP32 distance refinement, graph pruning, reverse-edge generation, and final
CAGRA queries remain GPU operations. Original vectors, full ID bits, candidate
counts, probes, refinement factor, native snapshot format and fixed SDK queries
are preserved by the staging mechanism itself. The separate quality-recovery
configuration increases intermediate edges and probes; it is not a claim that
those parameters equal the prior submission. No data is omitted. All DMA and buffer allocation occur inside
the normal timed index-construction path, before CREATE INDEX returns.

The final validation path remains the fixed library's `cuvsCagraSearch` and its
actual returned IDs. This patch does not modify or replace hipVS/cuVS.

## Lifetime and failure checks

- The host buffer is declared before StreamFence, so the fence executes before
  host destruction on return or exception. The scoped search pool also syncs
  before destroying its fresh resource and resetting the process memory resource.
- Same-stream order serializes both D2H slices before the next refinement reuses
  the device batch buffer.
- Full graph upload starts only after successful batch/error validation and pool
  teardown. Its event synchronization finishes before freeing the host source.
- Invalid shapes, allocation/copy errors and non-finite/invalid candidate results
  still fail construction; no CPU computation or substituted answers are used.
- The build's existing SDK-entry mutex is retained around all bridge operations
  that can overlap the process-wide RMM resource change.

## Validation

`tests/test_knn_staging.cpp` runs without HIP. It independently encodes batches,
uses the same copy-slice helper as the product, then decodes the complete matrix.
It covers both storage formats, high ID bytes, sentinels, partial final batches,
bounds/overflow guards and exact250K x192 and250K x256 packed matrices. This tests byte
transport only, not HIP kernels, SQL, Recall, physical VRAM or performance.

Required next GPU run: compile the bridge and seekdb version metadata; run exact
FP32/full-file snapshot roundtrip; run full250K SQL construction+first query,
4096 Top10/100 queries and normal reopen; compare same-device physical VRAM and
Recall against the frozen binary. Confirm the fixed CAGRA query implementation
and CPU lib=vsag functionality remain intact. Publish only after actual acceptance.

## References

- [HIP7.2 pinned host memory](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/memory_management/host_memory.html)
- [Frozen hipVS C resource and pool API](https://github.com/AMD-Ecosystem/hipVS/blob/f96a663d0a4441d6f5344bd3ef552b71ae7b370b/cpp/include/cuvs/core/c_api.h)
