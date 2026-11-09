# GPU graph quality recovery

The user reports that the previously delivered approximately2.6x public build
candidate failed the official `hidden250k:recall@100_baseline_delta` gate.
That invalidates its use as a qualified submission. Candidate and baseline
Recall values were not supplied. Public SIFT/Gaussian passes do not establish
Hidden250K quality, and no hidden data is available or inferred here.

Source review did not identify a confirmed defect in GPU detour counting,
reverse-edge generation, or merge semantics. The important algorithm difference
is the intermediate graph: the frozen baseline uses GPU NN-Descent and256
intermediate edges, while the failed candidate used GPU IVF-PQ and192 edges.
This change is a quality-recovery experiment, not a proven diagnosis or fix.

## Configuration

| Parameter | Failed candidate | Recovery candidate |
|---|---:|---:|
| Probed IVF lists (capped at actual list count) |32|64|
| Intermediate neighbors |192|256|
| Refinement factor |4|4|
| Maximum candidates before exact FP32 refinement |768|1024|
| Outer query batch |1024|1024|
| SDK maximum internal query batch |512|256|
| Final CAGRA degree |128|128|

The higher probe count broadens candidate coverage. The larger intermediate
graph gives unchanged GPU pruning more neighbors from which to select the final
graph. Both properties need Recall measurement; neither guarantees the gate.
The internal batch is reduced to limit concurrent IVF search workspace while
the complete intermediate graph is staged as lossless bytes on the host.
There is no CPU graph construction or CPU distance computation.

At250K rows, the packed intermediate graph is192,000,000 bytes. A1024-row packed
output batch is786,432 bytes. These are allocation calculations, not measured
peak VRAM. Increasing probes and candidate count can increase both runtime and
memory. The physical baseline-relative memory gate must be measured again.

Bridge and GPU translation units share `SEEKDB_IVFPQ_INTERMEDIATE=256` in the
build command and header default. Their common minimum custom-build size is258:
256 and257 rows use native construction;258 rows can provide256 non-self
neighbors plus the extra refinement result. The CPU fake exercises the actual
bridge dispatcher at these boundaries and its ownership failure paths. It does
not run HIP or establish ANN quality. The explicit CPU-test macro only enables
the fake C harness; real HIP C++ builds retain SDK-entry serialization.

## Preserved behavior and required validation

Original FP32 vectors, IDs, native full-file snapshots, graph-pruning semantics,
and real fixed-library `cuvsCagraSearch` results are preserved. Search width4 and
`itopk_size=max(512,4*topk)` are unchanged. hipVS/cuVS itself is not modified.
Pinned staging transfers complete IDs on the same stream, releases the search
pool, then uploads the complete graph before GPU pruning and index publication.

Before delivery, link with the fixed SDK on AMD, then compare against the frozen
baseline on the same instance: full250K SQL construction plus first query,
4096 Top10/100 queries, exact ID integrity, Recall deltas, physical VRAM,
normal reopen, and the existing `lib=vsag` path. Keep timings and all failed
results. An object cross-compile alone is not a full build, GPU runtime test,
quality pass, or performance score. Official Hidden250K acceptance remains
dependent on the organizer's evaluation.
