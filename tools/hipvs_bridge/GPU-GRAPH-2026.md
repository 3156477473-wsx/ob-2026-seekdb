# Bounded-memory GPU graph construction

This implementation descends from the frozen AMD V5 baseline. It changes graph
construction inside this repository; the installed hipVS/cuVS libraries remain
unchanged. All queries still call the installed `cuvsCagraSearch` and return its
actual results.

## Construction

- Build an IVF-PQ candidate generator through the public C API, using FP32
  coarse-search, lookup-table, and accumulated-distance types. In particular,
  high-norm FP32 inputs must not overflow a half-precision candidate search.
- Probe 32 lists (bounded by the actual number of lists), retrieve four times
  the intermediate degree, and recompute candidate squared-L2 distances in a
  repository-owned GPU kernel. Retain 192 valid unique non-self neighbors.
- Apply GPU detour pruning, CSR reverse-edge generation, and merging to obtain
  a 128-neighbor graph. Reject invalid IDs, non-finite distances, incomplete
  rows, self edges, and duplicates; failures are not accepted as a valid graph.
- Import the resulting graph and original FP32 dataset with the public
  `cuvsCagraIndexFromArgs`. The bridge owns borrowed device buffers until after
  index destruction. Small collections retain the native GPU builder.

The outer dispatch batch is 1024, while the SDK's independent internal batch is
128. This amortizes C API dispatch without increasing the SDK scratch batch.
The candidate-refinement kernel uses 8 KiB shared memory per block instead of
materializing candidate-by-dimension vectors. Per-batch diagnostic events are
disabled; phase timings, stream synchronization, and error checks remain.

## Persistence and compatibility

The existing V3 snapshot format, original-ID mapping, FP32 dataset recovery,
legacy V2 reader, and fixed CAGRA native serialization are retained. This patch
does not replace the `lib=vsag` CPU implementation. The production build entry
remains `bash tools/hipvs_bridge/build.sh [OUTPUT_SO]` and needs only the fixed
SDK's public C headers and the installed ROCm toolchain, without new downloads.

The build script explicitly selects the AMD platform and `gfx1100`, the
competition's Radeon PRO W7900D target. Compiling on a host without a visible GPU
does not change this target. For development on another AMD architecture, set
`SEEKDB_HIP_ARCH` to its HIP target ID. The same target is passed to both source
compilations and the final link; setting it only at link time cannot repair an
object already compiled for another GPU.

## Verification and limits

Development receipts and reproducible drivers are saved outside the repository
under `build-artifacts/competition-amd-native-20261030-r1`. Experiments include
public SIFT and a public seeded Gaussian corpus, not the private Hidden data.
Component screens do not constitute a SQL or OJ score. Final delivery receipts
identify the exact source commit, native ELF, bridge, query counts, quality,
VRAM sampling, and normal-stop/reopen checks separately.

CAGRA is approximate: independently executed queries can select different valid
neighbors, including ties. Reopen validation records exact-ID differences but
must also check the original table IDs, unique valid TopK results, native GPU
reload evidence, and explicit absolute/relative Recall requirements. It must
not label non-identical ANN results as identical or skip the quality checks.
