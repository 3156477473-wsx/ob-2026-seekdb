# AMD V5 candidate — 2026-10-29

Base: `634888d9b037153fa143d2c913ba4b25503e1d62`, branch `amd-oceanbase-official-contest-base-v5`, tag `amd-cagra-contest-baseline-v5` in `zihaomu/seekdb`. Fixed hipVS: `f96a663d0a4441d6f5344bd3ef552b71ae7b370b`. This candidate replaces the old non-V5 submission; it does not import the old GPU-to-CPU HNSW conversion.

## Changes

1. **Device-resident build input.** For FP32 row dimensions divisible by four, the bridge transfers the real input once and gives the existing CAGRA builder a device DLPack tensor. The index borrows this aligned buffer; the bridge retains it until after stream synchronization and index destruction. Unaligned dimensions retain the original host path. This avoids repeated host-to-device input transfers; CAGRA's internal graph-sort copy still occurs. Deserialization retains the library-owned dataset with no extra bridge allocation.
2. **Persistent large-stack query dispatcher.** A synchronous 32MiB worker replaces the per-SQL pthread creation/join in the existing single-query adaptor. Each SQL still executes its own unchanged GPU job and waits for completion under the existing registry lock. No queries are combined, results cached or work moved outside the CREATE-to-first-query interval. Cancellation cannot release the caller's stack before the GPU job returns. Worker creation failure is retryable, and normal process shutdown joins the worker.

FP32/L2 inputs, original ID mapping, NN_DESCENT/graph-degree128/intermediate-degree256/max-iterations100, search parameters, GPU snapshot format and CPU `lib=vsag` behavior remain unchanged. No external hipVS code, evaluator SQL/data, build commands, dependency manifests or baseline metrics are changed.

## Local evidence and limits

Linux host tests passed for 4096 synchronous calls, concurrent callers, worker reuse, creation failure, TLS cleanup, recursive dispatch rejection, fork rejection and cancellation. C11 and C++17 fake-backend tests passed for aligned/unaligned input, input ownership, allocation/copy/create/build failure cleanup, serialization lifetime and deserialization without double-free. The real changed adaptor translation unit passed Clang17 syntax checking against existing local development headers with `OB_BUILD_CUVS` enabled.

**These are host logic checks, not a full frozen V5 build or GPU validation.** The local machine lacks ROCm/hipVS SDK and the organizer-frozen vendor set; the declaration stubs intentionally do not establish SDK ABI. No cloud instance was started. Real HIP compilation/linkage, GPU CAGRA execution/reload, Recall, QPS, peak memory and performance scores remain unverified. No speedup or score is claimed.

The device input is 128,000,000 bytes at 250K×128 and 256,000,000 at 250K×256. It replaces the final library dataset for the aligned path but is resident earlier during building. Actual peak VRAM must still satisfy both the baseline×1.25 and 45GB limits; host tests cannot establish either limit.

## Reproduction

Host-only checks use GCC/G++ and pthreads, require no network, database or GPU, and keep binaries in the chosen output directory:

```bash
make -C tools/hipvs_bridge/tests -j8 check OUT="$PWD/build_host_tests"
```

The fake headers are confined to this standalone test Makefile. The production bridge build continues using the real fixed SDK and the original entry point:

```bash
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh "$PWD/build_gpu/libseekdb_cuvs_bridge.so"
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB="$PWD/build_gpu/libseekdb_cuvs_bridge.so"
cmake --build build_release --target seekdb
```

For a real GPU comparison, run frozen V5 and this committed candidate with fresh indexes on the same machine and inputs. Use fixed `CREATE VECTOR INDEX ... (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64)`; include the first legal query in build time. Preserve GPU profiling, original/unique IDs, Top10/100 Recall and 4096 single-SQL-query QPS, resource peaks, normal reopen and CPU `lib=vsag` results. No prior graph or precomputed results may be reused. The existing validation scripts in this directory provide lifecycle smoke checks; they do not reproduce the hidden official dataset or establish its score.
