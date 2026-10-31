# Search-quality reserve candidate

This candidate changes only the minimum `itopk_size` supplied to the fixed
library's actual `cuvsCagraSearch`:512 to640, retaining the existing `4*topk`
lower bound and search_width4. No alternative search, reranking of returned
results, CPU fallback, or precomputed answers are introduced. Original vectors,
external IDs, SQL inputs and index-construction parameters are unchanged.

The larger search budget is intended to provide Recall margin after the prior
official Hidden250K delta failure. It can reduce query throughput and must be
compared against both the512-budget parent and the frozen baseline. It is not
assumed to fix Hidden250K, retain the90% throughput floor, or reach a score.
The formal query result must remain the actual fixed CAGRA library output.

At commit time the512 parent is still completing its paired baseline evaluation.
This640 candidate has no claimed GPU runtime or performance pass. Build, run
the real GPU search path, then perform full same-device SQL Recall, throughput,
VRAM, persistence and CPU-path checks before delivery. Keep the parent and any
negative results intact.

Fixedf96 `cpp/include/cuvs/neighbors/cagra.hpp` describes itopk_size as the
intermediate search-result budget and a quality/speed tradeoff. A source
declaration alone does not establish that a particular installed build accepts
or benefits from640; actual runtime checks remain required.
