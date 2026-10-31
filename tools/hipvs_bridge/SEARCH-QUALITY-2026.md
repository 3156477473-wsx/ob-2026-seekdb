# Search-quality reserve candidate

The earlier unrun640-only candidate is not valid under the fixed C API default:
its default is SINGLE_CTA, and that mode rejects budgets above512. This
successor explicitly requests the fixed library's native MULTI_CTA mode and a
minimum `itopk_size` of640, retaining the existing `4*topk` lower bound and
public search_width4. It is a SINGLE512 to MULTI640 mode/budget change, not an
assumption that the old code already selected MULTI. No alternative search, reranking of returned
results, CPU fallback, or precomputed answers are introduced. Original vectors,
external IDs, SQL inputs and index-construction parameters are unchanged.

The parallel native search and larger global result budget are intended to provide Recall margin after the prior
official Hidden250K delta failure. It can reduce query throughput and must be
compared against both the512-budget parent and the frozen baseline. It is not
assumed to fix Hidden250K, retain the90% throughput floor, or reach a score.
The formal query result must remain the actual fixed CAGRA library output.
The configured policy is logged once per opaque bridge handle after its first
successful query, rather than logging every query in a throughput loop.

In the pinned implementation, MULTI_CTA uses one hardware wave per local
candidate list and `max(search_width,ceil(itopk/warp_size))` CTAs per query.
Thus640 means20 CTAs for a32-lane wave or10 for64 lanes; it does not mean640
entries per CTA. The fixed SDK handles traversal, merging and actual top-k.

At commit time the512 parent is still completing its paired baseline evaluation.
This640 candidate has no claimed GPU runtime or performance pass. Build, run
the real GPU search path, then perform full same-device SQL Recall, throughput,
VRAM, persistence and CPU-path checks before delivery. Keep the parent and any
negative results intact.

Fixedf96 `cpp/include/cuvs/neighbors/cagra.hpp` describes itopk_size as the
intermediate search-result budget and a quality/speed tradeoff. A source
declaration alone does not establish that a particular installed build accepts
or benefits from640; actual runtime checks remain required.
