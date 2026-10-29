# Resource-bounded SDK OpenMP policy

The fixed SDK's host graph-maintenance loops use LLVM OpenMP. Its NN-descent
also creates native std::threads, which do not inherit another thread's
omp_set_num_threads task setting. An independent native probe observed this
distinction. A task-local setting alone cannot budget all workers.

Before the first seekdb cuVS SDK entry, one process-lifetime policy resolves
the already-loaded bridge's OpenMP dependency. Getter, setter, C defaults
entry and fork entry must belong to the same libomp object, including the
global fork binding. The policy does not load or modify the fixed SDK and
does not alter environment variables or system settings.

The positive minimum of visible cgroup CPU quotas, configured seekdb CPU
count and the calling worker's affinity bounds the thread budget. Unknown
limits leave the policy unchanged. Cgroup v1/v2 ancestors are checked;
existing memory-limit reading behavior is preserved. The isolated new header
does not change the widely included utility.h.

The verified C ABI kmpc_set_defaults sets OMP_NUM_THREADS once for future SDK
native threads. Existing tasks are scoped to the same budget and restored
after each SDK call. Build, search, serialization, deserialization and free
entries pass the same first-use barrier. If optional defaults are unavailable,
only the verified task-local setting is used. Cached runtime references remain
alive through registry teardown.

This first-use process policy does not track later CPU resource changes; a
restart refreshes its captured budget. Its effect on queries and real GPU
construction still requires product measurements. Patch application and the
independent runtime probe are not product acceptance. No blocktime setting
is changed in this candidate.
