/* Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
 * GPU graph construction using the unchanged public hipVS C API.
 * Detour pruning and reverse-edge policy follow the documented fixed CAGRA
 * construction, with all graph transforms performed on the GPU.
 * No SDK C++ types, private symbols, global allocator or search replacement. */
#include "ivfpq_graph.h"
#include <cuvs/neighbors/ivf_pq.h>
#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#ifndef SEEKDB_IVFPQ_BATCH_EVENTS
#define SEEKDB_IVFPQ_BATCH_EVENTS 0
#endif
#ifndef SEEKDB_IVFPQ_MEMORY_INFO
#define SEEKDB_IVFPQ_MEMORY_INFO 1
#endif
#if SEEKDB_IVFPQ_BATCH_EVENTS
#include <array>
#include <vector>
#endif

#ifndef SEEKDB_IVFPQ_PROBES
#define SEEKDB_IVFPQ_PROBES 32
#endif
#ifndef SEEKDB_IVFPQ_BATCH
#define SEEKDB_IVFPQ_BATCH 1024
#endif
#ifndef SEEKDB_IVFPQ_INTERNAL_BATCH
#define SEEKDB_IVFPQ_INTERNAL_BATCH 128
#endif
#ifndef SEEKDB_IVFPQ_INTERMEDIATE
#define SEEKDB_IVFPQ_INTERMEDIATE 192
#endif
#ifndef SEEKDB_IVFPQ_REFINEMENT
#define SEEKDB_IVFPQ_REFINEMENT 4
#endif
namespace {
constexpr unsigned degree = 128;
constexpr unsigned input_degree = SEEKDB_IVFPQ_INTERMEDIATE;
constexpr unsigned hash_size = 1024;
constexpr unsigned candidate_capacity = input_degree * SEEKDB_IVFPQ_REFINEMENT <= 1024 ? 1024 : 2048;
static_assert(input_degree > degree && input_degree <= 256, "supported input degree");
static_assert(SEEKDB_IVFPQ_BATCH > 0, "positive batch");
static_assert(SEEKDB_IVFPQ_INTERNAL_BATCH > 0 && SEEKDB_IVFPQ_INTERNAL_BATCH <= SEEKDB_IVFPQ_BATCH,
              "positive internal batch no larger than the input batch");
static_assert(input_degree * SEEKDB_IVFPQ_REFINEMENT <= candidate_capacity, "shared candidate sort capacity");
using Clock = std::chrono::steady_clock;

struct DeviceMemory {
  void* pointer = nullptr;
  ~DeviceMemory() { if (pointer) hipFree(pointer); }
  bool allocate(size_t bytes) { return hipMalloc(&pointer, bytes) == hipSuccess; }
  void reset() { if (pointer) hipFree(pointer); pointer = nullptr; }
  template<class T> T* as() const { return static_cast<T*>(pointer); }
  unsigned* release_graph() { unsigned* p = as<unsigned>(); pointer = nullptr; return p; }
};
struct IvfObjects {
  cuvsIvfPqIndex_t index = nullptr;
  cuvsIvfPqIndexParams_t build = nullptr;
  cuvsIvfPqSearchParams_t search = nullptr;
  ~IvfObjects() { reset(); }
  void reset() {
    if (index) cuvsIvfPqIndexDestroy(index);
    if (search) cuvsIvfPqSearchParamsDestroy(search);
    if (build) cuvsIvfPqIndexParamsDestroy(build);
    index = nullptr; search = nullptr; build = nullptr;
  }
};
struct Timer {
  hipEvent_t begin = nullptr, end = nullptr;
  Clock::time_point wall;
#if SEEKDB_IVFPQ_MEMORY_INFO
  size_t memory_free_before = 0, memory_total_before = 0;
  bool memory_valid = false;
#endif
  ~Timer() { if (begin) hipEventDestroy(begin); if (end) hipEventDestroy(end); }
  bool start(hipStream_t stream) {
    wall = Clock::now();
#if SEEKDB_IVFPQ_MEMORY_INFO
    memory_valid = hipMemGetInfo(&memory_free_before, &memory_total_before) == hipSuccess;
#endif
    if (!begin && hipEventCreate(&begin) != hipSuccess) return false;
    if (!end && hipEventCreate(&end) != hipSuccess) return false;
    return hipEventRecord(begin, stream) == hipSuccess;
  }
  bool report(const char* name, hipStream_t stream) {
    float ms = 0;
    if (hipEventRecord(end, stream) != hipSuccess || hipEventSynchronize(end) != hipSuccess ||
        hipEventElapsedTime(&ms, begin, end) != hipSuccess) return false;
    const double seconds = std::chrono::duration<double>(Clock::now() - wall).count();
    fprintf(stderr, "[seekdb_cuvs] ivfpq_gpu_stage=%s gpu_ms=%.3f wall_s=%.6f\n", name, ms, seconds);
#if SEEKDB_IVFPQ_MEMORY_INFO
    size_t memory_free_after = 0, memory_total_after = 0;
    if (memory_valid && hipMemGetInfo(&memory_free_after, &memory_total_after) == hipSuccess) {
      fprintf(stderr, "[seekdb_cuvs] ivfpq_gpu_memory_stage=%s used_before=%zu used_after=%zu total_before=%zu total_after=%zu\n",
              name, memory_total_before - memory_free_before, memory_total_after - memory_free_after,
              memory_total_before, memory_total_after);
    }
#endif
    return true;
  }
};
struct StreamFence {
  hipStream_t stream;
  ~StreamFence() { hipStreamSynchronize(stream); }
};
#if SEEKDB_IVFPQ_BATCH_EVENTS
struct BatchEvents {
  std::vector<std::array<hipEvent_t, 3>> values;
  ~BatchEvents() { for (auto& row : values) for (auto event : row) if (event) hipEventDestroy(event); }
  bool create(size_t batches) {
    values.resize(batches);
    for (auto& row : values) for (auto& event : row) if (hipEventCreate(&event) != hipSuccess) return false;
    return true;
  }
  bool report() {
    float search_ms = 0, refine_ms = 0;
    for (auto& row : values) {
      float first = 0, second = 0;
      if (hipEventElapsedTime(&first, row[0], row[1]) != hipSuccess ||
          hipEventElapsedTime(&second, row[1], row[2]) != hipSuccess) return false;
      search_ms += first; refine_ms += second;
    }
    fprintf(stderr, "[seekdb_cuvs] ivfpq_gpu_search_ms=%.3f ivfpq_gpu_refine_ms=%.3f batches=%zu\n",
            search_ms, refine_ms, values.size());
    return true;
  }
};
#endif
DLManagedTensor tensor(void* data, int64_t* shape, uint8_t code, uint8_t bits, int device) {
  DLManagedTensor value{};
  value.dl_tensor.data = data;
  value.dl_tensor.device.device_type = kDLCUDA;
  value.dl_tensor.device.device_id = device;
  value.dl_tensor.ndim = 2;
  value.dl_tensor.dtype = {code, bits, 1};
  value.dl_tensor.shape = shape;
  return value;
}
__device__ unsigned hash_id(unsigned id) { return (id * 2654435761u) & (hash_size - 1); }

#include "candidate_refine_gpu.hip.inc"

/* Same detour count and (saturating count, original rank) order as fixed
 * graph_core.cuh:187-245,1411-1458. A per-row hash replaces the innermost
 * linear target lookup, reducing O(D^3) comparisons to expected O(D^2). */
__global__ void prune_graph(const unsigned* graph, unsigned total, unsigned* output) {
  __shared__ unsigned keys[hash_size];
  __shared__ unsigned ranks[hash_size];
  __shared__ unsigned counts[input_degree];
  const unsigned row = blockIdx.x;
  for (unsigned k = threadIdx.x; k < hash_size; k += blockDim.x) keys[k] = UINT32_MAX;
  if (threadIdx.x < input_degree) counts[threadIdx.x] = 0;
  __syncthreads();
  const unsigned* neighbors = graph + size_t(row) * input_degree;
  if (threadIdx.x < input_degree) {
    const unsigned id = neighbors[threadIdx.x];
    unsigned slot = hash_id(id);
    while (atomicCAS(keys + slot, UINT32_MAX, id) != UINT32_MAX) slot = (slot + 1) & (hash_size - 1);
    ranks[slot] = threadIdx.x;
  }
  __syncthreads();
  for (unsigned pair = threadIdx.x; pair < (input_degree - 1) * input_degree; pair += blockDim.x) {
    const unsigned ad = pair / input_degree;
    const unsigned db = pair % input_degree;
    const unsigned middle = neighbors[ad];
    const unsigned target = graph[size_t(middle) * input_degree + db];
    unsigned slot = hash_id(target);
    while (keys[slot] != UINT32_MAX && keys[slot] != target) slot = (slot + 1) & (hash_size - 1);
    if (keys[slot] == target && ranks[slot] > ad) atomicAdd(counts + ranks[slot], 1u);
  }
  __syncthreads();
  if (threadIdx.x < input_degree) {
    const unsigned k = threadIdx.x;
    const unsigned count = min(counts[k], 255u);
    unsigned rank = 0;
    for (unsigned j = 0; j < input_degree; ++j) {
      const unsigned other = min(counts[j], 255u);
      rank += other < count || (other == count && j < k);
    }
    if (rank < degree) output[size_t(row) * degree + rank] = neighbors[k];
  }
}

__global__ void count_incoming(const unsigned* graph, size_t edges, unsigned* counts) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < edges; i += size_t(blockDim.x) * gridDim.x) atomicAdd(counts + graph[i], 1u);
}
/* CPU submits only fixed-size launches. Prefix sums and graph contents stay on GPU. */
__global__ void prefix_blocks(const unsigned* counts, size_t* offsets, size_t* blocks, unsigned rows) {
  if (threadIdx.x) return;
  const unsigned begin = blockIdx.x * 256;
  const unsigned end = min(rows, begin + 256);
  size_t sum = 0;
  for (unsigned i = begin; i < end; ++i) { offsets[i] = sum; sum += counts[i]; }
  blocks[blockIdx.x] = sum;
}
__global__ void prefix_block_totals(size_t* blocks, unsigned total) {
  size_t sum = 0;
  for (unsigned i = 0; i < total; ++i) { const size_t value = blocks[i]; blocks[i] = sum; sum += value; }
}
__global__ void add_block_offsets(size_t* offsets, const size_t* blocks, unsigned rows) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) offsets[row] += blocks[row / 256];
}
__global__ void transpose_graph(const unsigned* graph, size_t edges, const size_t* offsets,
                                unsigned* cursors, unsigned* sources, uint8_t* ranks) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < edges; i += size_t(blockDim.x) * gridDim.x) {
    const unsigned dest = graph[i];
    const size_t slot = offsets[dest] + atomicAdd(cursors + dest, 1u);
    sources[slot] = unsigned(i / degree);
    ranks[slot] = uint8_t(i % degree);
  }
}
/* Preserve fixed optimizer's rank-major incoming-edge priority. Ordering of
 * sources tied at one rank is unspecified in both implementations because
 * the fixed reverse kernel also atomically appends source IDs. */
__global__ void merge_reverse(unsigned* graph, const unsigned* counts, const size_t* offsets,
                              const unsigned* sources, const uint8_t* incoming_ranks) {
  __shared__ unsigned hist[degree], cursors[degree], starts[degree], reverse[degree];
  const unsigned row = blockIdx.x;
  if (threadIdx.x < degree) { hist[threadIdx.x] = 0; cursors[threadIdx.x] = 0; reverse[threadIdx.x] = UINT32_MAX; }
  __syncthreads();
  const size_t begin = offsets[row];
  for (unsigned j = threadIdx.x; j < counts[row]; j += blockDim.x)
    atomicAdd(hist + incoming_ranks[begin + j], 1u);
  __syncthreads();
  if (threadIdx.x < degree) {
    unsigned sum = 0;
    for (unsigned k = 0; k < threadIdx.x; ++k) sum += hist[k];
    starts[threadIdx.x] = sum;
  }
  __syncthreads();
  for (unsigned j = threadIdx.x; j < counts[row]; j += blockDim.x) {
    const unsigned rank = incoming_ranks[begin + j];
    const unsigned pos = starts[rank] + atomicAdd(cursors + rank, 1u);
    if (pos < degree) reverse[pos] = sources[begin + j];
  }
  __syncthreads();
  if (threadIdx.x) return;
  unsigned* out = graph + size_t(row) * degree;
  unsigned remaining = min(counts[row], degree);
  while (remaining) {
    const unsigned candidate = reverse[--remaining];
    unsigned pos = 0;
    while (pos < degree && out[pos] != candidate) ++pos;
    if (pos < degree / 2) continue;
    if (pos >= degree) pos = degree - 1;
    for (unsigned j = pos; j > degree / 2; --j) out[j] = out[j - 1];
    out[degree / 2] = candidate;
  }
}
__global__ void validate_graph(const unsigned* graph, unsigned rows, unsigned* errors) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const unsigned* values = graph + size_t(row) * degree;
  for (unsigned k = 0; k < degree; ++k) {
    if (values[k] >= rows || values[k] == row) { atomicAdd(errors, 1u); return; }
    for (unsigned j = 0; j < k; ++j)
      if (values[k] == values[j]) { atomicAdd(errors, 1u); return; }
  }
}
bool gpu_ok() {
  const hipError_t code = hipGetLastError();
  if (code != hipSuccess) fprintf(stderr, "[seekdb_cuvs] ivfpq HIP error: %s\n", hipGetErrorString(code));
  return code == hipSuccess;
}
bool graph_errors(DeviceMemory& errors, hipStream_t stream) {
  unsigned host = 0;
  if (hipMemcpyAsync(&host, errors.pointer, sizeof(host), hipMemcpyDeviceToHost, stream) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess) return true;
  if (host) fprintf(stderr, "[seekdb_cuvs] ivfpq graph validation errors=%u\n", host);
  return host != 0;
}
int construct(cuvsResources_t res, cuvsCagraIndexParams_t params, DLManagedTensor* original,
              cuvsCagraIndex_t output, unsigned** storage) {
  const DLTensor& input = original->dl_tensor;
  const int64_t rows64 = input.shape[0], dim = input.shape[1];
  if (rows64 <= input_degree + 1 || rows64 > UINT32_MAX || params->graph_degree != degree ||
      input.dtype.code != kDLFloat || input.dtype.bits != 32 || input.dtype.lanes != 1 ||
      input.strides || input.byte_offset || dim <= 0 || input.ndim != 2 || params->metric != L2Expanded) return 1;
  const unsigned rows = unsigned(rows64);
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) return 1;
  cudaStream_t cuda_stream{};
  if (cuvsStreamGet(res, &cuda_stream) != CUVS_SUCCESS) return 1;
  const hipStream_t stream = reinterpret_cast<hipStream_t>(cuda_stream);
  Timer timer;
  DeviceMemory upload, knn, graph, candidates, candidate_distances;
  DeviceMemory errors, counts, offsets, block_offsets, cursors, sources, incoming_ranks;
  IvfObjects ivf;
  StreamFence fence{stream};
  float* dataset_ptr = static_cast<float*>(input.data);
  if (input.device.device_type == kDLCPU) {
    if (!upload.allocate(size_t(rows) * size_t(dim) * sizeof(float)) ||
        hipMemcpyAsync(upload.pointer, input.data, size_t(rows) * size_t(dim) * sizeof(float),
                       hipMemcpyHostToDevice, stream) != hipSuccess) return 1;
    dataset_ptr = upload.as<float>();
  } else if (input.device.device_type != kDLCUDA && input.device.device_type != kDLROCM) return 1;
  int64_t dataset_shape[2] = {rows64, dim};
  DLManagedTensor dataset = tensor(dataset_ptr, dataset_shape, kDLFloat, 32, device);
  if (!timer.start(stream) || cuvsIvfPqIndexParamsCreate(&ivf.build) != CUVS_SUCCESS ||
      cuvsIvfPqSearchParamsCreate(&ivf.search) != CUVS_SUCCESS ||
      cuvsIvfPqIndexCreate(&ivf.index) != CUVS_SUCCESS) return 1;
  ivf.build->metric = params->metric;
  ivf.build->add_data_on_build = true;
  ivf.build->n_lists = std::max(1u, rows / 2000u);
  ivf.build->kmeans_n_iters = 10;
  const double minimum_fraction = std::min(1.0, 32.0 * ivf.build->n_lists / rows64);
  ivf.build->kmeans_trainset_fraction = std::clamp(1.0 / std::sqrt(rows64 * 1e-5), minimum_fraction, 1.0);
  ivf.build->pq_bits = dim <= 32 ? 8 : 4;
  ivf.build->pq_dim = dim <= 32 ? unsigned(std::min<int64_t>(dim, 16)) :
      dim <= 64 ? 32 : dim <= 128 ? 64 : dim <= 192 ? 96 : unsigned(((dim / 2 + 127) / 128) * 128);
  ivf.build->codebook_kind = PER_SUBSPACE;
  ivf.search->n_probes = std::min(unsigned(SEEKDB_IVFPQ_PROBES), ivf.build->n_lists);
  ivf.search->lut_dtype = HIP_R_32F;
  ivf.search->internal_distance_dtype = HIP_R_32F;
  ivf.search->coarse_search_dtype = HIP_R_32F;
  ivf.search->max_internal_batch_size = SEEKDB_IVFPQ_INTERNAL_BATCH;
  fprintf(stderr, "[seekdb_cuvs] ivfpq_gpu probes=%u refine=%u batch=%u intermediate=%u graph=%u n=%u dim=%lld lists=%u\n",
          ivf.search->n_probes, unsigned(SEEKDB_IVFPQ_REFINEMENT), unsigned(SEEKDB_IVFPQ_BATCH),
          input_degree, degree, rows, (long long)dim, ivf.build->n_lists);
  fprintf(stderr, "[seekdb_cuvs] ivfpq_gpu internal_batch=%u batch_events=%u memory_info=%u\n",
          unsigned(SEEKDB_IVFPQ_INTERNAL_BATCH), unsigned(SEEKDB_IVFPQ_BATCH_EVENTS), unsigned(SEEKDB_IVFPQ_MEMORY_INFO));
  if (cuvsIvfPqBuild(res, ivf.build, &dataset, ivf.index) != CUVS_SUCCESS ||
      !timer.report("pq_train_build", stream)) return 1;
  const unsigned batch = std::min(unsigned(SEEKDB_IVFPQ_BATCH), rows);
  const unsigned candidate_k = std::min(rows, std::max(input_degree + 1, input_degree * unsigned(SEEKDB_IVFPQ_REFINEMENT)));
  fprintf(stderr, "[seekdb_cuvs] ivfpq_gpu candidate_k=%u shared_capacity=%u shared_bytes=%zu dtype=FP32\n",
          candidate_k, candidate_capacity, size_t(candidate_capacity) * (sizeof(float) + sizeof(unsigned)));
#if SEEKDB_IVFPQ_BATCH_EVENTS
  BatchEvents batch_events;
  if (!batch_events.create((size_t(rows) + batch - 1) / batch)) return 1;
#endif
  if (!knn.allocate(size_t(rows) * input_degree * sizeof(unsigned)) ||
      !candidates.allocate(size_t(batch) * candidate_k * sizeof(int64_t)) ||
      !candidate_distances.allocate(size_t(batch) * candidate_k * sizeof(float)) ||
      !errors.allocate(sizeof(unsigned)) ||
      hipMemsetAsync(errors.pointer, 0, sizeof(unsigned), stream) != hipSuccess ||
      !timer.start(stream)) return 1;
  for (unsigned offset = 0; offset < rows; offset += batch) {
    const unsigned count = std::min(batch, rows - offset);
    int64_t qs[2] = {count, dim}, cs[2] = {count, candidate_k};
    DLManagedTensor queries = tensor(dataset_ptr + size_t(offset) * dim, qs, kDLFloat, 32, device);
    DLManagedTensor neighbors = tensor(candidates.pointer, cs, kDLInt, 64, device);
    DLManagedTensor distances = tensor(candidate_distances.pointer, cs, kDLFloat, 32, device);
#if SEEKDB_IVFPQ_BATCH_EVENTS
    auto& events = batch_events.values[offset / batch];
    if (hipEventRecord(events[0], stream) != hipSuccess) return 1;
#endif
    if (cuvsIvfPqSearch(res, ivf.search, ivf.index, &queries, &neighbors, &distances) != CUVS_SUCCESS) return 1;
#if SEEKDB_IVFPQ_BATCH_EVENTS
    if (hipEventRecord(events[1], stream) != hipSuccess) return 1;
#endif
    hipLaunchKernelGGL((seekdb_refine_candidate_graph_l2<input_degree, candidate_capacity>), dim3(count), dim3(256), 0, stream,
                       dataset_ptr, dim, candidates.as<int64_t>(), candidate_k,
                       count, offset, rows, knn.as<unsigned>(), errors.as<unsigned>());
    if (!gpu_ok()) return 1;
#if SEEKDB_IVFPQ_BATCH_EVENTS
    if (hipEventRecord(events[2], stream) != hipSuccess) return 1;
#endif
  }
  if (!timer.report("pq_search_refine_normalize", stream) || graph_errors(errors, stream)) return 1;
#if SEEKDB_IVFPQ_BATCH_EVENTS
  if (!batch_events.report()) return 1;
#endif
  ivf.reset();
  candidates.reset(); candidate_distances.reset();
  if (!graph.allocate(size_t(rows) * degree * sizeof(unsigned)) || !timer.start(stream)) return 1;
  hipLaunchKernelGGL(prune_graph, dim3(rows), dim3(256), 0, stream, knn.as<unsigned>(), rows, graph.as<unsigned>());
  if (!gpu_ok() || !timer.report("detour_prune", stream)) return 1;
  knn.reset();
  const size_t edges = size_t(rows) * degree;
  const unsigned prefix_blocks_count = unsigned((size_t(rows) + 255) / 256);
  if (!counts.allocate(size_t(rows) * sizeof(unsigned)) || !offsets.allocate(size_t(rows) * sizeof(size_t)) ||
      !block_offsets.allocate(size_t(prefix_blocks_count) * sizeof(size_t)) ||
      !cursors.allocate(size_t(rows) * sizeof(unsigned)) || !sources.allocate(edges * sizeof(unsigned)) ||
      !incoming_ranks.allocate(edges) || hipMemsetAsync(counts.pointer, 0, size_t(rows) * sizeof(unsigned), stream) != hipSuccess ||
      hipMemsetAsync(cursors.pointer, 0, size_t(rows) * sizeof(unsigned), stream) != hipSuccess || !timer.start(stream)) return 1;
  hipLaunchKernelGGL(count_incoming, dim3(4096), dim3(256), 0, stream, graph.as<unsigned>(), edges, counts.as<unsigned>());
  hipLaunchKernelGGL(prefix_blocks, dim3(prefix_blocks_count), dim3(1), 0, stream, counts.as<unsigned>(), offsets.as<size_t>(), block_offsets.as<size_t>(), rows);
  hipLaunchKernelGGL(prefix_block_totals, dim3(1), dim3(1), 0, stream, block_offsets.as<size_t>(), prefix_blocks_count);
  hipLaunchKernelGGL(add_block_offsets, dim3(prefix_blocks_count), dim3(256), 0, stream, offsets.as<size_t>(), block_offsets.as<size_t>(), rows);
  hipLaunchKernelGGL(transpose_graph, dim3(4096), dim3(256), 0, stream, graph.as<unsigned>(), edges, offsets.as<size_t>(), cursors.as<unsigned>(), sources.as<unsigned>(), incoming_ranks.as<uint8_t>());
  hipLaunchKernelGGL(merge_reverse, dim3(rows), dim3(256), 0, stream, graph.as<unsigned>(), counts.as<unsigned>(), offsets.as<size_t>(), sources.as<unsigned>(), incoming_ranks.as<uint8_t>());
  if (!gpu_ok() || !timer.report("reverse_csr_merge", stream)) return 1;
  counts.reset(); offsets.reset(); block_offsets.reset(); cursors.reset(); sources.reset(); incoming_ranks.reset();
  if (!timer.start(stream)) return 1;
  hipLaunchKernelGGL(validate_graph, dim3((rows + 127) / 128), dim3(128), 0, stream, graph.as<unsigned>(), rows, errors.as<unsigned>());
  if (!gpu_ok() || graph_errors(errors, stream) || !timer.report("validate_graph", stream)) return 1;
  int64_t graph_shape[2] = {rows64, degree};
  DLManagedTensor gpu_graph = tensor(graph.pointer, graph_shape, kDLUInt, 32, device);
  if (!timer.start(stream) || cuvsCagraIndexFromArgs(res, params->metric, &gpu_graph, original, output) != CUVS_SUCCESS) return 1;
  *storage = graph.release_graph();
  if (cuvsStreamSync(res) != CUVS_SUCCESS || !timer.report("cagra_index_from_args", stream)) return 1;
  return 0;
}
} // namespace
extern "C" int seekdb_ivfpq_gpu_graph(cuvsResources_t res, cuvsCagraIndexParams_t params,
                                      DLManagedTensor* dataset, cuvsCagraIndex_t index,
                                      unsigned** storage) {
  if (!res || !params || !dataset || !dataset->dl_tensor.shape || dataset->dl_tensor.ndim != 2 ||
      !dataset->dl_tensor.data || !index || index->addr || !storage || *storage) return 1;
  try {
    const int result = construct(res, params, dataset, index, storage);
    if (result) {
      cuvsStreamSync(res);
      fprintf(stderr, "[seekdb_cuvs] public C IVF-PQ/GPU construction failed: %s\n", cuvsGetLastErrorText());
    }
    return result;
  } catch (...) {
    cuvsStreamSync(res);
    fprintf(stderr, "[seekdb_cuvs] public C IVF-PQ/GPU construction threw\n");
    return 1;
  }
}
