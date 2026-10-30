/* Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
 * Candidate-only FP32 squared-L2 refinement for GPU graph construction.
 * Include after HIP runtime / stdint declarations; no SDK C++ dependency.
 * This is not a query implementation and must not replace cuvsCagraSearch.
 */

/* Error bits share the caller's zero-initialized device errors word. The caller
 * MUST read/check it after stream synchronization before using the graph.
 * 0x100: invalid shape / candidate ID; 0x200: non-finite distance;
 * 0x400: not enough distinct non-self candidate IDs.
 * Every failure leaves the overall construction rejected, never a success with
 * substituted/fabricated neighbors. Output is UINT32_MAX until actually filled.
 */
/* Three-byte IDs: naturally aligned low16[count], then high8[count].
 * 0xffffff is reserved for UINT32_MAX; every real ID must be < rows. */
static_assert(sizeof(unsigned) == 4, "CAGRA uint32 ID storage");
__host__ __device__ inline bool seekdb_knn_use_u24(uint64_t rows)
{ return rows > 0 && rows <= UINT64_C(0x00ffffff); }
static inline bool seekdb_knn_storage_size(size_t rows, unsigned columns, bool packed, size_t* bytes)
{
  const size_t width = packed ? 3 : sizeof(unsigned);
  if (!bytes || !rows || !columns || rows > SIZE_MAX / columns) return false;
  const size_t entries = rows * columns;
  if (entries > SIZE_MAX / width) return false;
  *bytes = entries * width;
  return true;
}
template<bool Packed>
__device__ inline void seekdb_knn_store(unsigned* storage, size_t entries, size_t at, unsigned id)
{
  if constexpr (Packed) {
    const unsigned value = id == UINT32_MAX ? 0x00ffffffu : id;
    reinterpret_cast<uint16_t*>(storage)[at] = uint16_t(value);
    (reinterpret_cast<uint8_t*>(storage) + entries * 2)[at] = uint8_t(value >> 16);
  } else storage[at] = id;
}
template<bool Packed>
__device__ inline unsigned seekdb_knn_load(const unsigned* storage, size_t entries, size_t at)
{
  if constexpr (Packed) {
    const unsigned value = unsigned(reinterpret_cast<const uint16_t*>(storage)[at]) |
        (unsigned((reinterpret_cast<const uint8_t*>(storage) + entries * 2)[at]) << 16);
    return value == 0x00ffffffu ? UINT32_MAX : value;
  } else return storage[at];
}

template <unsigned OutputDegree, unsigned MaxCandidates = 1024, bool PackedIds = false>
__global__ void seekdb_refine_candidate_graph_l2(
    const float* dataset,
    int64_t dim,
    const int64_t* candidates,
    unsigned candidate_k,
    unsigned rows_count,
    unsigned offset,
    unsigned nrows,
    unsigned* graph,
    unsigned* errors)
{
  static_assert(OutputDegree > 0 && OutputDegree <= 256,
                "supported graph output degree");
  static_assert(MaxCandidates == 1024 || MaxCandidates == 2048, "supported shared candidate capacity");
  constexpr unsigned max_candidates = MaxCandidates;
  __shared__ float distances[max_candidates];
  __shared__ unsigned ids[max_candidates];
  const unsigned row = blockIdx.x;
  if (row >= rows_count) return;
  const uint64_t self64 = uint64_t(offset) + row;
  if (!dataset || !candidates || !graph || !errors || dim <= 0 ||
      candidate_k < OutputDegree || candidate_k > max_candidates || self64 >= nrows ||
      (PackedIds && !seekdb_knn_use_u24(nrows)) ||
      blockDim.x < unsigned(warpSize) || blockDim.x % unsigned(warpSize) != 0) {
    if (threadIdx.x == 0 && errors) atomicOr(errors, 0x100u);
    return;
  }
  const unsigned self = unsigned(self64);
  const size_t entries = size_t(nrows) * OutputDegree;
  const size_t output = size_t(self) * OutputDegree;
  for (unsigned i = threadIdx.x; i < OutputDegree; i += blockDim.x)
    seekdb_knn_store<PackedIds>(graph, entries, output + i, UINT32_MAX);
  unsigned padded = 1;
  while (padded < candidate_k) padded <<= 1;
  for (unsigned i = threadIdx.x; i < padded; i += blockDim.x) {
    distances[i] = __int_as_float(0x7f800000);
    ids[i] = UINT32_MAX;
  }
  __syncthreads();

  /* Use the actual hardware wave size (32 or 64), never assume CUDA warp32.
   * Each wave refines one candidate at a time, reading only its logical vector.
   * Shared storage is 8 KiB for capacity1024 or 16 KiB for capacity2048,
   * selected at compile time, independent of dataset identity or dimension. */
  const unsigned lane = threadIdx.x % unsigned(warpSize);
  const unsigned wave = threadIdx.x / unsigned(warpSize);
  const unsigned waves = blockDim.x / unsigned(warpSize);
  const float* query = dataset + size_t(self) * size_t(dim);
  for (unsigned c = wave; c < candidate_k; c += waves) {
    const int64_t candidate = candidates[size_t(row) * candidate_k + c];
    const bool valid = candidate >= 0 && uint64_t(candidate) < uint64_t(nrows);
    float sum = 0.0f;
    if (valid) {
      const float* base = dataset + size_t(candidate) * size_t(dim);
      for (int64_t d = lane; d < dim; d += warpSize) {
        const float difference = query[d] - base[d];
        sum = fmaf(difference, difference, sum);
      }
      for (unsigned delta = unsigned(warpSize) / 2; delta != 0; delta >>= 1)
        sum += __shfl_down(sum, delta, warpSize);
    }
    if (lane == 0) {
      if (!valid) {
        atomicOr(errors, 0x100u);
      } else if (!isfinite(sum)) {
        atomicOr(errors, 0x200u);
      } else {
        distances[c] = sum;
        ids[c] = unsigned(candidate);
      }
    }
  }
  __syncthreads();

  /* Bitonic ordering is (FP32 squared distance, original row ID). Padding is
   * +infinity/UINT32_MAX; no NaN reaches a comparator. Exactly one thread owns
   * each compare pair, and all pairs complete before the next network stage. */
  for (unsigned width = 2; width <= padded; width <<= 1) {
    for (unsigned stride = width >> 1; stride != 0; stride >>= 1) {
      for (unsigned i = threadIdx.x; i < padded; i += blockDim.x) {
        const unsigned other = i ^ stride;
        if (other > i) {
          const float left_distance = distances[i];
          const float right_distance = distances[other];
          const unsigned left_id = ids[i], right_id = ids[other];
          const bool right_before_left = right_distance < left_distance ||
              (right_distance == left_distance && right_id < left_id);
          const bool left_before_right = left_distance < right_distance ||
              (left_distance == right_distance && left_id < right_id);
          const bool ascending = (i & width) == 0;
          if (ascending ? right_before_left : left_before_right) {
            distances[i] = right_distance;
            distances[other] = left_distance;
            ids[i] = right_id;
            ids[other] = left_id;
          }
        }
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    unsigned count = 0;
    unsigned previous = UINT32_MAX;
    for (unsigned c = 0; c < candidate_k && count < OutputDegree; ++c) {
      const unsigned id = ids[c];
      if (id == UINT32_MAX) break;
      if (id == self || id == previous) continue;
      seekdb_knn_store<PackedIds>(graph, entries, output + count++, id);
      previous = id;
    }
    if (count != OutputDegree) atomicOr(errors, 0x400u);
  }
}
