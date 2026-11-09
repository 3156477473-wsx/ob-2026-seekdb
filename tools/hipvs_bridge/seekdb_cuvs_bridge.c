/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include "ivfpq_graph.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#ifdef __cplusplus
#include <mutex>
/* The pool changes the process-wide current MR during graph construction. */
static std::recursive_mutex sk_cuvs_sdk_mutex;
#define SK_CUVS_SDK_GUARD() \
  std::lock_guard<std::recursive_mutex> sk_cuvs_sdk_guard(sk_cuvs_sdk_mutex)
#elif defined(__HIP_PLATFORM_AMD__) && !defined(SEEKDB_CUVS_CPU_TEST)
#error "The HIP bridge requires C++ SDK serialization; use the supplied build.sh"
#else
/* The explicit CPU fake has no real SDK or shared memory resource. */
#define SK_CUVS_SDK_GUARD() ((void)0)
#endif

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  float*                base_device; /* owned; idx may hold a non-owning view */
  unsigned int*         graph_device; /* owned GPU graph; index borrows it */
  long n, dim;
  int search_policy_reported;
} sk_cuvs_index;

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  SK_CUVS_SDK_GUARD();
  if (!base || n <= 0 || dim <= 0 ||
      (size_t)n > SIZE_MAX / sizeof(float) / (size_t)dim) return NULL;
  const size_t base_bytes = (size_t)n * (size_t)dim * sizeof(float);
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; int64_t ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  /* With 16-byte-aligned rows, cuVS retains a view of device input instead of
   * copying it again for the final index. NN-descent and graph sorting also
   * consume this device input, avoiding their repeated host-to-device loads.
   * Keep the allocation until after idx is destroyed. Unaligned rows retain
   * the original host path, where cuVS owns its padded device dataset. */
  if ((dim % 4) == 0) {
    if (cudaMalloc((void**)&h->base_device, base_bytes) != cudaSuccess) goto fail_base;
    if (cudaMemcpy(h->base_device, base, base_bytes, cudaMemcpyDefault) != cudaSuccess) goto fail_base;
    dset.dl_tensor.data = h->base_device;
    dset.dl_tensor.device.device_type = kDLCUDA;
  }

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_base;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 128;
  h->ip->intermediate_graph_degree = 160;
  h->ip->nn_descent_niter = 100;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  /* Preserve native GPU construction when PQ training has too few samples
   * or there are too few unique vertices for this intermediate degree. */
  if (n < SEEKDB_IVFPQ_MIN_ROWS) {
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  } else if (seekdb_ivfpq_gpu_graph(h->res, h->ip, &dset, h->idx, &h->graph_device) != 0) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: (void)cuvsStreamSync(h->res); cuvsCagraIndexDestroy(h->idx);
  if (h->graph_device) cudaFree(h->graph_device);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_base: if (h->base_device) cudaFree(h->base_device);
  cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  SK_CUVS_SDK_GUARD();
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !h->idx) return 1;
  int rc = 0;
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float)*nq*h->dim)  != CUVS_SUCCESS) { rc=10; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned)*nq*topk) != CUVS_SUCCESS) { rc=11; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float)*nq*topk)    != CUVS_SUCCESS) { rc=12; goto done; }
  if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; int64_t qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      /* Spend a larger fixed-library CAGRA search budget on graph-quality
       * variation. The returned neighbors still come directly from that SDK. */
      if (required_itopk < 640) required_itopk = 640;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      /* The fixed C API defaults to SINGLE_CTA, whose budget is limited to512.
       * Select the library's native parallel mode explicitly for larger beams. */
      sp->algo = MULTI_CTA;
      sp->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
    if (!h->search_policy_reported) {
      fprintf(stderr, "[seekdb_cuvs] cagra_search_policy algo=MULTI_CTA itopk=%zu width=%zu\n",
              sp->itopk_size, sp->search_width);
      h->search_policy_reported = 1;
    }
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float)*nq*topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned)*nq*topk);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float)*nq*h->dim);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  SK_CUVS_SDK_GUARD();
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  (void)cuvsStreamSync(h->res);
  const int buffers_safe = !h->idx || cuvsCagraIndexDestroy(h->idx) == CUVS_SUCCESS;
  if (buffers_safe) {
    if (h->graph_device) cudaFree(h->graph_device);
    if (h->base_device) cudaFree(h->base_device);
  }
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

/* Copy logical GPU rows without host staging or cudaMemcpyDefault. GetDataset
 * and GetGraph can expose padded rows and a byte offset; metadata is not GPU
 * ownership. Validate the full source span before any allocation or copy. */
static int sk_cuvs_snapshot_view(const DLManagedTensor* view, long n, int64_t dim,
                                 uint8_t code, const unsigned char** source,
                                 size_t* row_bytes, size_t* row_pitch)
{
  const DLTensor* t = &view->dl_tensor;
  const int gpu = t->device.device_type == kDLCUDA
#ifdef __HIP_PLATFORM_AMD__
      || t->device.device_type == kDLROCM
#endif
      ;
  if (!view->deleter || !t->data || !t->shape || t->ndim != 2 ||
      t->shape[0] != n || t->shape[1] != dim || !gpu || t->device.device_id < 0 ||
      t->dtype.code != code || t->dtype.bits != 32 || t->dtype.lanes != 1 ||
      dim <= 0 || (uint64_t)dim > SIZE_MAX / 4 ||
      (size_t)n > SIZE_MAX / ((size_t)dim * 4)) return 1;
  const int64_t stride = t->strides ? t->strides[0] : dim;
  if ((t->strides && t->strides[1] != 1) || stride < dim ||
      (uint64_t)stride > SIZE_MAX / 4 || t->byte_offset > SIZE_MAX) return 1;
  *row_bytes = (size_t)dim * 4;
  *row_pitch = (size_t)stride * 4;
  if ((size_t)(n - 1) > (SIZE_MAX - *row_bytes) / *row_pitch) return 1;
  const size_t span = (size_t)(n - 1) * *row_pitch + *row_bytes;
  const uintptr_t address = (uintptr_t)t->data;
  const size_t offset = (size_t)t->byte_offset;
  if (address > UINTPTR_MAX - offset || address + offset > UINTPTR_MAX - (span - 1)) return 1;
  *source = (const unsigned char*)t->data + offset;
  return 0;
}

/* SKCAGR04: 64-byte LE header, uint32 graph rows, then exact FP32 bits.
 * No SDK serialization or host-side graph/vector conversion is involved. */
#define SK_SNAPSHOT_CHUNK (8u * 1024u * 1024u)
static const unsigned char sk_snapshot_magic[8] = {'S','K','C','A','G','R','0','4'};
static uint64_t sk_snapshot_u64(const unsigned char* p)
{
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}
static uint32_t sk_snapshot_u32(const unsigned char* p)
{
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void sk_snapshot_put64(unsigned char* p, uint64_t v)
{ for (unsigned i = 0; i < 8; ++i) p[i] = (unsigned char)(v >> (8 * i)); }
static void sk_snapshot_put32(unsigned char* p, uint32_t v)
{ for (unsigned i = 0; i < 4; ++i) p[i] = (unsigned char)(v >> (8 * i)); }
static int sk_snapshot_platform(void)
{
  const uint32_t one = 1;
  return sizeof(float) == 4 && sizeof(unsigned int) == 4 && FLT_RADIX == 2 &&
         FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
         *(const unsigned char*)&one == 1;
}
/* Word-wise FNV-style corruption check; chunks always end on uint32 boundaries.
 * This is not a cryptographic signature or a replacement for external SHA256. */
static uint64_t sk_snapshot_checksum(uint64_t hash, const unsigned char* bytes, size_t size)
{
  for (size_t i = 0; i < size; i += 4)
    hash = (hash ^ sk_snapshot_u32(bytes + i)) * UINT64_C(1099511628211);
  return hash;
}
static int sk_snapshot_write_view(FILE* file, unsigned char* buffer, const unsigned char* source,
                                   long n, size_t row_bytes, size_t pitch, uint64_t* checksum)
{
  const size_t total = (size_t)n * row_bytes;
  for (size_t at = 0; at < total;) {
    size_t count = total - at;
    const unsigned char* src = source + at;
    if (pitch != row_bytes) {
      const size_t row = at / row_bytes, col = at % row_bytes;
      src = source + row * pitch + col;
      count = row_bytes - col;
    }
    if (count > SK_SNAPSHOT_CHUNK) count = SK_SNAPSHOT_CHUNK;
    if (cudaMemcpy(buffer, src, count, cudaMemcpyDeviceToHost) != cudaSuccess ||
        fwrite(buffer, 1, count, file) != count) return 1;
    *checksum = sk_snapshot_checksum(*checksum, buffer, count);
    at += count;
  }
  return 0;
}
static int sk_snapshot_read_gpu(FILE* file, unsigned char* buffer, void* output,
                                 size_t bytes, uint64_t* checksum)
{
  for (size_t at = 0; at < bytes;) {
    size_t count = bytes - at;
    if (count > SK_SNAPSHOT_CHUNK) count = SK_SNAPSHOT_CHUNK;
    if (fread(buffer, 1, count, file) != count) return 1;
    *checksum = sk_snapshot_checksum(*checksum, buffer, count);
    if (cudaMemcpy((unsigned char*)output + at, buffer, count,
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    at += count;
  }
  return 0;
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  SK_CUVS_SDK_GUARD();
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !h->idx || !filename || h->n <= 0 || h->dim <= 0 ||
      (uint64_t)h->n > UINT32_MAX || !sk_snapshot_platform()) return 1;
  DLManagedTensor base_view = {0}, graph_view = {0};
  const unsigned char *base = NULL, *graph = NULL;
  size_t base_row = 0, base_pitch = 0, graph_row = 0, graph_pitch = 0;
  int64_t n = 0, dim = 0, degree = 0;
  unsigned char header[64] = {0};
  unsigned char* buffer = NULL;
  FILE* file = NULL;
  uint64_t checksum = UINT64_C(14695981039346656037);
  int rc = 1;
  do {
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS ||
        cuvsCagraIndexGetSize(h->idx, &n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(h->idx, &dim) != CUVS_SUCCESS ||
        cuvsCagraIndexGetGraphDegree(h->idx, &degree) != CUVS_SUCCESS ||
        n != h->n || dim != h->dim || degree <= 0 ||
        cuvsCagraIndexGetGraph(h->idx, &graph_view) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDataset(h->idx, &base_view) != CUVS_SUCCESS) break;
    if (sk_cuvs_snapshot_view(&base_view, h->n, h->dim, kDLFloat, &base, &base_row, &base_pitch) ||
        sk_cuvs_snapshot_view(&graph_view, h->n, degree, kDLUInt, &graph, &graph_row, &graph_pitch) ||
        base_view.dl_tensor.device.device_id != graph_view.dl_tensor.device.device_id) break;
    const size_t graph_bytes = (size_t)h->n * graph_row, base_bytes = (size_t)h->n * base_row;
    if (graph_bytes > SIZE_MAX - 64 || base_bytes > SIZE_MAX - 64 - graph_bytes) break;
    memcpy(header, sk_snapshot_magic, 8);
    sk_snapshot_put32(header + 8, 4);
    sk_snapshot_put32(header + 12, (uint32_t)L2Expanded);
    sk_snapshot_put64(header + 16, (uint64_t)h->n);
    sk_snapshot_put64(header + 24, (uint64_t)h->dim);
    sk_snapshot_put64(header + 32, (uint64_t)degree);
    sk_snapshot_put64(header + 40, (uint64_t)graph_bytes);
    sk_snapshot_put64(header + 48, (uint64_t)base_bytes);
    buffer = (unsigned char*)malloc(SK_SNAPSHOT_CHUNK);
    if (!buffer || !(file = fopen(filename, "wb")) || fwrite(header, 1, sizeof(header), file) != sizeof(header)) break;
    if (sk_snapshot_write_view(file, buffer, graph, h->n, graph_row, graph_pitch, &checksum) ||
        sk_snapshot_write_view(file, buffer, base, h->n, base_row, base_pitch, &checksum)) break;
    sk_snapshot_put64(header + 56, checksum);
    if (fseek(file, 0, SEEK_SET) != 0 || fwrite(header, 1, sizeof(header), file) != sizeof(header)) break;
    rc = 0;
  } while (0);
  if (file && fclose(file) != 0) rc = 1;
  free(buffer);
  if (base_view.deleter) base_view.deleter(&base_view);
  if (graph_view.deleter) graph_view.deleter(&graph_view);
  if (rc) fprintf(stderr, "[seekdb_cuvs] stream serialize failed: %s\n", cuvsGetLastErrorText());
  return rc;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  SK_CUVS_SDK_GUARD();
  if (!filename || n <= 0 || dim <= 0) return NULL;
  FILE* file = fopen(filename, "rb");
  if (!file) return NULL;
  unsigned char header[64] = {0};
  unsigned char* buffer = NULL;
  sk_cuvs_index* h = NULL;
  const size_t prefix = fread(header, 1, 8, file);
  const int streaming = prefix == 8 && memcmp(header, sk_snapshot_magic, 8) == 0;
  int ok = 0;
  if (prefix != 8) { fclose(file); return NULL; }
  if (!streaming) {
    if (fclose(file) != 0) return NULL;
    file = NULL; /* Legacy SDK files are read by the unchanged SDK reader. */
  }
  h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) { if (file) fclose(file); return NULL; }
  do {
    if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS ||
        cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS ||
        cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) break;
    if (!streaming) {
      if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) break;
    } else {
      if (!sk_snapshot_platform() || fread(header + 8, 1, 56, file) != 56) break;
      const uint64_t rows = sk_snapshot_u64(header + 16), columns = sk_snapshot_u64(header + 24);
      const uint64_t degree = sk_snapshot_u64(header + 32), graph_bytes = sk_snapshot_u64(header + 40);
      const uint64_t base_bytes = sk_snapshot_u64(header + 48), expected_checksum = sk_snapshot_u64(header + 56);
      if (sk_snapshot_u32(header + 8) != 4 || sk_snapshot_u32(header + 12) != (uint32_t)L2Expanded ||
          rows != (uint64_t)n || columns != (uint64_t)dim || rows > UINT32_MAX ||
          !degree || degree > INT64_MAX || columns > SIZE_MAX / 4 || degree > SIZE_MAX / 4 ||
          rows > SIZE_MAX / (columns * 4) || rows > SIZE_MAX / (degree * 4) ||
          base_bytes != rows * columns * 4 || graph_bytes != rows * degree * 4 ||
          graph_bytes > SIZE_MAX - 64 || base_bytes > SIZE_MAX - 64 - graph_bytes) break;
      buffer = (unsigned char*)malloc(SK_SNAPSHOT_CHUNK);
      if (!buffer || cudaMalloc((void**)&h->base_device, (size_t)base_bytes) != cudaSuccess ||
          cudaMalloc((void**)&h->graph_device, (size_t)graph_bytes) != cudaSuccess) break;
      uint64_t checksum = UINT64_C(14695981039346656037);
      if (sk_snapshot_read_gpu(file, buffer, h->graph_device, (size_t)graph_bytes, &checksum) ||
          sk_snapshot_read_gpu(file, buffer, h->base_device, (size_t)base_bytes, &checksum) ||
          checksum != expected_checksum) break;
      const int tail = fgetc(file);
      if (tail != EOF || ferror(file)) break;
      if (fclose(file) != 0) { file = NULL; break; }
      file = NULL;
      if (cudaDeviceSynchronize() != cudaSuccess) break;
      int device = -1;
      if (cudaGetDevice(&device) != cudaSuccess || device < 0) break;
      int64_t base_shape[2] = {n, dim}, graph_shape[2] = {n, (int64_t)degree};
      DLManagedTensor base = {0}, graph = {0};
      base.dl_tensor.data = h->base_device;
      base.dl_tensor.device.device_type = kDLCUDA; base.dl_tensor.device.device_id = device;
      base.dl_tensor.ndim = 2; base.dl_tensor.dtype.code = kDLFloat;
      base.dl_tensor.dtype.bits = 32; base.dl_tensor.dtype.lanes = 1;
      base.dl_tensor.shape = base_shape;
      graph.dl_tensor = base.dl_tensor;
      graph.dl_tensor.data = h->graph_device; graph.dl_tensor.dtype.code = kDLUInt;
      graph.dl_tensor.shape = graph_shape;
      if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph, &base, h->idx) != CUVS_SUCCESS ||
          !h->idx->addr || cuvsStreamSync(h->res) != CUVS_SUCCESS || cudaDeviceSynchronize() != cudaSuccess) break;
      int64_t actual_degree = 0;
      if (cuvsCagraIndexGetGraphDegree(h->idx, &actual_degree) != CUVS_SUCCESS || (uint64_t)actual_degree != degree) break;
    }
    int64_t actual_n = 0, actual_dim = 0;
    if (cuvsCagraIndexGetSize(h->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(h->idx, &actual_dim) != CUVS_SUCCESS || actual_n != n || actual_dim != dim) break;
    h->n = n; h->dim = dim;
    ok = 1;
  } while (0);
  if (file && fclose(file) != 0) ok = 0;
  if (!ok && h->res) (void)cudaDeviceSynchronize();
  free(buffer);
  if (ok) return h;
  if (h->res) (void)cuvsStreamSync(h->res);
  int buffers_safe = !h->idx || cuvsCagraIndexDestroy(h->idx) == CUVS_SUCCESS;
  if (buffers_safe) {
    if (h->graph_device) cudaFree(h->graph_device);
    if (h->base_device) cudaFree(h->base_device);
  }
  if (h->ip) cuvsCagraIndexParamsDestroy(h->ip);
  if (h->res) cuvsResourcesDestroy(h->res);
  free(h);
  fprintf(stderr, "[seekdb_cuvs] stream/legacy deserialize failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_copy_dataset(void* handle, float* out_base, long n, long dim)
{
  SK_CUVS_SDK_GUARD();
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !h->idx || !out_base || n <= 0 || dim <= 0 || h->n != n || h->dim != dim ||
      (size_t)n > SIZE_MAX / sizeof(float) / (size_t)dim) return 1;
  DLManagedTensor view;
  memset(&view, 0, sizeof(view));
  int rc = 2;
  do {
    if (cuvsCagraIndexGetDataset(h->idx, &view) != CUVS_SUCCESS) break;
    rc = 3;
    const DLTensor* t = &view.dl_tensor;
    /* The CUDA-compatibility fork may retain the CUDA DLPack tag on HIP.
     * HIP also understands an explicitly tagged ROCm device allocation. */
    const int gpu_device = t->device.device_type == kDLCUDA
#ifdef __HIP_PLATFORM_AMD__
        || t->device.device_type == kDLROCM
#endif
        ;
    if (!view.deleter || !t->data || t->ndim != 2 || !t->shape ||
        t->shape[0] != n || t->shape[1] != dim ||
        t->dtype.code != kDLFloat || t->dtype.bits != 32 || t->dtype.lanes != 1 ||
        !gpu_device || t->device.device_id < 0) break;
    const int64_t stride = t->strides ? t->strides[0] : (int64_t)dim;
    if ((t->strides && t->strides[1] != 1) || stride < dim ||
        (uint64_t)stride > SIZE_MAX / sizeof(float) || t->byte_offset > SIZE_MAX) break;
    const size_t row_bytes = (size_t)dim * sizeof(float);
    const size_t row_pitch = (size_t)stride * sizeof(float);
    if ((size_t)(n - 1) > (SIZE_MAX - row_bytes) / row_pitch) break;
    const size_t source_span = (size_t)(n - 1) * row_pitch + row_bytes;
    const size_t offset = (size_t)t->byte_offset;
    const uintptr_t address = (uintptr_t)t->data;
    if (address > UINTPTR_MAX - offset ||
        address + offset > UINTPTR_MAX - (source_span - 1)) break;
    const unsigned char* source = (const unsigned char*)t->data + offset;
    rc = 4;
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) break;
    rc = 5;
    if (row_pitch == row_bytes) {
      if (cudaMemcpy(out_base, source, (size_t)n * row_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) break;
    } else {
      /* CAGRA may pad unaligned rows. Copy only logical columns, preserving
       * FP32 bits without treating padding as the next vector's contents. */
      int failed = 0;
      for (size_t row = 0; row < (size_t)n; ++row) {
        if (cudaMemcpy(out_base + row * (size_t)dim, source + row * row_pitch,
                       row_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
          failed = 1;
          break;
        }
      }
      if (failed) break;
    }
    rc = 0;
  } while (0);
  /* GetDataset owns the view metadata; its deleter does not own GPU data.
   * Also release a partial view if the fixed C API reports an error. */
  if (view.deleter) view.deleter(&view);
  if (rc) fprintf(stderr, "[seekdb_cuvs] dataset restore rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  SK_CUVS_SDK_GUARD();
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
