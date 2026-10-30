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

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  float*                base_device; /* owned; idx may hold a non-owning view */
  unsigned int*         graph_device; /* owned GPU graph; index borrows it */
  long n, dim;
} sk_cuvs_index;

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
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
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
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
      if (required_itopk < 512) required_itopk = 512;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      sp->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
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
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  (void)cuvsStreamSync(h->res);
  cuvsCagraIndexDestroy(h->idx);
  if (h->graph_device) cudaFree(h->graph_device);
  if (h->base_device) cudaFree(h->base_device);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_copy_dataset(void* handle, float* out_base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !out_base || n <= 0 || dim <= 0 || h->n != n || h->dim != dim ||
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
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
