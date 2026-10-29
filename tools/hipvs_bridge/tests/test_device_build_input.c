/* Linux LP64 CPU fake: exercises the actual candidate C file, never HIP/ANN. */
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#if LONG_MAX != INT64_MAX
#error This test requires Linux LP64; DLPack shape is int64_t, not a Windows long shim.
#endif

enum Fault { F_NONE, F_HOST_CALLOC, F_RESOURCE, F_MALLOC, F_COPY, F_PARAMS,
             F_INDEX, F_BUILD, F_LOAD, F_GET_DATASET, F_SYNC };
static enum Fault fault;
static void *bridge_calloc(size_t n, size_t bytes)
{
  return fault == F_HOST_CALLOC ? NULL : calloc(n, bytes);
}
#define calloc bridge_calloc
#include "../seekdb_cuvs_bridge.c"
#undef calloc

static int resources, params, indexes, owned_datasets, allocations, frees, copies;
static int build_calls, serialize_calls, load_calls, sync_calls, last_device;
static int operation_clock, last_sync, last_index_destroy, last_build;
static int dataset_views, dataset_view_deletes, copy_fail_at;
enum ViewFault { V_VALID, V_N, V_DIM, V_DTYPE, V_LANES, V_DEVICE, V_DEVICE_ID,
                 V_DATA, V_SHAPE, V_NDIM, V_COL_STRIDE, V_ROW_STRIDE,
                 V_NEG_STRIDE, V_SPAN_OVERFLOW, V_OFFSET_OVERFLOW, V_ADDRESS_OVERFLOW };
static enum ViewFault view_fault;
static int64_t view_row_stride;
static uint64_t view_byte_offset;
static int null_view_strides;
static int rocm_view_device;
static void *device_input;
static size_t device_bytes;
static cuvsCagraIndex_t active_index;
static const float *expected;
static int64_t expected_n, expected_dim;

static void no_live_resources(void)
{
  assert(!resources && !params && !indexes && !owned_datasets && !device_input && !dataset_views);
  assert(allocations == frees);
}
cudaError_t cudaMalloc(void **ptr, size_t bytes)
{
  assert(!device_input && bytes);
  if (fault == F_MALLOC) { *ptr = NULL; return cudaErrorUnknown; }
  device_input = malloc(bytes); assert(device_input);
  *ptr = device_input; device_bytes = bytes; ++allocations; return cudaSuccess;
}
cudaError_t cudaFree(void *ptr)
{
  assert(ptr == device_input);
  /* Index must stop borrowing this buffer before the bridge frees it. */
  assert(!active_index || active_index->dataset != ptr);
  if (last_build) assert(last_index_destroy > last_build && last_sync > last_build);
  free(ptr); device_input = NULL; device_bytes = 0; ++frees; return cudaSuccess;
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, int kind)
{
  assert((kind == cudaMemcpyDefault || kind == cudaMemcpyDeviceToHost) && dst && src); ++copies;
  if (fault == F_COPY || copies == copy_fail_at) return cudaErrorUnknown;
  if (dst == device_input) assert(bytes == device_bytes);
  memcpy(dst, src, bytes); return cudaSuccess;
}
const char *cuvsGetLastErrorText(void) { return "injected CPU fake error"; }
cuvsError_t cuvsResourcesCreate(cuvsResources_t *r)
{
  if (fault == F_RESOURCE) return CUVS_ERROR;
  *r = 1; ++resources; return CUVS_SUCCESS;
}
cuvsError_t cuvsResourcesDestroy(cuvsResources_t r)
{
  assert(r && !indexes && !params); --resources; return CUVS_SUCCESS;
}
cuvsError_t cuvsStreamSync(cuvsResources_t r)
{
  assert(r); ++sync_calls; last_sync = ++operation_clock;
  return fault == F_SYNC ? CUVS_ERROR : CUVS_SUCCESS;
}
cuvsError_t cuvsRMMAlloc(cuvsResources_t r, void **p, size_t b)
{
  assert(r); *p = malloc(b); return *p ? CUVS_SUCCESS : CUVS_ERROR;
}
cuvsError_t cuvsRMMFree(cuvsResources_t r, void *p, size_t b)
{
  assert(r && p && b); free(p); return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t *p)
{
  if (fault == F_PARAMS) return CUVS_ERROR;
  *p = (cuvsCagraIndexParams_t)calloc(1, sizeof(**p)); assert(*p); ++params; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t p)
{
  assert(p); free(p); --params; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t *p)
{
  if (fault == F_INDEX) return CUVS_ERROR;
  assert(!active_index);
  *p = (cuvsCagraIndex_t)calloc(1, sizeof(**p)); assert(*p);
  active_index = *p; ++indexes; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t p)
{
  assert(p == active_index);
  if (last_build) assert(last_sync > last_build);
  if (p->borrowed) assert(p->dataset == device_input);
  else if (p->dataset) { free(p->dataset); --owned_datasets; }
  free(p); active_index = NULL; --indexes;
  last_index_destroy = ++operation_clock; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraBuild(cuvsResources_t r, cuvsCagraIndexParams_t p,
                         DLManagedTensor *t, cuvsCagraIndex_t i)
{
  assert(r && p && i == active_index && t); ++build_calls;
  assert(p->build_algo == NN_DESCENT && p->graph_degree == 128 &&
         p->intermediate_graph_degree == 160 && p->nn_descent_niter == 100);
  DLTensor *d = &t->dl_tensor;
  assert(d->ndim == 2 && d->shape[0] == expected_n && d->shape[1] == expected_dim);
  assert(d->dtype.code == kDLFloat && d->dtype.bits == 32 && d->dtype.lanes == 1);
  assert(!d->strides && !d->byte_offset);
  size_t bytes = (size_t)expected_n * (size_t)expected_dim * sizeof(float);
  last_device = d->device.device_type;
  assert(!memcmp(d->data, expected, bytes));
  i->n = expected_n; i->dim = expected_dim;
  if (d->device.device_type == kDLCUDA) {
    assert(d->data == device_input); i->dataset = (float *)d->data; i->borrowed = 1;
  } else {
    assert(d->device.device_type == kDLCPU && d->data == expected && !device_input);
    i->dataset = (float *)malloc(bytes); assert(i->dataset);
    memcpy(i->dataset, d->data, bytes); ++owned_datasets;
  }
  /* Even the failing backend has retained a reference, exposing cleanup order. */
  last_build = ++operation_clock;
  return fault == F_BUILD ? CUVS_ERROR : CUVS_SUCCESS;
}
cuvsError_t cuvsCagraSerialize(cuvsResources_t r, const char *path,
                             cuvsCagraIndex_t i, bool include_dataset)
{
  assert(r && path && i == active_index && include_dataset && i->dataset);
  assert(!memcmp(i->dataset, expected, (size_t)i->n * (size_t)i->dim * sizeof(float)));
  ++serialize_calls; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraDeserialize(cuvsResources_t r, const char *path, cuvsCagraIndex_t i)
{
  assert(r && path && i == active_index); ++load_calls;
  if (fault == F_LOAD) return CUVS_ERROR;
  size_t bytes = (size_t)expected_n * (size_t)expected_dim * sizeof(float);
  i->n = expected_n; i->dim = expected_dim;
  i->dataset = (float *)malloc(bytes); assert(i->dataset);
  memcpy(i->dataset, expected, bytes); ++owned_datasets; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t i, int64_t *n)
{ *n = i->n; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t i, int64_t *dim)
{ *dim = i->dim; return CUVS_SUCCESS; }
typedef struct { int64_t shape[2], strides[2]; } DatasetViewMetadata;
static void delete_dataset_view(DLManagedTensor *t)
{
  assert(t && t->manager_ctx && dataset_views);
  /* Only metadata is owned here: the index retains its dataset. */
  free(t->manager_ctx); t->manager_ctx = NULL; t->deleter = NULL;
  --dataset_views; ++dataset_view_deletes;
}
cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t i, DLManagedTensor *t)
{
  assert(i == active_index && t && !t->deleter);
  DatasetViewMetadata *m = (DatasetViewMetadata *)malloc(sizeof(*m)); assert(m);
  m->shape[0] = i->n; m->shape[1] = i->dim;
  m->strides[0] = view_row_stride; m->strides[1] = 1;
  t->manager_ctx = m; t->deleter = delete_dataset_view; ++dataset_views;
  t->dl_tensor.data = i->dataset;
  t->dl_tensor.device.device_type = rocm_view_device ? kDLROCM : kDLCUDA;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2; t->dl_tensor.shape = m->shape;
  t->dl_tensor.strides = null_view_strides ? NULL : m->strides;
  t->dl_tensor.byte_offset = view_byte_offset;
  t->dl_tensor.dtype.code = kDLFloat; t->dl_tensor.dtype.bits = 32; t->dl_tensor.dtype.lanes = 1;
  switch (view_fault) {
    case V_N: ++m->shape[0]; break;
    case V_DIM: ++m->shape[1]; break;
    case V_DTYPE: t->dl_tensor.dtype.bits = 16; break;
    case V_LANES: t->dl_tensor.dtype.lanes = 2; break;
    case V_DEVICE: t->dl_tensor.device.device_type = kDLCPU; break;
    case V_DEVICE_ID: t->dl_tensor.device.device_id = -1; break;
    case V_DATA: t->dl_tensor.data = NULL; break;
    case V_SHAPE: t->dl_tensor.shape = NULL; break;
    case V_NDIM: t->dl_tensor.ndim = 1; break;
    case V_COL_STRIDE: m->strides[1] = 2; break;
    case V_ROW_STRIDE: m->strides[0] = i->dim - 1; break;
    case V_NEG_STRIDE: m->strides[0] = -1; break;
    case V_SPAN_OVERFLOW: m->strides[0] = INT64_MAX; break;
    case V_OFFSET_OVERFLOW: t->dl_tensor.byte_offset = UINT64_MAX; break;
    case V_ADDRESS_OVERFLOW:
      t->dl_tensor.data = (void *)(UINTPTR_MAX - 7); t->dl_tensor.byte_offset = 16; break;
    default: break;
  }
  /* Deliberately allocate metadata even on error to check partial-view cleanup. */
  return fault == F_GET_DATASET ? CUVS_ERROR : CUVS_SUCCESS;
}
/* Link the real bridge search code without substituting a second product source. */
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t *p)
{ *p = (cuvsCagraSearchParams_t)calloc(1, sizeof(**p)); return *p ? CUVS_SUCCESS : CUVS_ERROR; }
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t p)
{ free(p); return CUVS_SUCCESS; }
cuvsError_t cuvsCagraSearch(cuvsResources_t r, cuvsCagraSearchParams_t p,
    cuvsCagraIndex_t i, DLManagedTensor *q, DLManagedTensor *n, DLManagedTensor *d, cuvsFilter f)
{
  (void)r; (void)p; (void)i; (void)q; (void)n; (void)d; (void)f;
  return CUVS_ERROR; /* Search is outside this ownership test's assertions. */
}

static void reset_order(void)
{ last_build = last_sync = last_index_destroy = operation_clock = 0; }
static void success_case(long dim)
{
  expected_n = 4; expected_dim = dim; fault = F_NONE; reset_order();
  float *base = (float *)malloc((size_t)expected_n * (size_t)dim * sizeof(float)); assert(base);
  for (int64_t j = 0; j < expected_n * dim; ++j) base[j] = (float)j * 0.125f;
  expected = base;
  int a = allocations, c = copies;
  void *h = seekdb_cuvs_build(base, (long)expected_n, dim); assert(h);
  assert(last_device == ((dim % 4) ? kDLCPU : kDLCUDA));
  assert(allocations - a == ((dim % 4) ? 0 : 1));
  assert(copies - c == ((dim % 4) ? 0 : 1));
  if (!(dim % 4)) assert(((sk_cuvs_index *)h)->base_device == device_input);
  assert(seekdb_cuvs_serialize(h, "fake.snapshot") == 0);
  seekdb_cuvs_free(h); no_live_resources();

  /* Loaded indexes own their dataset through cuVS, never through base_device. */
  reset_order(); a = allocations; c = build_calls;
  h = seekdb_cuvs_deserialize("fake.snapshot", (long)expected_n, dim); assert(h);
  assert(!((sk_cuvs_index *)h)->base_device && allocations == a && build_calls == c);
  assert(seekdb_cuvs_serialize(h, "fake.loaded") == 0);
  seekdb_cuvs_free(h); no_live_resources(); free(base); expected = NULL;
}
static void dataset_restore_cases(int padded, int offset, int no_strides)
{
  float base[4 * 5], restored[4 * 5];
  for (size_t j = 0; j < 20; ++j) {
    uint32_t bits = j == 0 ? 0x80000000u : (j == 1 ? 0x7fc00123u : 0x3f800000u + (uint32_t)j);
    memcpy(base + j, &bits, sizeof(bits));
  }
  expected = base; expected_n = 4; expected_dim = 5;
  fault = F_NONE; view_fault = V_VALID; reset_order();
  void *h = seekdb_cuvs_deserialize("fake.dataset", 4, 5); assert(h);
  view_row_stride = padded ? 8 : 5; view_byte_offset = offset ? 16 : 0;
  null_view_strides = no_strides;
  if (padded || offset) {
    float *storage = (float *)malloc((size_t)view_byte_offset + 4 * (size_t)view_row_stride * sizeof(float)); assert(storage);
    memset(storage, 0xa5, (size_t)view_byte_offset + 4 * (size_t)view_row_stride * sizeof(float));
    for (size_t row = 0; row < 4; ++row) {
      memcpy((char *)storage + view_byte_offset + row * (size_t)view_row_stride * sizeof(float), base + row * 5, 5 * sizeof(float));
    }
    free(active_index->dataset); active_index->dataset = storage;
  }
  int released = dataset_view_deletes;
  assert(seekdb_cuvs_copy_dataset(h, restored, 4, 5) == 0);
  assert(!memcmp(restored, base, sizeof(base)) && !dataset_views && dataset_view_deletes == released + 1);
  assert(active_index->dataset); /* Metadata release must not destroy borrowed device storage. */
  rocm_view_device = 1;
  assert(seekdb_cuvs_copy_dataset(h, restored, 4, 5) == 0);
  assert(!memcmp(restored, base, sizeof(base)) && !dataset_views);
  rocm_view_device = 0;
  if (!no_strides) {
    for (int v = V_N; v <= V_ADDRESS_OVERFLOW; ++v) {
      view_fault = (enum ViewFault)v; released = dataset_view_deletes;
      int before = copies;
      assert(seekdb_cuvs_copy_dataset(h, restored, 4, 5) != 0);
      assert(copies == before && !dataset_views && dataset_view_deletes == released + 1);
    }
  }
  view_fault = V_VALID;
  const enum Fault errors[] = {F_GET_DATASET, F_SYNC, F_COPY};
  for (size_t j = 0; j < sizeof(errors) / sizeof(errors[0]); ++j) {
    fault = errors[j]; released = dataset_view_deletes;
    assert(seekdb_cuvs_copy_dataset(h, restored, 4, 5) != 0);
    assert(!dataset_views && dataset_view_deletes == released + 1);
  }
  fault = F_NONE;
  if (padded) {
    copy_fail_at = copies + 2; released = dataset_view_deletes;
    assert(seekdb_cuvs_copy_dataset(h, restored, 4, 5) != 0);
    assert(!dataset_views && dataset_view_deletes == released + 1);
    copy_fail_at = 0;
  }
  released = dataset_view_deletes;
  assert(seekdb_cuvs_copy_dataset(NULL, restored, 4, 5) != 0);
  assert(seekdb_cuvs_copy_dataset(h, NULL, 4, 5) != 0);
  assert(seekdb_cuvs_copy_dataset(h, restored, 0, 5) != 0);
  assert(seekdb_cuvs_copy_dataset(h, restored, 4, -1) != 0);
  assert(seekdb_cuvs_copy_dataset(h, restored, 5, 5) != 0);
  assert(seekdb_cuvs_copy_dataset(h, restored, 4, 6) != 0);
  assert(seekdb_cuvs_copy_dataset(h, restored, LONG_MAX, LONG_MAX) != 0);
  /* Exercise the multiplication guard with matching handle dimensions. */
  ((sk_cuvs_index *)h)->n = LONG_MAX; ((sk_cuvs_index *)h)->dim = LONG_MAX;
  assert(seekdb_cuvs_copy_dataset(h, restored, LONG_MAX, LONG_MAX) != 0);
  ((sk_cuvs_index *)h)->n = 4; ((sk_cuvs_index *)h)->dim = 5;
  assert(dataset_view_deletes == released);
  seekdb_cuvs_free(h); no_live_resources(); expected = NULL;
}
int main(void)
{
  success_case(128); success_case(256); success_case(129);
  dataset_restore_cases(0, 0, 0); dataset_restore_cases(1, 1, 0);
  dataset_restore_cases(0, 1, 1);
  float base[4 * 128] = {0}; expected = base; expected_n = 4; expected_dim = 128;
  for (int f = F_HOST_CALLOC; f <= F_BUILD; ++f) {
    fault = (enum Fault)f; reset_order();
    assert(!seekdb_cuvs_build(base, 4, 128)); no_live_resources();
  }
  fault = F_LOAD; reset_order();
  assert(!seekdb_cuvs_deserialize("fake.snapshot", 4, 128)); no_live_resources();
  fault = F_NONE; reset_order();
  int a = allocations, b = build_calls;
  assert(!seekdb_cuvs_build(NULL, 4, 128));
  assert(!seekdb_cuvs_build(base, 0, 128));
  assert(!seekdb_cuvs_build(base, 4, -1));
  assert(!seekdb_cuvs_build(base, LONG_MAX, 128));
  assert(!seekdb_cuvs_build(base, 4, LONG_MAX));
  assert(!seekdb_cuvs_deserialize(NULL, 4, 128));
  seekdb_cuvs_free(NULL);
  assert(allocations == a && build_calls == b); no_live_resources();
  printf("PASS CPU fake ownership: aligned128/256, unaligned129, build failures, reload, overflow; alloc/free=%d/%d build=%d serialize=%d load=%d sync=%d\n",
         allocations, frees, build_calls, serialize_calls, load_calls, sync_calls);
  return 0;
}
