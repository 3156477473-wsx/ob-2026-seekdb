#define _POSIX_C_SOURCE 200809L
/* Linux LP64 CPU fake: exercises the actual candidate C file, never HIP/ANN. */
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <dirent.h>
#if LONG_MAX != INT64_MAX
#error This test requires Linux LP64; DLPack shape is int64_t, not a Windows long shim.
#endif

enum Fault { F_NONE, F_HOST_CALLOC, F_RESOURCE, F_MALLOC, F_COPY, F_PARAMS,
             F_INDEX, F_BUILD, F_LOAD, F_GET_DATASET, F_SYNC,
             F_GRAPH_ALLOC, F_GRAPH_WORK, F_GRAPH_PUBLISHED,
             F_FROM_ARGS, F_SERIALIZE, F_OLD_DESTROY };
static enum Fault fault;
static char test_directory[4096];
static const char* test_path(const char* name)
{
  static char path[4096];
  int size = snprintf(path, sizeof(path), "%s/%s", test_directory, name);
  assert(size > 0 && (size_t)size < sizeof(path));
  return path;
}
static void *bridge_calloc(size_t n, size_t bytes)
{
  return fault == F_HOST_CALLOC ? NULL : calloc(n, bytes);
}
#define calloc bridge_calloc
#include "../seekdb_cuvs_bridge.c"
#undef calloc

static int resources, params, indexes, owned_datasets, owned_graphs, allocations, frees, copies;
static int build_calls, serialize_calls, load_calls, sync_calls, last_device;
static int gpu_graph_calls, graph_was_published;
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
static unsigned int *device_graph;
static size_t device_bytes;
static cuvsCagraIndex_t active_index, secondary_index;
static const float *expected;
static int64_t expected_n, expected_dim;

static void no_live_resources(void)
{
  assert(!resources && !params && !indexes && !owned_datasets && !owned_graphs && !secondary_index && !device_input && !device_graph && !dataset_views);
  assert(allocations == frees);
}
cudaError_t cudaMalloc(void **ptr, size_t bytes)
{
  assert(bytes && (!device_input || !device_graph));
  if (fault == F_MALLOC) { *ptr = NULL; return cudaErrorUnknown; }
  *ptr = malloc(bytes); assert(*ptr);
  if (!device_input) { device_input = *ptr; device_bytes = bytes; }
  else device_graph = (unsigned int *)*ptr;
  ++allocations; return cudaSuccess;
}
cudaError_t cudaFree(void *ptr)
{
  assert(ptr && (ptr == device_input || ptr == device_graph));
  /* Index must stop borrowing this buffer before the bridge frees it. */
  assert(!active_index || ((void *)active_index->dataset != ptr && (void *)active_index->graph != ptr));
  assert(!secondary_index || ((void *)secondary_index->dataset != ptr && (void *)secondary_index->graph != ptr));
  if (last_build) {
    assert(last_sync > last_build);
    if (ptr == device_input || graph_was_published) assert(last_index_destroy > last_build);
  }
  if (ptr == device_graph) { free(ptr); device_graph = NULL; }
  else { free(ptr); device_input = NULL; device_bytes = 0; }
  ++frees; return cudaSuccess;
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, int kind)
{
  assert((kind == cudaMemcpyDefault || kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyHostToDevice) && dst && src); ++copies;
  if (fault == F_COPY || copies == copy_fail_at) return cudaErrorUnknown;
  if (dst == device_input) assert(bytes <= device_bytes);
  memcpy(dst, src, bytes); return cudaSuccess;
}
cudaError_t cudaGetDevice(int *device)
{ *device = 0; return cudaSuccess; }
cudaError_t cudaDeviceSynchronize(void)
{
  ++sync_calls; last_sync = ++operation_clock;
  return fault == F_SYNC ? cudaErrorUnknown : cudaSuccess;
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
  assert(!secondary_index);
  *p = (cuvsCagraIndex_t)calloc(1, sizeof(**p)); assert(*p);
  if (active_index) secondary_index = *p; else active_index = *p;
  ++indexes; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t p)
{
  assert(p == active_index || p == secondary_index);
  int destroyed_old_error = fault == F_OLD_DESTROY && p == active_index && secondary_index;
  if (last_build) assert(last_sync > last_build);
  if (p->borrowed_graph) assert(p->graph == device_graph);
  else if (p->graph) { free(p->graph); --owned_graphs; }
  if (p->borrowed) assert(p->dataset == device_input);
  else if (p->dataset) { free(p->dataset); --owned_datasets; }
  if (p == active_index) { active_index = secondary_index; secondary_index = NULL; }
  else secondary_index = NULL;
  free(p); --indexes;
  last_index_destroy = ++operation_clock;
  return destroyed_old_error ? CUVS_ERROR : CUVS_SUCCESS;
}
static void retain_dataset(cuvsResources_t r, cuvsCagraIndexParams_t p,
                           DLManagedTensor *t, cuvsCagraIndex_t i)
{
  assert(r && p && i == active_index && t);
  assert(p->build_algo == NN_DESCENT && p->graph_degree == 128 &&
         p->intermediate_graph_degree == 160 && p->nn_descent_niter == 100);
  DLTensor *d = &t->dl_tensor;
  assert(d->ndim == 2 && d->shape[0] == expected_n && d->shape[1] == expected_dim);
  assert(d->dtype.code == kDLFloat && d->dtype.bits == 32 && d->dtype.lanes == 1);
  assert(!d->strides && !d->byte_offset);
  size_t bytes = (size_t)expected_n * (size_t)expected_dim * sizeof(float);
  last_device = d->device.device_type;
  assert(!memcmp(d->data, expected, bytes));
  i->n = expected_n; i->dim = expected_dim; i->degree = 128;
  if (d->device.device_type == kDLCUDA) {
    assert(d->data == device_input); i->dataset = (float *)d->data; i->borrowed = 1;
  } else {
    assert(d->device.device_type == kDLCPU && d->data == expected && !device_input);
    i->dataset = (float *)malloc(bytes); assert(i->dataset);
    memcpy(i->dataset, d->data, bytes); ++owned_datasets;
  }
  /* Even the failing backend has retained a reference, exposing cleanup order. */
  last_build = ++operation_clock;
}
cuvsError_t cuvsCagraBuild(cuvsResources_t r, cuvsCagraIndexParams_t p,
                         DLManagedTensor *t, cuvsCagraIndex_t i)
{
  ++build_calls;
  retain_dataset(r, p, t, i);
  i->graph = (unsigned int *)malloc((size_t)i->n * (size_t)i->degree * 4); assert(i->graph);
  for (int64_t j = 0; j < i->n * i->degree; ++j) i->graph[j] = (unsigned int)(j % i->n);
  ++owned_graphs;
  return fault == F_BUILD ? CUVS_ERROR : CUVS_SUCCESS;
}
/* Only an ownership fake: the actual GPU helper is never built or linked into
 * these CPU tests. Distinct counters reject accidental native-path dispatch. */
int seekdb_ivfpq_gpu_graph(cuvsResources_t r, cuvsCagraIndexParams_t p,
                          DLManagedTensor *t, cuvsCagraIndex_t i,
                          unsigned int **storage)
{
  assert(r && p && t && i == active_index && storage && !*storage && !device_graph);
  assert(t->dl_tensor.shape[0] >= SEEKDB_IVFPQ_MIN_ROWS);
  ++gpu_graph_calls; last_build = ++operation_clock;
  if (fault == F_GRAPH_ALLOC) return 1;
  device_graph = (unsigned int *)malloc((size_t)t->dl_tensor.shape[0] * 128 * sizeof(unsigned int));
  assert(device_graph); ++allocations;
  for (int64_t j = 0; j < expected_n * 128; ++j) device_graph[j] = (unsigned int)(j % expected_n);
  if (fault == F_GRAPH_WORK) {
    /* Unpublished work is fenced and released inside the helper. */
    assert(cuvsStreamSync(r) == CUVS_SUCCESS);
    assert(cudaFree(device_graph) == cudaSuccess);
    return 1;
  }
  retain_dataset(r, p, t, i);
  i->graph = device_graph; i->borrowed_graph = 1;
  *storage = device_graph; graph_was_published = 1;
  /* Includes failure after FromArgs publishes borrowed graph/data references. */
  return fault == F_BUILD || fault == F_GRAPH_PUBLISHED;
}
cuvsError_t cuvsCagraSerialize(cuvsResources_t r, const char *path,
                             cuvsCagraIndex_t i, bool include_dataset)
{
  assert(r && path && i == active_index && include_dataset && i->dataset);
  assert(!memcmp(i->dataset, expected, (size_t)i->n * (size_t)i->dim * sizeof(float)));
  ++serialize_calls; return fault == F_SERIALIZE ? CUVS_ERROR : CUVS_SUCCESS;
}
cuvsError_t cuvsCagraDeserialize(cuvsResources_t r, const char *path, cuvsCagraIndex_t i)
{
  assert(r && path && i == active_index); ++load_calls;
  if (fault == F_LOAD) return CUVS_ERROR;
  size_t bytes = (size_t)expected_n * (size_t)expected_dim * sizeof(float);
  i->n = expected_n; i->dim = expected_dim;
  i->dataset = (float *)malloc(bytes); assert(i->dataset);
  memcpy(i->dataset, expected, bytes); ++owned_datasets;
  i->degree = 3; /* Deliberately not the competition graph degree. */
  i->graph = (unsigned int *)malloc((size_t)i->n * (size_t)i->degree * 4); assert(i->graph);
  for (int64_t j = 0; j < i->n * i->degree; ++j) i->graph[j] = (unsigned int)(j % i->n);
  ++owned_graphs; return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t i, int64_t *n)
{ *n = i->n; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t i, int64_t *dim)
{ *dim = i->dim; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t i, int64_t *degree)
{ *degree = i->degree; return CUVS_SUCCESS; }
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
  m->strides[0] = view_row_stride ? view_row_stride : i->dim; m->strides[1] = 1;
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
cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t i, DLManagedTensor *t)
{
  assert(i == active_index && t && !t->deleter && i->graph);
  DatasetViewMetadata *m = (DatasetViewMetadata *)malloc(sizeof(*m)); assert(m);
  m->shape[0] = i->n; m->shape[1] = i->degree;
  m->strides[0] = i->degree; m->strides[1] = 1;
  t->manager_ctx = m; t->deleter = delete_dataset_view; ++dataset_views;
  t->dl_tensor.data = i->graph;
  t->dl_tensor.device.device_type = kDLCUDA; t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2; t->dl_tensor.shape = m->shape; t->dl_tensor.strides = m->strides;
  t->dl_tensor.dtype.code = kDLUInt; t->dl_tensor.dtype.bits = 32; t->dl_tensor.dtype.lanes = 1;
  return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t r, int metric, DLManagedTensor *g,
                                 DLManagedTensor *d, cuvsCagraIndex_t i)
{
  assert(r && metric == L2Expanded && i == active_index && !secondary_index && !i->dataset && !i->graph);
  assert(!memcmp(d->dl_tensor.data, expected, (size_t)expected_n * (size_t)expected_dim * 4));
  assert(!d->dl_tensor.strides && !g->dl_tensor.strides && !d->dl_tensor.byte_offset && !g->dl_tensor.byte_offset);
  i->n = d->dl_tensor.shape[0]; i->dim = d->dl_tensor.shape[1]; i->degree = g->dl_tensor.shape[1];
  i->dataset = (float *)d->dl_tensor.data; i->graph = (unsigned int *)g->dl_tensor.data;
  i->borrowed = i->borrowed_graph = 1; last_build = ++operation_clock;
  i->addr = (uintptr_t)i;
  return fault == F_FROM_ARGS ? CUVS_ERROR : CUVS_SUCCESS;
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
{ last_build = last_sync = last_index_destroy = operation_clock = graph_was_published = 0; }
static void success_case(long n, long dim)
{
  expected_n = n; expected_dim = dim; fault = F_NONE; reset_order();
  float *base = (float *)malloc((size_t)expected_n * (size_t)dim * sizeof(float)); assert(base);
  for (int64_t j = 0; j < expected_n * dim; ++j) base[j] = (float)j * 0.125f;
  expected = base;
  int a = allocations, c = copies, native_before = build_calls, gpu_before = gpu_graph_calls;
  void *h = seekdb_cuvs_build(base, (long)expected_n, dim); assert(h);
  assert(last_device == ((dim % 4) ? kDLCPU : kDLCUDA));
  assert(allocations - a == ((dim % 4) ? 0 : 1) + (n >= SEEKDB_IVFPQ_MIN_ROWS));
  assert(copies - c == ((dim % 4) ? 0 : 1));
  if (!(dim % 4)) assert(((sk_cuvs_index *)h)->base_device == device_input);
  assert(build_calls - native_before == (n < SEEKDB_IVFPQ_MIN_ROWS));
  assert(gpu_graph_calls - gpu_before == (n >= SEEKDB_IVFPQ_MIN_ROWS));
  assert(((sk_cuvs_index *)h)->graph_device == device_graph);
  assert(!!device_graph == (n >= SEEKDB_IVFPQ_MIN_ROWS));
  assert(seekdb_cuvs_serialize(h, test_path("fake.snapshot")) == 0);
  seekdb_cuvs_free(h); no_live_resources();

  /* Loaded aligned indexes acquire bridge ownership only when serialized. */
  reset_order(); a = allocations; c = build_calls;
  h = seekdb_cuvs_deserialize(test_path("fake.snapshot"), (long)expected_n, dim); assert(h);
  assert(((sk_cuvs_index *)h)->base_device == device_input && ((sk_cuvs_index *)h)->graph_device == device_graph && allocations == a + 2 && build_calls == c);
  assert(seekdb_cuvs_serialize(h, test_path("fake.loaded")) == 0);
  assert(((sk_cuvs_index *)h)->base_device == device_input);
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
  void *h = seekdb_cuvs_deserialize(test_path("fake.dataset"), 4, 5); assert(h);
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

/* Real files, exact graph/FP32 bytes, legacy padded-view conversion, and
 * corrupt/truncated/extra payloads: no mocked serialization can make this pass. */
static void stream_roundtrip_case(void)
{
  float base[4 * 5], restored[4 * 5];
  for (size_t j = 0; j < 20; ++j) {
    uint32_t bits = j == 0 ? UINT32_C(0x80000000) : j == 1 ? UINT32_C(0x7fc00123) : UINT32_C(0x3f800000) + (uint32_t)j;
    memcpy(base + j, &bits, 4);
  }
  expected = base; expected_n = 4; expected_dim = 5;
  fault = F_NONE; view_fault = V_VALID; reset_order();
  sk_cuvs_index *h = (sk_cuvs_index *)seekdb_cuvs_deserialize(test_path("fake.legacy"), 4, 5); assert(h);
  float *padded = (float *)malloc(16 + 4 * 8 * 4); assert(padded);
  memset(padded, 0xa5, 16 + 4 * 8 * 4);
  for (size_t row = 0; row < 4; ++row) memcpy((char *)padded + 16 + row * 8 * 4, base + row * 5, 5 * 4);
  free(active_index->dataset); active_index->dataset = padded;
  view_row_stride = 8; view_byte_offset = 16; null_view_strides = 0;
  int sdk_calls = serialize_calls;
  assert(seekdb_cuvs_serialize(h, test_path("stream.first")) == 0 && serialize_calls == sdk_calls);
  seekdb_cuvs_free(h); no_live_resources();
  view_row_stride = 0; view_byte_offset = 0;
  h = (sk_cuvs_index *)seekdb_cuvs_deserialize(test_path("stream.first"), 4, 5); assert(h);
  assert(h->base_device == device_input && h->graph_device == device_graph);
  assert(seekdb_cuvs_copy_dataset(h, restored, 4, 5) == 0 && !memcmp(base, restored, sizeof(base)));
  assert(active_index->degree == 3);
  for (size_t j = 0; j < 12; ++j) assert(active_index->graph[j] == j % 4);
  assert(seekdb_cuvs_serialize(h, test_path("stream.second")) == 0 && serialize_calls == sdk_calls);
  seekdb_cuvs_free(h); no_live_resources();
  unsigned char first[192], second[192]; /*64+4*3*4+4*5*4*/
  FILE *file = fopen(test_path("stream.first"), "rb"); assert(file);
  assert(fread(first, 1, sizeof(first), file) == sizeof(first) && fgetc(file) == EOF && !ferror(file) && fclose(file) == 0);
  file = fopen(test_path("stream.second"), "rb"); assert(file);
  assert(fread(second, 1, sizeof(second), file) == sizeof(second) && fgetc(file) == EOF && !ferror(file) && fclose(file) == 0);
  assert(!memcmp(first, second, sizeof(first)) && !memcmp(first, "SKCAGR04", 8));
  for (int damage = 0; damage < 3; ++damage) {
    file = fopen(test_path("stream.bad"), "wb"); assert(file);
    unsigned char damaged[192]; memcpy(damaged, first, sizeof(first));
    if (damage == 0) damaged[sizeof(damaged) - 1] ^= 1;
    const size_t bytes = damage == 1 ? sizeof(damaged) - 4 : sizeof(damaged);
    assert(fwrite(damaged, 1, bytes, file) == bytes);
    if (damage == 2) assert(fputc(0, file) == 0);
    assert(fclose(file) == 0);
    reset_order(); assert(!seekdb_cuvs_deserialize(test_path("stream.bad"), 4, 5)); no_live_resources();
  }
  /* Allocation/copy cleanup, and FromArgs retaining buffers before failure. */
  const enum Fault failures[] = {F_MALLOC, F_COPY, F_FROM_ARGS};
  for (size_t j = 0; j < sizeof(failures) / sizeof(failures[0]); ++j) {
    fault = failures[j]; reset_order();
    assert(!seekdb_cuvs_deserialize(test_path("stream.first"), 4, 5)); no_live_resources();
  }
  fault = F_NONE; expected = NULL;
}
int main(void)
{
  const char *temporary = getenv("TMPDIR");
  assert(temporary && temporary[0] == '/');
  int path_size = snprintf(test_directory, sizeof(test_directory), "%s/seekdb-stream-XXXXXX", temporary);
  assert(path_size > 0 && (size_t)path_size < sizeof(test_directory) && mkdtemp(test_directory));
  const char *legacy_names[] = {"fake.dataset", "fake.legacy"};
  for (size_t j = 0; j < 2; ++j) {
    FILE *file = fopen(test_path(legacy_names[j]), "wb"); assert(file);
    assert(fwrite("LEGACY00", 1, 8, file) == 8 && fclose(file) == 0);
  }

  success_case(4, 128); success_case(4, 256); success_case(4, 129);
  success_case(SEEKDB_IVFPQ_MIN_ROWS - 1, 128);
  success_case(SEEKDB_IVFPQ_MIN_ROWS, 128);
  success_case(256, 256); success_case(256, 129);
  dataset_restore_cases(0, 0, 0); dataset_restore_cases(1, 1, 0);
  dataset_restore_cases(0, 1, 1);
  stream_roundtrip_case();
  float base[4 * 128] = {0}; expected = base; expected_n = 4; expected_dim = 128;
  for (int f = F_HOST_CALLOC; f <= F_BUILD; ++f) {
    fault = (enum Fault)f; reset_order();
    assert(!seekdb_cuvs_build(base, 4, 128)); no_live_resources();
  }
  float *gpu_base = (float *)calloc((size_t)256 * 128, sizeof(float)); assert(gpu_base);
  expected = gpu_base; expected_n = 256; expected_dim = 128;
  for (int f = F_HOST_CALLOC; f <= F_BUILD; ++f) {
    fault = (enum Fault)f; reset_order();
    assert(!seekdb_cuvs_build(gpu_base, 256, 128)); no_live_resources();
  }
  const enum Fault graph_faults[] = {F_GRAPH_ALLOC, F_GRAPH_WORK, F_GRAPH_PUBLISHED};
  for (size_t j = 0; j < sizeof(graph_faults) / sizeof(graph_faults[0]); ++j) {
    fault = graph_faults[j]; reset_order();
    int before = gpu_graph_calls;
    assert(!seekdb_cuvs_build(gpu_base, 256, 128));
    assert(gpu_graph_calls == before + 1); no_live_resources();
  }
  free(gpu_base); expected = base; expected_n = 4; expected_dim = 128;
  fault = F_LOAD; reset_order();
  assert(!seekdb_cuvs_deserialize(test_path("fake.legacy"), 4, 128)); no_live_resources();
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
  printf("PASS CPU fake ownership: native/custom dispatch, aligned128/256, unaligned129, unpublished/published graph failures, reload, overflow; alloc/free=%d/%d native=%d custom=%d serialize=%d load=%d sync=%d\n",
         allocations, frees, build_calls, gpu_graph_calls, serialize_calls, load_calls, sync_calls);
  DIR *directory = opendir(test_directory); assert(directory);
  struct dirent *entry;
  while ((entry = readdir(directory)))
    if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) assert(unlink(test_path(entry->d_name)) == 0);
  assert(closedir(directory) == 0 && rmdir(test_directory) == 0);
  return 0;
}
