#pragma once
#include <cuvs/core/c_api.h>
typedef struct {
  int build_algo;
  size_t graph_degree, intermediate_graph_degree, nn_descent_niter;
} *cuvsCagraIndexParams_t;
typedef struct {
  uintptr_t addr;
  int64_t n, dim, degree;
  float *dataset;
  int borrowed;
  unsigned int *graph;
  int borrowed_graph;
} *cuvsCagraIndex_t;
typedef struct { size_t itopk_size, search_width; } *cuvsCagraSearchParams_t;
typedef struct { int type; uintptr_t addr; } cuvsFilter;
enum { NN_DESCENT = 2, NO_FILTER = 0, L2Expanded = 0 };
#ifdef __cplusplus
extern "C" {
#endif
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t *);
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t);
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t *);
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t);
cuvsError_t cuvsCagraBuild(cuvsResources_t, cuvsCagraIndexParams_t, DLManagedTensor *, cuvsCagraIndex_t);
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t *);
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t);
cuvsError_t cuvsCagraSearch(cuvsResources_t, cuvsCagraSearchParams_t, cuvsCagraIndex_t,
                          DLManagedTensor *, DLManagedTensor *, DLManagedTensor *, cuvsFilter);
cuvsError_t cuvsCagraSerialize(cuvsResources_t, const char *, cuvsCagraIndex_t, bool);
cuvsError_t cuvsCagraDeserialize(cuvsResources_t, const char *, cuvsCagraIndex_t);
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t, int64_t *);
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t, int64_t *);
cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t, int64_t *);
cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t, DLManagedTensor *);
cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t, DLManagedTensor *);
cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t, int, DLManagedTensor *, DLManagedTensor *, cuvsCagraIndex_t);
#ifdef __cplusplus
}
#endif
