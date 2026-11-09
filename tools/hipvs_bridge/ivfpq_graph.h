#pragma once
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifndef SEEKDB_IVFPQ_INTERMEDIATE
#define SEEKDB_IVFPQ_INTERMEDIATE 256
#endif
/* At least 256 samples for an 8-bit PQ codebook, and enough distinct vertices
 * for intermediate_degree non-self neighbors plus its extra refine result. */
#define SEEKDB_IVFPQ_MIN_ROWS ((SEEKDB_IVFPQ_INTERMEDIATE + 2 > 256) ? SEEKDB_IVFPQ_INTERMEDIATE + 2 : 256)
#ifdef __cplusplus
extern "C" {
#endif
/* On success graph_storage is a borrowed CAGRA graph owned by the caller.
 * Destroy the CAGRA index before cudaFree(graph_storage). */
int seekdb_ivfpq_gpu_graph(cuvsResources_t res, cuvsCagraIndexParams_t params,
                          DLManagedTensor* dataset, cuvsCagraIndex_t index,
                          unsigned int** graph_storage);
#ifdef __cplusplus
}
#endif
