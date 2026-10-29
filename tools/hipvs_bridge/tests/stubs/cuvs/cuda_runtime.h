#pragma once
#include <stddef.h>
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDefault = 4 };
#ifdef __cplusplus
extern "C" {
#endif
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaFree(void *);
cudaError_t cudaMemcpy(void *, const void *, size_t, int);
#ifdef __cplusplus
}
#endif
