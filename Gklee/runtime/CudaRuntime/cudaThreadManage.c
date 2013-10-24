//===-- cudaThreadManage.c ---------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <cuda/driver_types.h>

cudaError_t cudaThreadExit(void) {
  return cudaSuccess;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
  return cudaSuccess;
}

cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
  return cudaSuccess;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
  return cudaSuccess;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
  return cudaSuccess;
}

cudaError_t cudaThreadSynchronize(void) {
  return cudaSuccess;
}  	
