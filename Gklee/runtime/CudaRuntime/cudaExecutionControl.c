//===-- cudaEventControl.c ------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <cuda/driver_types.h>

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func) {
  return cudaSuccess;
}

cudaError_t cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) {
  return cudaSuccess;
}

cudaError_t cudaLaunch(const char *entry) {
  return cudaSuccess;
}

cudaError_t cudaSetDoubleForDevice(double *d) {
  return cudaSuccess;
}

cudaError_t cudaSetDoubleForHost(double *d) {
  return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  return cudaSuccess;
}
