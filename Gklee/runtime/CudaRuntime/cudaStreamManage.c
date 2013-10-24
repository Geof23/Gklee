//===-- cudaStreamManage.c ------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <string.h>
#include <cuda/driver_types.h>

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  return cudaSuccess;
} 

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize (cudaStream_t stream) {
  return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  return cudaSuccess;
}
