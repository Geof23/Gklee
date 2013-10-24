//===-- cudaEventManage.c ------------------------------------------------===//
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

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return cudaSuccess;
} 

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return cudaSuccess;
}

/*cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0) {
  return cudaSuccess;
}*/

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return cudaSuccess;
}
