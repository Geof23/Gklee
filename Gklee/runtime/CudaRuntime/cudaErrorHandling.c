//===-- cudaErrorHandling.c -----------------------------------------------===//
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

static char msg[25];

const char* cudaGetErrorString(cudaError_t error) {
  char tmp[20] = "No Error Message"; 
  memcpy(msg, tmp, strlen(tmp)+1);
  return msg;
}

cudaError_t cudaGetLastError(void) {
  return cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void) {
  return cudaSuccess;
}
