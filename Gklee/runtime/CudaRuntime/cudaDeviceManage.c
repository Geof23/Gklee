//===-- cudaDeviceManage.c ------------------------------------------------===//
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

cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
  *device = 0;
  return cudaSuccess;
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId) {
  *device = 0;
  return cudaSuccess;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
  return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
  return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
  memcpy(pciBusId, "id0", 4);
  return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
  return cudaSuccess;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
  return cudaSuccess;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
  *device = 0;
  return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count) {
  *count = 1;
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  strcpy(prop->name, "GPU device");
  return cudaSuccess;
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
  return cudaSuccess;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
  return cudaSuccess;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
  return cudaSuccess;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
  return cudaSuccess;
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  return cudaSuccess;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
  return cudaSuccess; 
}

cudaError_t cudaSetValidDevices(int *device_arr, int len) {
  return cudaSuccess;
}
