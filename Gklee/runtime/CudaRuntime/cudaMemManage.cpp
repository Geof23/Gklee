//===-- cudaMemManage.c ---------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda/driver_types.h>

void __set_device();
void __clear_device();
void __set_host();
void __clear_host();

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, 
                             unsigned int *flags, struct cudaArray *array) {
  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  free(devPtr);
  return cudaSuccess;
}

cudaError_t cudaFreeArray(struct cudaArray *array) {
  free(array);
  return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr) {
  free(ptr);
  return cudaSuccess;
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol) {
  return cudaSuccess;
}

cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol) {
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  return cudaSuccess;
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
  return cudaSuccess;
}

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
  return cudaSuccess;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
  return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *ptr) {
  return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  __set_device();
  *devPtr = (void *)malloc(size);
  __clear_device();

  return cudaSuccess;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent) {
  return cudaSuccess;
}

cudaError_t cudaMalloc3DArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, 
                              struct cudaExtent extent, unsigned int flags=0) {
  return cudaSuccess;
}

cudaError_t cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, 
                            size_t width, size_t height=0, unsigned int flags=0) {
  return cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
  __set_host();
  *ptr = (void *)malloc(size);
  __clear_host();

  return cudaSuccess;
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  memcpy(dst, src, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, 
                         size_t width, size_t height, enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, 
                                     const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, 
                                     size_t width, size_t height, enum cudaMemcpyKind kind=cudaMemcpyDeviceToDevice) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, 
                              size_t width, size_t height, enum cudaMemcpyKind kind, 
                              cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, 
                                  size_t wOffset, size_t hOffset, size_t width, size_t height, 
                                  enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, 
                                       size_t wOffset, size_t hOffset, size_t width, size_t height, 
                                       enum cudaMemcpyKind kind, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, 
                                const void *src, size_t spitch, size_t width, size_t height, 
                                enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, 
                                     const void *src, size_t spitch, size_t width, size_t height, 
                                     enum cudaMemcpyKind kind, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
  return cudaSuccess;
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, 
                                   const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, 
                                   size_t count, enum cudaMemcpyKind kind=cudaMemcpyDeviceToDevice) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, 
                            enum cudaMemcpyKind kind, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, 
                                size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, 
                                     size_t hOffset, size_t count, enum cudaMemcpyKind kind, 
                                     cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, 
                                 size_t offset=0, enum cudaMemcpyKind kind=cudaMemcpyDeviceToHost) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, 
                                      size_t offset, enum cudaMemcpyKind kind, 
                                      cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
  memcpy(dst, src, count);
  return cudaSuccess;
}
 
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, 
                                size_t count, cudaStream_t stream=0) {
  memcpy(dst, src, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, 
                              const void *src, size_t count, enum cudaMemcpyKind kind) {
  return cudaSuccess;
} 

cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, 
                                   const void *src, size_t count, enum cudaMemcpyKind kind, 
                                   cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(char *symbol, const void *src, size_t count, 
                               size_t offset=0, enum cudaMemcpyKind kind=cudaMemcpyHostToDevice) {
  memcpy(symbol+offset, src, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, 
                                    enum cudaMemcpyKind kind, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  memset(devPtr, value, count);
  return cudaSuccess;
}

cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, 
                         size_t width, size_t height) {
  return cudaSuccess;
}

cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, 
                              size_t width, size_t height, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, 
                         struct cudaExtent extent) {
  return cudaSuccess;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, 
                              struct cudaExtent extent, cudaStream_t stream=0) {
  return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream=0) {
  return cudaSuccess;
}

#ifdef __cplusplus
}
#endif
