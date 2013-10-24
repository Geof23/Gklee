//===-- cudaTextureManage.c -----------------------------------------------===//
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

cudaError_t cudaBindTexture(size_t * offset,
		            const struct textureReference * texref,
		            const void * devPtr,
		            const struct cudaChannelFormatDesc * desc,
		            size_t size) {
  texref = devPtr;
  return cudaSuccess;
} 	

cudaError_t cudaBindTexture2D(size_t * 	offset,
		              const struct textureReference * texref,
		              const void * devPtr,
		              const struct cudaChannelFormatDesc * desc,
		              size_t width,
		              size_t height,
		              size_t pitch) {
  return cudaSuccess;
} 

cudaError_t cudaBindTextureToArray(const struct textureReference * texref,
		                   const struct cudaArray * array,
		                   const struct cudaChannelFormatDesc * desc) {
  return cudaSuccess;
} 	

struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                                   enum cudaChannelFormatKind f) {
  struct cudaChannelFormatDesc desc;
  desc.x = x;
  desc.y = y;
  desc.z = z;
  desc.w = w;
  desc.f = f;
  return desc;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc,
		               const struct cudaArray * array) {
  return cudaSuccess;
}

cudaError_t cudaGetTextureAlignmentOffset(size_t * offset,
		                          const struct textureReference * texref) {
  return cudaSuccess;
} 	

cudaError_t cudaUnbindTexture(const struct textureReference * texref) {
  return cudaSuccess;
}  	
