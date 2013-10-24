#ifndef __CUTIL_H__
#define __CUTIL_H__

#include <stdlib.h>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

// variable modifiers
#define __shared__ __attribute((section ("__shared__"))) static
#define __device__ __attribute((section ("__device__"))) static
#define __constant__ __attribute((section ("__const__"))) static

// function modifiers
#define __global__ __attribute((noinline))
#define __kernel__ __attribute((noinline))
#define __host__ __attribute((noinline))

// extra modifiers
#define __input__ static

// default types
typedef struct { unsigned int x, y; } uint2;
typedef struct { unsigned int x, y, z; } uint3;
typedef struct { long x, y, z; } long3;
typedef struct { unsigned long x, y, z; } ulong3;
typedef struct { short x, y, z, w; } short4;
typedef struct { unsigned short x, y, z, w; } ushort4;
typedef struct { int x, y, z, w; } int4;
typedef struct { unsigned int x, y, z, w; } uint4;
typedef struct { long x, y, z, w; } long4;
typedef struct { unsigned long x, y, z, w; } ulong4;

typedef uint3 dim3;
typedef uint2 dim2;

// GPU configuration variables
dim2 gridDim = {1,1};
__shared__ dim2 blockIdx;
dim3 blockDim = {2,1,1};
dim3 threadIdx;

// some commonly used macros
#define atomicAdd(x,y) x + y
#define __mul24(x,y) x * y
#define __umul24(x,y) x * y
extern uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);

// GPU interface functions
#define cutilSafeCall(f) f

void cudaMalloc(void** devPtr, size_t size) {
  *devPtr = malloc(size);
}
#define cudaFree delete
/* void cudaFree(void* devPtr) { */
/*   delete devPtr; */
/* } */

enum HDType {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost};
void cudaMemcpy(void* a, void* b, size_t size, ...) {
  memcpy(a,b,size);
};

// GPU functions
extern void __syncthreads();
#define _bar __syncthreads

extern void __begin_GPU(...);
extern void __end_GPU();

#ifdef __cplusplus
}
#endif

#endif
