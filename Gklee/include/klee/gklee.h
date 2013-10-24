#ifndef __GKLEE_H__
#define __GKLEE_H__

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __HOST_DEFINES_H__
// Variable and function qualifiers...
#define __global__ __attribute__((global)) __attribute__ ((noinline))
#define __host__ __attribute__((host)) 
#define __device__ __attribute__((device)) __attribute__ ((section("__device__")))
#define __constant__ __attribute__((constant)) __attribute__ ((section("__constant__"))) static 
#define __extern__shared__ __attribute__((shared)) __attribute__((section("__shared__"))) extern 
#define __shared__ __attribute__((shared)) __attribute__((section("__shared__"))) static

#endif

#include <vector_types.h>

extern dim3 gridDim;
extern dim3 blockDim;
extern dim3 blockIdx;
extern dim3 threadIdx;

extern void __set_CUDAConfig(dim3, dim3, ...);

#ifdef __cplusplus
}
#endif

#endif
