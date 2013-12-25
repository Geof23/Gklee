/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
//#include <cutil_inline.h>
#include <assert.h>
#include <stdio.h>
#include "mergeSort_common.h"


inline __device__ void Comparator(
    uint& keyA,
    uint& valA,
    uint& keyB,
    uint& valB,
    uint arrowDir
){
    uint t;
    if( (keyA > keyB) == arrowDir ){
        t = keyA; keyA = keyB; keyB = t;
        t = valA; valA = valB; valB = t;
    }
}

__global__ void bitonicSortSharedKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint sortDir
){
    //Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for(uint size = 2; size < arrayLength; size <<= 1){
        //Bitonic merge
        uint dir = (threadIdx.x & (size / 2)) != 0;
        for(uint stride = size / 2; stride > 0; stride >>= 1){
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
            );
        }
    }

    //ddd == sortDir for the last bitonic merge step
    {
        for(uint stride = arrayLength / 2; stride > 0; stride >>= 1){
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                sortDir
            );
        }
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L){
    if(!L){
        *log2L = 0;
        return 0;
    }else{
        for(*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);
        return L;
    }
}

extern "C" void bitonicSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint sortDir
){
    //Nothing to sort
    if(arrayLength < 2)
        return;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert( factorizationRemainder == 1 );

    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    assert(arrayLength <= SHARED_SIZE_LIMIT);
    assert( (batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0 );

    printf("blockCount: %d, threadCount: %d\n", blockCount, threadCount);
    //bitonicSortSharedKernel<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, sortDir);
    bitonicSortSharedKernel<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, sortDir);
    //cutilCheckMsg("bitonicSortSharedKernel<<<>>> failed!\n");
}

int main() {
  uint h_SrcKey[SHARED_SIZE_LIMIT];
  uint h_SrcVal[SHARED_SIZE_LIMIT];
  klee_make_symbolic(h_SrcKey, sizeof(uint) * SHARED_SIZE_LIMIT, "srckey_input");
  klee_make_symbolic(h_SrcVal, sizeof(uint) * SHARED_SIZE_LIMIT, "srcval_input");

  uint *d_DstKey, *d_DstVal, *d_SrcKey, *d_SrcVal;
  cudaMalloc((void**)&d_DstKey, sizeof(uint) * SHARED_SIZE_LIMIT);
  cudaMalloc((void**)&d_DstVal, sizeof(uint) * SHARED_SIZE_LIMIT);
  cudaMalloc((void**)&d_SrcKey, sizeof(uint) * SHARED_SIZE_LIMIT);
  cudaMalloc((void**)&d_SrcVal, sizeof(uint) * SHARED_SIZE_LIMIT);

  cudaMemcpy(d_SrcKey, h_SrcKey, sizeof(uint) * SHARED_SIZE_LIMIT, cudaMemcpyHostToDevice);
  cudaMemcpy(d_SrcVal, h_SrcVal, sizeof(uint) * SHARED_SIZE_LIMIT, cudaMemcpyHostToDevice);

  bitonicSortShared(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, 1, SHARED_SIZE_LIMIT, 1);

}

