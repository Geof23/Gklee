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



#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mergeSort_common.h"
#include "mergeSort.cu"
#include "mergeSort_validate.cpp"



////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){
    uint 
        *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
    uint 
        *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
    uint hTimer;

    //const uint   N = 4 * 1048576;
    const uint   N = 4 * 1024;
    const uint DIR = 1;

    const uint numValues = 65536;


    printf("Allocating and initializing host arrays...\n\n");
        h_SrcKey = (uint *)malloc(N * sizeof(uint));
        h_SrcVal = (uint *)malloc(N * sizeof(uint));
        h_DstKey = (uint *)malloc(N * sizeof(uint));
        h_DstVal = (uint *)malloc(N * sizeof(uint));

        srand(2009);
        for(uint i = 0; i < N; i++)
            h_SrcKey[i] = rand() % numValues;
        fillValues(h_SrcVal, N);

    printf("Allocating and initializing CUDA arrays...\n\n");
        cudaMalloc((void **)&d_DstKey, N * sizeof(uint));
        cudaMalloc((void **)&d_DstVal, N * sizeof(uint));
        cudaMalloc((void **)&d_BufKey, N * sizeof(uint));
        cudaMalloc((void **)&d_BufVal, N * sizeof(uint));
        cudaMalloc((void **)&d_SrcKey, N * sizeof(uint));
        cudaMalloc((void **)&d_SrcVal, N * sizeof(uint));
        cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice);

    printf("Initializing GPU merge sort...\n");
        initMergeSort();

    printf("Running GPU merge sort...\n");
        cudaDeviceSynchronize();
            mergeSort(
                d_DstKey,
                d_DstVal,
                d_BufKey,
                d_BufVal,
                d_SrcKey,
                d_SrcVal,
                N,
                DIR
            );
        cudaDeviceSynchronize();

    printf("Reading back GPU merge sort results...\n");
        cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost);

    printf("Inspecting the results...\n");
        uint keysFlag = validateSortedKeys(
            h_DstKey,
            h_SrcKey,
            1,
            N,
            numValues,
            DIR
        );

        uint valuesFlag = validateSortedValues(
            h_DstKey,
            h_DstVal,
            h_SrcKey,
            1,
            N
        );

    printf( (keysFlag && valuesFlag) ? "TEST PASSED\n" : "TEST FAILED\n");

    printf("Shutting down...\n");
        closeMergeSort();
        cudaFree(d_SrcVal);
        cudaFree(d_SrcKey);
        cudaFree(d_BufVal);
        cudaFree(d_BufKey);
        cudaFree(d_DstVal);
        cudaFree(d_DstKey);
        free(h_DstVal);
        free(h_DstKey);
        free(h_SrcVal);
        free(h_SrcKey);
        cudaDeviceReset();
}
