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
 
 /*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

// Utility and system includes
//#include <shrUtils.h>
//#include <cuda_runtime.h>
//#include <cutil_inline.h>

#include <stdio.h>
// project include
#include "histogram_common.h"

#include "histogram_gold.cpp"

#ifdef __DEVICE_EMULATION__
const int numRuns = 1;
#else
const int numRuns = 16;
#endif

//static char *sSDKsample = "[histogram]\0";

int main(int argc, char **argv){
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    uint hTimer;
    int PassFailFlag = 1;
    const uint byteCount = 64 * 16;
    uint uiSizeMult = 1;

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
    h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

    uchar t_data[64*2];
    klee_make_symbolic(t_data, sizeof(t_data), "input");
    for (int i = 0; i < 64*2; i++)
      h_Data[i] = t_data[i];
  
    for (int j = 64*2; j < byteCount; j++) {
      h_Data[j] = 0;
    }

    printf("...allocating GPU memory and copying input data\n\n");
    cudaMalloc((void **)&d_Data, byteCount);
    cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint));
    cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice);

#ifdef HIST64

    {
       printf("Starting up 64-bin histogram...\n\n");
       initHistogram64();

       printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

       histogram64(d_Histogram, d_Data, byteCount);

       printf("\nValidating GPU results...\n");
       printf(" ...reading back GPU results\n");
       cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost);

       printf(" ...histogram64CPU()\n");
       histogram64CPU(
              h_HistogramCPU,
              h_Data,
              byteCount
       );

#ifndef _SYM
       // printf(" ...comparing the results...\n");
       // for(uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
       //    if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
       // printf(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n" );
#endif

       printf("Shutting down 64-bin histogram...\n\n\n");
       closeHistogram64();
    }
#else
    {
       printf("Initializing 256-bin histogram...\n");
       initHistogram256();

       printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

       histogram256(d_Histogram, d_Data, byteCount);

       printf("\nValidating GPU results...\n");
       // printf(" ...reading back GPU results\n");
       cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost);

       printf(" ...histogram256CPU()\n");
       histogram256CPU(
             h_HistogramCPU,
             h_Data,
             byteCount
       );

#ifndef _SYM
       // printf(" ...comparing the results\n");
       // for(uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
       //   if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
       // printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n"  );
#endif

       printf("Shutting down 256-bin histogram...\n\n\n");
       closeHistogram256();
    }
#endif
      //printf("%s - Test Summary\n", sSDKsample);

    // pass or fail (for both 64 bit and 256 bit histograms)
    // printf("%s\n\n", PassFailFlag ? "PASSED" : "FAILED");
    
    printf("Shutting down...\n");
    cudaFree(d_Histogram);
    cudaFree(d_Data);
    free(h_HistogramGPU);
    free(h_HistogramCPU);
    free(h_Data);
}
