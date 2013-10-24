/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * Walsh transforms belong to a class of generalized Fourier transformations. 
 * They have applications in various fields of electrical engineering 
 * and numeric theory. In this sample we demonstrate efficient implementation 
 * of naturally-ordered Walsh transform 
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its 
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>



////////////////////////////////////////////////////////////////////////////////
// Reference CPU FWT
////////////////////////////////////////////////////////////////////////////////
extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN
);


////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "fastWalshTransform_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int log2Kernel = 7;

#ifndef __DEVICE_EMULATION__
    const   int log2Data = 23;
#else
    const   int log2Data = 15;
#endif
const int   dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int   DATA_SIZE = dataN   * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(float);

const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
    float
        *h_Data,
        *h_Kernel,
        *h_ResultCPU,
        *h_ResultGPU;

    float
        *d_Data,
        *d_Kernel;

    double
        delta, ref, sum_delta2, sum_ref2, L2norm, gpuTime;

    unsigned int hTimer;
    int i;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError( cutCreateTimer(&hTimer) );

    printf("Initializing data...\n");
        printf("...allocating CPU memory\n");
        cutilSafeMalloc( h_Kernel    = (float *)malloc(KERNEL_SIZE) );
        cutilSafeMalloc( h_Data      = (float *)malloc(DATA_SIZE)   );
        cutilSafeMalloc( h_ResultCPU = (float *)malloc(DATA_SIZE)   );
        cutilSafeMalloc( h_ResultGPU = (float *)malloc(DATA_SIZE)   );
        printf("...allocating GPU memory\n");
        cutilSafeCall( cudaMalloc((void **)&d_Kernel, DATA_SIZE) );
        cutilSafeCall( cudaMalloc((void **)&d_Data,   DATA_SIZE) );

        printf("...generating data\n");
        printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
        srand(2007);
        for (i = 0; i < kernelN; i++)
            h_Kernel[i] = (float)rand() / (float)RAND_MAX;

        for (i = 0; i < dataN; i++)
            h_Data[i] = (float)rand() / (float)RAND_MAX;

        cutilSafeCall( cudaMemset(d_Kernel, 0, DATA_SIZE) );
        cutilSafeCall( cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy(d_Data,   h_Data,     DATA_SIZE, cudaMemcpyHostToDevice) );

    printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
    cutilSafeCall( cudaThreadSynchronize() );
    cutilCheckError( cutResetTimer(hTimer) );
    cutilCheckError( cutStartTimer(hTimer) );
        fwtBatchGPU(d_Data, 1, log2Data);
        fwtBatchGPU(d_Kernel, 1, log2Data);
        modulateGPU(d_Data, d_Kernel, dataN);
        fwtBatchGPU(d_Data, 1, log2Data);
    cutilSafeCall( cudaThreadSynchronize() );
    cutilCheckError( cutStopTimer(hTimer) );
    gpuTime = cutGetTimerValue(hTimer);
    printf("GPU time: %f ms; GOP/s: %f\n", gpuTime, NOPS / (gpuTime * 0.001 * 1E+9));

    printf("Reading back GPU results...\n");
    cutilSafeCall( cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost) );

    printf("Running straightforward CPU dyadic convolution...\n");
    dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

    printf("Comparing the results...\n");
        sum_delta2 = 0;
        sum_ref2   = 0;
        for(i = 0; i < dataN; i++){
            delta       = h_ResultCPU[i] - h_ResultGPU[i];
            ref         = h_ResultCPU[i];
            sum_delta2 += delta * delta;
            sum_ref2   += ref * ref;
        }
        L2norm = sqrt(sum_delta2 / sum_ref2);
        printf("L2 norm: %E\n", L2norm);
    printf((L2norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");


    printf("Shutting down...\n");
        cutilCheckError(  cutDeleteTimer(hTimer) );
        cutilSafeCall( cudaFree(d_Data)   );
        cutilSafeCall( cudaFree(d_Kernel) );
        free(h_ResultGPU);
        free(h_ResultCPU);
        free(h_Data);
        free(h_Kernel);

    cudaThreadExit();

    cutilExit(argc, argv);
}
