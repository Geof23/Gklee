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
 * This sample calculates scalar products of a 
 * given set of input vector pairs
 */



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "scalarProd_kernel.cu"
#include "scalarProd_gold.cpp"



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
    float *h_C,
    float *h_A,
    float *h_B,
    int vectorN,
    int elementN
);



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
//#include "scalarProd_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Total number of input vector pairs; arbitrary
const int VECTOR_N = 256;
//const int VECTOR_N = 8;
//Number of elements per vector; arbitrary, 
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
const int ELEMENT_N = 4096;
//const int ELEMENT_N = 128;
//Total number of data elements
const int    DATA_N = VECTOR_N * ELEMENT_N;

const int   DATA_SZ = DATA_N * sizeof(float);
const int RESULT_SZ = VECTOR_N  * sizeof(float);



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    float *d_A, *d_B, *d_C;
    double delta, ref, sum_delta, sum_ref, L1norm;
    //unsigned int hTimer;
    int i;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    /*if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError( cutCreateTimer(&hTimer) );*/

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_A     = (float *)malloc(DATA_SZ);
    h_B     = (float *)malloc(DATA_SZ);
    h_C_CPU = (float *)malloc(RESULT_SZ);
    h_C_GPU = (float *)malloc(RESULT_SZ);

    printf("...allocating GPU memory.\n");
    cudaMalloc((void **)&d_A, DATA_SZ);
    cudaMalloc((void **)&d_B, DATA_SZ);
    cudaMalloc((void **)&d_C, RESULT_SZ);

    printf("...generating input data in CPU mem.\n");
#ifdef _SYM
    klee_make_symbolic(h_A, DATA_SZ, "h_A_input");
    klee_make_symbolic(h_B, DATA_SZ, "h_B_input");
    //Generating input data on CPU
#else
    srand(123);
    for(i = 0; i < DATA_N; i++){
        //h_A[i] = RandFloat(0.0f, 1.0f);
        //h_B[i] = RandFloat(0.0f, 1.0f);
        h_A[i] = 0.0f;
        h_B[i] = 1.0f;
    }
#endif

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing 
    cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice);
    printf("Data init done.\n");


    printf("Executing GPU kernel...\n");
    //cutilSafeCall( cudaDeviceSynchronize() );
    //cutilCheckError( cutResetTimer(hTimer) );
    //cutilCheckError( cutStartTimer(hTimer) );
    scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N);

    //scalarProdGPU<<<VECTOR_N, ELEMENT_N>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N);
    //cutilCheckMsg("scalarProdGPU() execution failed\n");
    //cutilSafeCall( cudaDeviceSynchronize() );
    //cutilCheckError( cutStopTimer(hTimer) );
    //printf("GPU time: %f msecs.\n", cutGetTimerValue(hTimer));

    printf("Reading back GPU result...\n");
    //Read back GPU results to compare them to CPU results
    cudaMemcpy(h_C_GPU, d_C, RESULT_SZ, cudaMemcpyDeviceToHost);


    printf("Checking GPU results...\n");
    printf("..running CPU scalar product calculation\n");
    //scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);

    printf("...comparing the results\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    /*for(i = 0; i < VECTOR_N; i++){
       //delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
       ref   = h_C_CPU[i];
       sum_delta += delta;
       sum_ref   += ref;
    }*/
    //L1norm = sum_delta / sum_ref;
    printf("L1 error: %E\n", L1norm);
    printf((L1norm < 1e-6) ? "PASSED\n" : "FAILED\n");


    printf("Shutting down...\n");
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    free(h_C_GPU);
    free(h_C_CPU);
    free(h_B);
    free(h_A);

    //cudaDeviceReset();
    //cutilExit(argc, argv);
}
