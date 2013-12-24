/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include "scanLargeArray_kernel.cu"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

inline bool 
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

float** g_scanBlockSums;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
            cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(float));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    printf("preallocBlockSums\n");
}

void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
    {
        cudaFree(g_scanBlockSums[i]);
    }

    //cutilCheckMsg("deallocBlockSums");
    printf("deallocBlockSums\n");
    
    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}


void prescanArrayRecursive(float *outArray, 
                           const float *inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(float) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // make sure there are no CUDA errors before we start
    printf("prescanArrayRecursive before kernels\n");

    // execute the scan
    if (numBlocks > 1)
    {
        printf("sharedMemSize: %u \n", sharedMemSize);
        prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
                                                                 inArray, 
                                                                 g_scanBlockSums[level],
                                                                 numThreads * 2, 0, 0);
        //cutilCheckMsg("prescanWithBlockSums");
        printf("prescanWithBlockSums\n");
        if (np2LastBlock)
        {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
            //cutilCheckMsg("prescanNP2WithBlockSums");
            printf("prescanNP2WithBlockSums\n");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        printf("before uniformAdd! \n");
        uniformAdd<<< grid, threads >>>(outArray, 
                                        g_scanBlockSums[level], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        //cutilCheckMsg("uniformAdd");
        printf("uniformAdd\n");
        if (np2LastBlock)
        {
            printf("this uniformAdd numThreadsLastBlock: %d \n", numThreadsLastBlock);
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, 
                                                     g_scanBlockSums[level], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
            //cutilCheckMsg("uniformAdd");
            printf("uniformAdd\n");
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
        //cutilCheckMsg("prescan");
        printf("prescan\n");
    }
    else
    {
         prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
                                                                  0, numElements, 0, 0);
         //cutilCheckMsg("prescanNP2");
         printf("prescanNP2\n");
    }
}

void prescanArray(float *outArray, float *inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}

void runTest( int argc, char** argv);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len)
{
  reference[0] = 0;
#ifdef _DEBUG
  double total_sum = 0;
#endif

  for( unsigned int i = 1; i < len; ++i) 
  {
#ifdef _DEBUG
      total_sum += idata[i-1];
#endif
      reference[i] = idata[i-1] + reference[i-1];
  }
#ifdef _DEBUG
  if (total_sum != reference[len-1])
      printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
#endif
  
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    //cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//#ifndef __DEVICE_EMULATION__
//  unsigned int num_test_iterations = 100;
//  unsigned int num_elements = 1000000; // can support large, non-power-of-2 arrays!
//#else
    unsigned int num_test_iterations = 1;
    unsigned int num_elements = 10000; // can support large, non-power-of-2 arrays!
//#endif
    
    unsigned int mem_size = sizeof( float) * num_elements;
    
    // allocate host memory to store the input data
    float* h_data = (float*) malloc( mem_size);
      
#ifndef _SYM
    // initialize the input data on the host
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_data[i] = 1.0f;//(int)(10 * rand()/32768.f);
    }
    // compute reference solution
    float* reference = (float*) malloc( mem_size); 
    //cutStartTimer(timerCPU);
    for (unsigned int i = 0; i < num_test_iterations; i++)
    {
        computeGold( reference, h_data, num_elements);
    }
#else
    klee_make_symbolic(h_data, mem_size, "h_data_input"); 
#endif

    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;

    cudaMalloc((void**) &d_idata, mem_size);
    cudaMalloc((void**) &d_odata, mem_size);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
    // initialize all the other device arrays to be safe
    cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice);

    printf("Running parallel prefix sum (prescan) of %d elements\n", num_elements);
    printf("This version is work efficient (O(n) adds)\n");
    printf("and has very few shared memory bank conflicts\n\n");

    preallocBlockSums(num_elements);

    // run once to remove startup overhead
    prescanArray(d_odata, d_idata, num_elements);

    // Run the prescan
    // cutStartTimer(timerGPU);
    for (unsigned int i = 0; i < num_test_iterations; i++)
    {
        //printf("prescanArray\n");
        prescanArray(d_odata, d_idata, num_elements);
    }
    //cutStopTimer(timerGPU);

    deallocBlockSums();    
    // copy result from device to host
    cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                cudaMemcpyDeviceToHost);

    // If this is a regression test write the results to a file
    /*if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test 
        cutWriteFilef( "./data/result.dat", h_data, num_elements, 0.0);
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        unsigned int result_regtest = cutComparef( reference, h_data, num_elements);
        printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
        printf( "Average GPU execution time: %f ms\n", cutGetTimerValue(timerGPU) / num_test_iterations);
        printf( "CPU execution time:         %f ms\n", cutGetTimerValue(timerCPU) / num_test_iterations);
    }*/

    printf("\nCheck out the CUDA Data Parallel Primitives Library for more on scan.\n");
    printf("http://www.gpgpu.org/developer/cudpp\n");

    // cleanup memory
    //cutDeleteTimer(timerCPU);
    //cutDeleteTimer(timerGPU);
    free( h_data);
#ifndef _SYM
    free( reference);
#endif
    cudaFree( d_odata);
    cudaFree( d_idata);
    //cudaThreadExit();
}

#endif // _PRESCAN_CU_
