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
 
// ----------------------------------------------------------------------------------------
// Transpose
//
// This file contains both device and host code for transposing a floating-point
// matrix.  It performs several transpose kernels, which incrementally improve performance
// through coalescing, removing shared memory bank conflicts, and eliminating partition
// camping.  Several of the kernels perform a copy, used to represent the best case
// performance that a transpose can achieve.
//
// Please see the whitepaper in the docs folder of the transpose project for a detailed
// description of this performance study.
// ----------------------------------------------------------------------------------------

// Utilities and system includes
//#include <shrUtils.h>
//#include <cutil_inline.h>
#include "transpose.cu"

int
main( int argc, char** argv) 
{
  //Start logs

  // Calculate number of tiles we will run for the Matrix Transpose performance tests
  int size_x, size_y, max_matrix_dim, matrix_size_test;

  // kernel pointer and descriptor
  size_x = TILE_DIM;
  size_y = TILE_DIM;

  void (*kernel)(float *, float *, int, int, int);
  char *kernelName;

  // execution configuration parameters
  dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
  
  // size of memory required to store the matrix
  const  int mem_size = sizeof(float) * size_x*size_y;

  // allocate host memory
  float *h_idata = (float*) malloc(mem_size);
  float *h_odata = (float*) malloc(mem_size);
  float *transposeGold = (float *) malloc(mem_size);  

  // allocate device memory
  float *d_idata, *d_odata;
  cudaMalloc( (void**) &d_idata, mem_size);
  cudaMalloc( (void**) &d_odata, mem_size);

  // initalize host data
  //for(  int i = 0; i < (size_x*size_y); ++i)
  //  h_idata[i] = (float) i;
  klee_make_symbolic(h_idata, mem_size, "hidata");
  
  // copy host data to device
  cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

  //float val = value_from_extern_lib(d_idata, size_x * size_y);
  //printf("The value from external library: %f \n", val);

  // Compute reference transpose solution
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  // print out common data for all kernels
  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n", 
	 size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

  // warmup to avoid timing startup
#if defined transpose0 
  kernel = &copy;                           
  kernelName = "simple copy       "; 
#elif defined transpose1
  kernel = &copySharedMem;                  
  kernelName = "shared memory copy"; 
#elif defined transpose2
  kernel = &transposeNaive;                 
  kernelName = "naive             "; 
#elif defined transpose3 
  kernel = &transposeCoalesced;             
  kernelName = "coalesced         "; 
#elif defined transpose4 
  kernel = &transposeNoBankConflicts;       
  kernelName = "optimized         "; 
#elif defined transpose5 
  kernel = &transposeCoarseGrained;         
  kernelName = "coarse-grained    "; 
#elif defined transpose6 
  kernel = &transposeFineGrained;           
  kernelName = "fine-grained      "; 
#else
  kernel = &transposeDiagonal;              
  kernelName = "diagonal          "; 
#endif 

  kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, 1);

  cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
 
  // cleanup
  free(h_idata);
  free(h_odata);
  free(transposeGold);
  cudaFree(d_idata);
  cudaFree(d_odata);

  return 0;
}
