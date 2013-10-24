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
#include <stdio.h>


const char *sSDKsample = "Transpose";

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TILE_DIM    16
#define BLOCK_ROWS  16

// Modified the amount of workload, so during device emulation it runs in a reasonable amount of time.
// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
#ifdef __DEVICE_EMULATION__
    int MATRIX_SIZE_X = 128;
    int MATRIX_SIZE_Y = 128;
    int MUL_FACTOR    = 4;
#else
    int MATRIX_SIZE_X = 1024;
    int MATRIX_SIZE_Y = 1024;
    int MUL_FACTOR    = TILE_DIM;
#endif

#define FLOOR(a,b) (a-(a%b))

// Compute the tile size necessary to illustrate performance cases for SM12+ hardware
int MAX_TILES_SM12 = (FLOOR(MATRIX_SIZE_X,512) * FLOOR(MATRIX_SIZE_Y,512)) / (TILE_DIM*TILE_DIM);   
// Compute the tile size necessary to illustrate performance cases for SM10,SM11 hardware
int MAX_TILES_SM10 = (FLOOR(MATRIX_SIZE_X,384) * FLOOR(MATRIX_SIZE_Y,384)) / (TILE_DIM*TILE_DIM);

// Number of repetitions used for timing.  Two sets of repetitions are performed:
// 1) over kernel launches and 2) inside the kernel over just the loads and stores

#define NUM_REPS  100

// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void copy(float *odata, float* idata, int width, int height, int nreps)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  
  int index  = xIndex + width*yIndex;
  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index+i*width] = idata[index+i*width];
    }
  }
}

__global__ void copySharedMem(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  
  int index  = xIndex + width*yIndex;
  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
	  if (xIndex < width && yIndex < height)
        tile[threadIdx.y][threadIdx.x] = idata[index];
    }
  
    __syncthreads();
  
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      if (xIndex < height && yIndex < width)
        odata[index] = tile[threadIdx.y][threadIdx.x];
    }
  }
}

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void transposeNaive(float *odata, float* idata, int width, int height, int nreps)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;
  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i] = idata[index_in+i*width];
    }
  }
}

// coalesced transpose (with bank conflicts)

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
  
    __syncthreads();
  
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}

// Coalesced transpose with no bank conflicts

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      printf("threadIdx.x: %d, threadIdx.y: %d, i: %d\n", threadIdx.x, threadIdx.y, i);
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
  
    __syncthreads();
  
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}

// Transpose that effectively reorders execution of thread blocks along diagonals of the 
// matrix (also coalesced and has no bank conflicts)
//
// Here blockIdx.x is interpreted as the distance along a diagonal and blockIdx.y as 
// corresponding to different diagonals
//
// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the more commonly 
// used cartesian coordinates so that the only changes to the code from the coalesced version 
// are the calculation of the blockIdx_x and blockIdx_y and replacement of blockIdx.x and 
// bloclIdx.y with the subscripted versions in the remaining code

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }    

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
  
    __syncthreads();
  
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}

// --------------------------------------------------------------------
// Partial transposes
// NB: the coarse- and fine-grained routines only perform part of a 
//     transpose and will fail the test against the reference solution
//
//     They are used to assess performance characteristics of different
//     components of a full transpose
// --------------------------------------------------------------------

__global__ void transposeFineGrained(float *odata, float *idata, int width, int height,  int nreps)
{
  __shared__ float block[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index = xIndex + (yIndex)*width;

  for (int r=0; r<nreps; r++) {
    for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
      block[threadIdx.y+i][threadIdx.x] = idata[index+i*width];
    }  
     
    __syncthreads();

    for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
      odata[index+i*height] = block[threadIdx.x][threadIdx.y+i];
    }
  }
}


__global__ void transposeCoarseGrained(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float block[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r<nreps; r++) {
    for (int i=0; i<TILE_DIM; i += BLOCK_ROWS) {
      block[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
  
    __syncthreads();

    for (int i=0; i<TILE_DIM; i += BLOCK_ROWS) {
      odata[index_out+i*height] = block[threadIdx.y+i][threadIdx.x];
    }
  }
}


// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float* gold, float* idata,
			  const  int size_x, const  int size_y)
{
  for(  int y = 0; y < size_y; ++y) {
    for(  int x = 0; x < size_x; ++x) {
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
    }
  }
}


/*void getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int &size_x, int &size_y, int max_tile_dim)
{
    // set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
  if ( cutGetCmdLineArgumenti(argc, (const char **) argv, "dimx", &size_x) || 
	   cutGetCmdLineArgumenti(argc, (const char **) argv, "dimX", &size_x) ) {
      if (size_x > max_tile_dim) {
          printf("> MatrixSize X = %d is greater than the recommended size = %d\n", size_x, max_tile_dim);
      } else {
          printf("> MatrixSize X = %d\n", size_x);
      }
  } else {
      size_x = max_tile_dim;
      // If this is SM12 hardware, we want to round down to a multiple of 512
      if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1 ) {
          size_x = FLOOR(size_x, 512);
      } else { // else for SM10,SM11 we round down to a multiple of 384
          size_x = FLOOR(size_x, 384);
      }
  }

  if ( cutGetCmdLineArgumenti(argc, (const char **) argv, "dimy", &size_y) ||
	   cutGetCmdLineArgumenti(argc, (const char **) argv, "dimY", &size_y) ) {
      if (size_y > max_tile_dim) {
          printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", size_y, max_tile_dim);
      } else {
          printf("> MatrixSize Y = %d\n", size_y);
      }
  } else {
      size_y = max_tile_dim;
      // If this is SM12 hardware, we want to round down to a multiple of 512
      if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1) {
          size_y = FLOOR(size_y, 512);
      } else { // else for SM10,SM11 we round down to a multiple of 384
          size_y = FLOOR(size_y, 384);
      }
  }
}*/


void
showHelp()
{
  printf("\n> Command line options\n", sSDKsample);
  printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
  printf("> The default matrix size can be overridden with these parameters\n");
  printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
  printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}


// ----
// main
// ----
