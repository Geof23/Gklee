/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


/*
 *Adapted for GKLEE
 *Guass Group, University of Utah
 *January 19, 2012
 *
 *Author(s): Tyler Sorensen
 *(Add your name if you work on it)
 *
 *This is a regular cuda file that
 *will show the improvement after fixing
 *some of the bank conflicts found by GKLEE
 *
 *Please see README.txt for complete
 *analysis of this example and how 
 *GKLEE was used both to find races
 *and improve performance!
 */


#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932f

//original kernel
__global__ void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float    shared[16][16];

    // now calculate the value at that position
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
                  (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

    // removing this syncthreads shows graphically what happens
    // when it doesn't exist.  this is an example of why we need it.
    __syncthreads();

    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

//kernel with fixed bank conflict hack
__global__ void kernel2( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float    shared[16][16+1];

    // now calculate the value at that position
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
                  (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

    // removing this syncthreads shows graphically what happens
    // when it doesn't exist.  this is an example of why we need it.
    __syncthreads();

    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}


// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    cudaEvent_t     start, stop;
    float   elapsedTime;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ) );
    int ITERATIONS=50;
    float running = 0;
      
      ////////////////////////////////////////////////////
      //Testing first kernel
      for(int i = 0; i < ITERATIONS; i++)
	{
	  data.dev_bitmap = dev_bitmap;
	  
	  HANDLE_ERROR( cudaEventCreate( &start ) );
	  HANDLE_ERROR( cudaEventCreate( &stop ) );
	  HANDLE_ERROR( cudaEventRecord( start, 0 ) );
	  
	  kernel<<<grids,threads>>>( dev_bitmap );
	  
	  HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
				    bitmap.image_size(),
				    cudaMemcpyDeviceToHost ) );
	  
	  // get stop time, and display the timing results
	  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	  HANDLE_ERROR( cudaEventSynchronize( stop ) );
	  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
					      start, stop ) );
	  printf("%f\n", elapsedTime);

	  running+=elapsedTime;
	  HANDLE_ERROR( cudaEventDestroy( start ) );
	  HANDLE_ERROR( cudaEventDestroy( stop ) );
	}
    printf( "First Kernel took:  %f ms\n\n", running/ITERATIONS );
    running = 0;
      
      //////////////////////////////////////////////////////
      //Testing optimized kernel
      for(int i = 0; i < ITERATIONS; i++)
	{
	  data.dev_bitmap = dev_bitmap;
	  
	  HANDLE_ERROR( cudaEventCreate( &start ) );
	  HANDLE_ERROR( cudaEventCreate( &stop ) );
	  HANDLE_ERROR( cudaEventRecord( start, 0 ) );
	  
	  kernel2<<<grids,threads>>>( dev_bitmap );
	  
	  HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
				    bitmap.image_size(),
				    cudaMemcpyDeviceToHost ) );
	  
	  // get stop time, and display the timing results
	  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	  HANDLE_ERROR( cudaEventSynchronize( stop ) );
	  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
					      start, stop ) );
	  running+=elapsedTime;
	  HANDLE_ERROR( cudaEventDestroy( start ) );
	  HANDLE_ERROR( cudaEventDestroy( stop ) );
	}
    
    printf( "Optimized Kernel took:  %f ms\n", running/ITERATIONS );
    
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
}
