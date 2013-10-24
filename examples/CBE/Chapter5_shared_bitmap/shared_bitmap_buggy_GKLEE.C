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
 *if ran with GKLEE this example shows
 *a race, as it should. Modifications
 *to the original file are noted.
 *
 *Please see README.txt for complete
 *analysis of this example and how 
 *GKLEE was used both to find races
 *and improve performance!
 */

//REMOVED
//#include "cuda.h"

//REMOVED (Klee-l++ doesn't like some of the methods)
//#include "../common/book.h"

#define NULL 0

#include "../common/cpu_bitmap.h"


//ADDED 
#include "cutil.h"
#include "klee.h"
#include <stdio.h>
#include "math.h"


//Lowered DIM from 1024

#define DIM 8

#define PI 3.1415926535897932f

__global__ void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    //Lowered from [16][16] to [8][8]
    __shared__ float    shared[8][8];

    // now calculate the value at that position
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
                  (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;


    // removing this syncthreads shows graphically what happens
    // when it doesn't exist.  this is an example of why we need it.
    //__syncthreads();  

    ptr[offset*4 + 0] = 0;

    //CHANGED to reflect lower number of threads
    //ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
    ptr[offset*4 + 1] = shared[7-threadIdx.x][7-threadIdx.y];
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
    
    //Removed error handle
    //HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
    //                          bitmap.image_size() ) );

    cudaMalloc( (void**)&dev_bitmap,
		bitmap.image_size());

    data.dev_bitmap = dev_bitmap;

    //REMOVED
    //dim3    grids(DIM/16,DIM/16);
    //dim3    threads(16,16);
    //kernel<<<grids,threads>>>( dev_bitmap );

    //ADDED (and lowered values
    __modify_Grid(DIM/8, DIM/8);
    __modify_Block(8,8);

    //ADDED modified kernel call
    __begin_GPU();
    kernel(dev_bitmap);
    __end_GPU();

    //REMOVED HANDLE_ERROR
    //HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
    //                          bitmap.image_size(),
    //                          cudaMemcpyDeviceToHost ) );

    cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
		bitmap.image_size(),
		cudaMemcpyDeviceToHost );

                              
    //REMOVED HANDLE ERROR
    //HANDLE_ERROR( cudaFree( dev_bitmap ) );
    cudaFree( dev_bitmap );
   
    //REMOVED (we don't need to see it to test if)
    //bitmap.display_and_exit();
}
