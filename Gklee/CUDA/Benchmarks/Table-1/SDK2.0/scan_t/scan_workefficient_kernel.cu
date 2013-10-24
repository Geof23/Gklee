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

#ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
#define _SCAN_WORKEFFICIENT_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Work-efficient compute implementation of scan, one thread per 2 elements
//! Work-efficient: O(log(n)) steps, and O(n) adds.
//! Also shared storage efficient: Uses n elements in shared mem -- no ping-ponging
//! Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
//! and Their Applications", or Prins and Chatterjee PRAM course notes:
//! https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
//!
//! Pro: Work Efficient
//! Con: Shared memory bank conflicts due to the addressing used.
//
//! @param g_odata  output data in global memory
//! @param g_idata  input data in global memory
//! @param n        input number of elements to scan from input data
///////////////////////////////////////////////////////////////////////////////
    __extern__shared__  float temp[];
__global__ void scan_workefficient(float *g_odata, float *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels

    int thid = threadIdx.x;

    int offset = 1;

    // Cache the computational window in shared memory
    temp[2*thid]   = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        temp[n - 1] = 0;
    }   

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[2*thid]   = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}

#endif // #ifndef _SCAN_WORKEFFICIENT_KERNEL_H_
