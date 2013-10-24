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

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Naive compute implementation of scan, one thread per element
//! Not work efficient: log(n) steps, but n * (log(n) - 1) adds.
//! Not shared storage efficient either -- this requires ping-ponging
//! arrays in shared memory due to hazards so 2 * n storage space.
//!
//! Pro: Simple
//! Con: Not work efficient
//!
//! @param g_odata  output data in global memory
//! @param g_idata  input data in global memory
//! @param n        input number of elements to scan from input data
///////////////////////////////////////////////////////////////////////////////
   __extern__shared__  float temp[];
__global__ void scan_naive(float *g_odata, float *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels

    int thid = threadIdx.x;

    int pout = 0;
    int pin = 1;

    // Cache the computational window in shared memory
    temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;

    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        temp[pout*n+thid] = temp[pin*n+thid];

        if (thid >= offset)
            temp[pout*n+thid] += temp[pin*n+thid - offset];
    }

    __syncthreads();

    g_odata[thid] = temp[pout*n+thid];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
