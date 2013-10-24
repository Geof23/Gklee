
**************************************************
*GKLEE example
*Guass Group, University of Utah
*January 19, 2012
*Author: Tyler Sorensen
*(Add your name if you work on it)
**************************************************

-------------------------------------------------------------
---Introduction----------------------------------------------
-------------------------------------------------------------
This example is taken from the book:
"CUDA BY EXAMPLE: An Introduction to General-Purpose
GPU Programming"

by:
Jason Sanders
Edward Kandrot

The specific example is:
shared_bitmap.cu in Chapter 5.

The original source can be downloaded from:
http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming

This example shows how GKLEE can be used both 
to spot races and reduce bank conflicts

-------------------------------------------------------------
---files-----------------------------------------------------
-------------------------------------------------------------

*shared_bitmap_buggy_GKLEE.C

This file shows how to turn the original shared_bitmap.cu into
a GKLEE compatable file. All changes are documented. If compiled
and ran through GKLEE, it will report a race, just as it should.

*shared_bitmap_correct_GKLEE.C 

This file is almost exactly the same as the one above but
with the added "__syncthreads" call which fixes the race reported
above.
 
*shared_bitmap_correct_BC_GKLEE.C

No races is good, but look at all those bank conflicts!

Can we fix them?

It turns out that there is simple hack to correct a lot of
bank conflicts

just switch the line:
__shared__ float    shared[8][8];
to
__shared__ float    shared[8][8+1];

Solution due to:
http://forums.nvidia.com/index.php?showtopic=152287

This file contains the updated line of code and you
should see a lot fewer bank conflicts.

*shared_bitmap_benchmarks.cu

This file is back to regular CUDA and has a simple
timing experiment to see how much time fixing all those
bank conflicts really saved us. When I sampled this on a
NVIDIA GeForce GTX 480, I saw a speed up of about ~.07 ms over
50 iterations of the kernel.
