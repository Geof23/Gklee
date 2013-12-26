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
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include "scan_common.h"
#include "scan_gold.cpp"
#include <stdio.h>

int main(int argc, char **argv)
{
  // Start logs
  //shrSetLogFileName ("scan.txt");
  printf("%s Starting...\n\n", argv[0]); 

  //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
  /*
  if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
     cutilDeviceInit(argc, argv);
  else
     cudaSetDevice( cutGetMaxGflopsDeviceId() );
  */
  uint *d_Input, *d_Output;
  uint *h_Input, *h_OutputCPU, *h_OutputGPU;
  uint hTimer;

  //const uint N = 13 * 1048576 / 2;
  const uint N = 2048;

  printf("Allocating and initializing host arrays...\n");
        //cutCreateTimer(&hTimer);
  h_Input     = (uint *)malloc(N * sizeof(uint));
  h_OutputCPU = (uint *)malloc(N * sizeof(uint));
  h_OutputGPU = (uint *)malloc(N * sizeof(uint));
        //srand(2009);
  //for(uint i = 0; i < N; i++)
  //  h_Input[i] = rand();
  klee_make_symbolic(h_Input, sizeof(uint) * N, "input"); 
  printf("Allocating and initializing CUDA arrays...\n");
  cudaMalloc((void **)&d_Input, N * sizeof(uint));
  cudaMalloc((void **)&d_Output, N * sizeof(uint));

  cudaMemcpy(d_Input, h_Input, N * sizeof(uint), cudaMemcpyHostToDevice);

  printf("Initializing CUDA-C scan...\n\n");
  initScan();

  int globalFlag = 1;
  size_t szWorkgroup;
  //const int iCycles = 100;
  const int iCycles = 5;
  printf("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", iCycles);
  for(uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength <<= 1){
     printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
               //cutilSafeCall( cudaDeviceSynchronize() );
                //cutResetTimer(hTimer);
                //cutStartTimer(hTimer);
                //for(int i = 0; i < iCycles; i++)
                //{
                    //printf("The arrayLength in scanExclusiveShort: %d, the i: %d\n", arrayLength, i);
     szWorkgroup = scanExclusiveShort(d_Output, d_Input, N / arrayLength, arrayLength);
                //}
                //cutilSafeCall( cudaDeviceSynchronize());
                //cutStopTimer(hTimer);
                //double timerValue = 1.0e-3 * cutGetTimerValue(hTimer) / iCycles;

     printf("Validating the results...\n");
     printf("...reading back GPU results\n");

     cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost);

     printf(" ...scanExclusiveHost()\n");
     scanExclusiveHost(h_OutputCPU, d_Input, N / arrayLength, arrayLength);

     // Compare GPU results with CPU results and accumulate error for this test
     printf(" ...comparing the results\n");
     int localFlag = 1;
#ifndef _SYM
     for(uint i = 0; i < N; i++)
     {
       if(h_OutputCPU[i] != h_OutputGPU[i])
       {
         localFlag = 0;
         break;
       }
     }
#endif

     // Log message on individual test result, then accumulate to global flag
     printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
     globalFlag = globalFlag && localFlag;

     // Data log
     if (arrayLength == MAX_SHORT_ARRAY_SIZE)
     {
        printf("\n");
        //printfEx(LOGBOTH | MASTER, 0, "scan-Short, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n", 
        //      (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);
        printf("\n");
     }
  }
  printf("***Running GPU scan for large arrays (%u identical iterations)...\n\n", iCycles);
  for(uint arrayLength = MIN_LARGE_ARRAY_SIZE; arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength <<= 1){
     printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
     //for(int i = 0; i < iCycles; i++)
     //{
     printf("The arrayLength in scanExclusiveLarge: %d\n", arrayLength);
     szWorkgroup = scanExclusiveLarge(d_Output, d_Input, N / arrayLength, arrayLength);
     //}

     printf("Validating the results...\n");
     printf("...reading back GPU results\n");
     cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost);

     printf("...scanExclusiveHost()\n");
     scanExclusiveHost(h_OutputCPU, d_Input, N / arrayLength, arrayLength);

     // Compare GPU results with CPU results and accumulate error for this test
     printf(" ...comparing the results\n");
     int localFlag = 1;
#ifndef _SYM
     for(uint i = 0; i < N; i++)
     {
         if(h_OutputCPU[i] != h_OutputGPU[i])
         {
           localFlag = 0;
           break;
         }
     }
#endif
     // Log message on individual test result, then accumulate to global flag
     printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
     globalFlag = globalFlag && localFlag;

     // Data log
     if (arrayLength == MAX_LARGE_ARRAY_SIZE)
     {
        printf("\n");
        //printfEx(LOGBOTH | MASTER, 0, "scan-Large, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %u, Workgroup = %u\n", 
        //      (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);
        printf("\n");
     }
  }
  // pass or fail (cumulative... all tests in the loop)
  printf(globalFlag ? "PASSED\n\n" : "FAILED\n\n");

  printf("Shutting down...\n");
  closeScan();

  cudaFree(d_Output);
  cudaFree(d_Input);

  free(h_Input);
  free(h_OutputCPU);
  free(h_OutputGPU);
}
