#include "stdio.h"

//Total number of input vector pairs; arbitrary
const unsigned vectorN = 8;

//Number of elements per vector; arbitrary,
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
const unsigned elementN = 8;

//Total number of data elements
#define NUM vectorN*elementN

// ACCUM_N has to be power of two at the second stage
const unsigned ACCUM_N = 128;

//Accumulators cache
__shared__ unsigned accumResult[ACCUM_N];

__global__ void ScalarProdKernel(
    unsigned *d_C,
    unsigned *d_A,
    unsigned *d_B,
    int vectorN,   // number of vectors
    int elementN   // number of elements in each vector
){
  for (int vec = blockIdx.x; vec < vectorN; vec += gridDim.x) {
    int vectorBase = elementN * vec;
    int vectorEnd  = vectorBase + elementN;

    for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x) {
      int sum = 0;
      
      for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N)
    	sum += d_A[pos] * d_B[pos];
      
      accumResult[iAccum] = sum;
      printf("t%d: write %u \n", threadIdx.x, iAccum);
    }
    
    for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x == blockDim.x-1)
	printf("-----------------------------------------------------\n");
    
      printf("stride: %d\n", stride);
      __syncthreads();
      for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
	accumResult[iAccum] += accumResult[stride + iAccum];
	printf("t%d: read/write %u, read %u \n", threadIdx.x, iAccum, stride+iAccum);
      }
    }
    
    if(threadIdx.x == 0) d_C[vec] = accumResult[0];
  }
}


int main() {
  unsigned A[NUM]; 
  unsigned B[NUM]; 
  unsigned C[NUM]; 

#ifdef _SYM
  klee_make_symbolic(A, sizeof(A), "A");
  klee_make_symbolic(B, sizeof(B), "B");
#else
  // for debugging
  printf("\nInput values:\n");
  for (int i = 0; i < NUM; i++) {
    printf("%u*%u ", A[i], B[i]);
  }
  printf("\n");
#endif

  unsigned *dA, *dB, *dC;
  cudaMalloc((void**)&dA, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dB, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dC, sizeof(unsigned)*NUM);
  
  cudaMemcpy(dA, A, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);

  // the following is equivalent to calling the kernel using <<<...>>>(BitonicKernel)
  ScalarProdKernel<<<1, NUM>>>(dC, dA, dB, vectorN, elementN);

  cudaMemcpy(C, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);

#ifndef _SYM
  // for debugging
  printf("\nOutput values:\n");
  for (int i = 0; i < NUM; i++) {
    printf("%u ", C[i]);
  }
  printf("\n");
#endif

  // postcondition
  int sum1[vectorN];
  for (int vec = 0; vec < vectorN; vec++) {
    sum1[vec] = 0;
    for (int i = elementN * vec; i < elementN * (vec+1); i++) {
      sum1[vec] += A[i] * B[i];
    }
    printf("sum1[%d] = %d, C[%d] = %d\n", vec, sum1[vec], vec, C[vec]);
    if (sum1[vec] != C[vec])
      printf("The kernel is incorrect!\n");
  }

  return 0;
}
