#include "stdio.h"

#define    NUM  4 

__shared__ int shared[NUM];

__device__ inline void swap(int & a, int & b) {
  // Alternative swap doesn't use a temporary register:
  // a ^= b;
  // b ^= a;
  // a ^= b;
  
  int tmp = a;
  a = b;
  b = tmp;
}

__global__ void BitonicKernel(int * values)
{
  unsigned int tid = threadIdx.x;
  // Copy input to shared mem.  
  shared[tid] = values[tid];
  printf("tid: %d, blockDim: %d\n", tid, blockDim.x);
  __syncthreads();

  // Parallel bitonic sort.
  for (unsigned int k = 2; k <= blockDim.x; k *= 2) {
    for (unsigned int j = k / 2; j>0; j /= 2) {
      unsigned int ixj = tid ^ j;
      if (ixj > tid) {
	if ((tid & k) == 0) {
	  if (shared[tid] > shared[ixj])
	    swap(shared[tid], shared[ixj]);
	}
	else {
	  if (shared[tid] < shared[ixj])
	    swap(shared[tid], shared[ixj]);
	}
      }
      __syncthreads();
    }
  } //end sort
  
  // Write result.
  values[tid] = shared[tid];
}


int main() {

  int *values = (int *)malloc(sizeof(int) * NUM); 
  // the following is equivalent to calling the kernel using <<<...>>>(BitonicKernel)
  klee_make_symbolic(values, sizeof(int)*NUM, "values");

  int *dvalues;
  cudaMalloc((void **)&dvalues, sizeof(int) * NUM);
  cudaMemcpy(dvalues, values, sizeof(int) * NUM, cudaMemcpyHostToDevice);
  
  BitonicKernel<<<1, NUM>>>(dvalues);

#ifndef _SYM
  // here blockDim.x should be NUM; we use this hack for convenience
  for (int i = 1; i < NUM; i++) {
    if (dvalues[i] < dvalues[i-1]) {
      printf("The sorting algorithm is incorrect since values[%d] < values[%d]!\n", i, i-1);
      return 1;
    }
  }
#endif

  cudaFree(dvalues);
  free(values);

  return 0;
}
