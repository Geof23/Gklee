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

__global__ void test(int *values) {
  unsigned int tid = threadIdx.x;
  shared[tid] = values[tid];
  __syncthreads();

  if (shared[tid] < shared[(tid+1)%NUM]) {
    printf("then path");
    //swap(shared[tid], shared[(tid+1)%NUM]);
    shared[tid]++;
  } else {
    printf("else path");
    //swap(shared[tid], shared[(tid+1)%NUM]);
    swap(shared[(tid+1)%NUM], shared[tid]);
  }
  __syncthreads();

  values[tid] = shared[tid];
}

int main(int argc, char ** argv) {
  int *values = (int *)malloc(sizeof(int) * NUM); 
  klee_make_symbolic(values, sizeof(int)*NUM, "values");

  int *dvalues;
  cudaMalloc((void **)&dvalues, sizeof(int) * NUM);
  cudaMemcpy(dvalues, values, sizeof(int) * NUM, cudaMemcpyHostToDevice);
  
  test<<<1, NUM>>>(dvalues);

  cudaMemcpy(values, dvalues, sizeof(int) * NUM, cudaMemcpyHostToDevice);
  for (int i = 0; i < NUM; i++) {
    printf("%d ", values[i]);
  }
  /*for (int i = 1; i < NUM; i++) {
    if (dvalues[i] < dvalues[i-1]) {
      printf("The sorting algorithm is incorrect since values[%d] < values[%d]!\n", i, i-1);
      return 1;
    }
  } */
}
