#include "stdio.h"

#define NUM 64

__global__ void SimpleKernel(unsigned * a) {
  __shared__ unsigned num_odd; 
  unsigned int tid = threadIdx.x;
 
  __syncthreads();

  if (tid % 2 == 0) {
    printf("Tid % 2 == 0!\n");
    num_odd++;
    if (tid % 4 == 0) {
      printf("Tid % 4 == 0!\n");
    } else {
      printf("Tid % 4 != 0!\n");
    }
    // %3 ...
    if (tid % 3 == 0) {
      printf("Tid % 3 == 0!\n");
    } else {
      printf("Tid % 3 != 0!\n");
    }
  } else {
    printf("Tid % 2 != 0!\n");
    if (tid % 4 == 0) {
      printf("Tid % 4 == 0!\n");
    } else {
      printf("Tid % 4 != 0!\n");
    }
    // %3 ...
    if (tid % 3 == 0) {
      printf("Tid % 3 == 0!\n");
    } else {
      printf("Tid % 3 != 0!\n");
    }
  } 

  __syncthreads();
  
  printf("Before determine accumulative\n");
  if (num_odd >= 1) {
    printf("The number of threads whose number is even is greater than or equal to 1\n");
  } else {
    printf("The number of threads whose number is even is 0\n");
  }
  printf("end here\n");
}

int main() {
  unsigned a[NUM];
  klee_make_symbolic(&a, sizeof(a), "input");
  unsigned *da;
  cudaMalloc((void **)&da, sizeof(unsigned) * NUM);
  cudaMemcpy(da, a, sizeof(unsigned) * NUM, cudaMemcpyHostToDevice);

  SimpleKernel<<<1, NUM>>>(da);

  cudaFree(da);
}
