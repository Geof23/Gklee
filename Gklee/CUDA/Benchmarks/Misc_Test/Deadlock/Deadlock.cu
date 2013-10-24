#include <stdio.h>

#define NUM 1024

__shared__ int v[NUM];

__global__ void deadlock() {
  if (threadIdx.x % 2 == 0) { 
    v[threadIdx.x]++;
    __syncthreads();
  }
  else {
    v[threadIdx.x]--;    
    __syncthreads();  // remove this one to incur a barrier dismatch
  }
}

int main() {
  deadlock<<<1,NUM>>>();
  return 0;
}
