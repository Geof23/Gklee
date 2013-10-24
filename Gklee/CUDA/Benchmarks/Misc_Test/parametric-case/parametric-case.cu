#include <stdio.h>

#define  BLOCK 16
#define  NUM  2048

__shared__ unsigned b[NUM];

__device__ void devicetest(unsigned *b) {
  unsigned sum = 0;
  for (unsigned i = 0; i < NUM; i++) {
    sum += b[i]; 
  }
  printf("sum: %u \n", sum);
} 

__global__ void paratest(unsigned * a) {
  unsigned bid = blockIdx.x;
  unsigned tid = threadIdx.x;

  b[tid] = a[tid];
  for (unsigned i = 0; i < 4; i++) {
    if (bid % 2 != 0) {
      if (tid < 1024) {
        unsigned idx = bid * blockDim.x + tid;
        b[tid] = a[idx] + 1;
        if (tid % 2 != 0) {
          b[tid] = 2;
        } else {
          if (tid > 0)
            b[tid] = b[tid-1]+1;
        }
      } else {
        b[tid] = b[tid-1];
      }
    } else {
      unsigned idx = bid * blockDim.x + tid;
      b[tid] = a[idx] + 1;
    }
  }

  __syncthreads();

  if (tid % 2 == 0) {
    printf("even number !\n");
  } else {
    printf("odd number !\n");
  }
}

int main() {
  unsigned *da;
  cudaMalloc((void **)&da, sizeof(unsigned)*BLOCK*NUM);

  //unsigned a = 8;
  //unsigned b = test_external_library(a);
  //printf("The returned b's value is %d\n", b);
  paratest<<<BLOCK, NUM>>>(da);

  cudaFree(da);
}
