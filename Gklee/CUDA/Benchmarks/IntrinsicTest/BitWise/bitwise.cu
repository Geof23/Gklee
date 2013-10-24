#include <cuda.h>
#include <stdio.h>

#define NUM 4

__global__
void clzKernel(int *uA, int *uB) {
  unsigned tid = threadIdx.x;
  uB[tid] = __clz(uA[tid]); 
}

__global__
void ffsKernel(int *uA, int *uB) {
  unsigned tid = threadIdx.x;
  uB[tid] = __ffs(uA[tid]); 
}

__global__
void popcKernel(unsigned *uA, unsigned *uB) {
  unsigned tid = threadIdx.x;
  uB[tid] = __popc(uA[tid]); 
}

__global__
void brevKernel(unsigned *uA, unsigned *uB) {
  unsigned tid = threadIdx.x;
  uB[tid] = __brev(uA[tid]); 
}

__global__
void bytePermKernel(unsigned *uA, unsigned *uB, 
                    unsigned *uC, unsigned *uD) {
  unsigned tid = threadIdx.x;
  uD[tid] = __byte_perm(uA[tid], uB[tid], uC[tid]); 
}

int main(int argv, char **argc) {
  int hA[NUM] = {1, 2, 3, 4};
  int hB[NUM] = {1, 2, 3, 4};

  int *dA, *dB;
  cudaMalloc((void**)&dA, sizeof(int)*NUM);
  cudaMalloc((void**)&dB, sizeof(int)*NUM);

  cudaMemcpy(dA, hA, sizeof(int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(int)*NUM, cudaMemcpyHostToDevice);

  clzKernel<<<1, NUM>>>(dA, dB);

  cudaMemcpy(hB, dB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After clz intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hB[%u]: %d\n", i, hB[i]);
  } 

  ffsKernel<<<1, NUM>>>(dA, dB);

  cudaMemcpy(hB, dB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After ffs intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hB[%u]: %d\n", i, hB[i]);
  } 

  unsigned huA[NUM] = {1, 2, 3, 4};
  unsigned huB[NUM] = {1, 2, 3, 4};
  unsigned huC[NUM] = {1, 2, 3, 4};
  unsigned huD[NUM] = {1, 2, 3, 4};

  unsigned *duA, *duB, *duC, *duD;

  cudaMalloc((void**)&duA, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&duB, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&duC, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&duD, sizeof(unsigned)*NUM);

  cudaMemcpy(duA, huA, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(duB, huB, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(duC, huC, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(duD, huD, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);

  popcKernel<<<1, NUM>>>(duA, duB); 

  cudaMemcpy(huB, duB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After popc intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("huB[%u]: %u\n", i, huB[i]);
  } 

  brevKernel<<<1, NUM>>>(duA, duB); 

  cudaMemcpy(huB, duB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After brev intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("huB[%u]: %u\n", i, huB[i]);
  } 

  bytePermKernel<<<1, NUM>>>(duA, duB, duC, duD); 

  cudaMemcpy(huD, duD, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After bytePerm intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("huD[%u]: %u\n", i, huD[i]);
  } 

  cudaFree(dA);
  cudaFree(dB);

  cudaFree(duA);
  cudaFree(duB);
  cudaFree(duC);
  cudaFree(duD);
}
