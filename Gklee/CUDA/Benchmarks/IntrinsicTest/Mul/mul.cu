#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM 4

__global__
void mulHiKernel(int *uA, int *uB, int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = __mulhi(uA[tid], uB[tid]); 
}

__global__
void uMulHiKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = __umulhi(uA[tid], uB[tid]); 
}

__global__
void mul64HiKernel(long long int *uA, long long int *uB, long long int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = __mul64hi(uA[tid], uB[tid]); 
}

__global__
void uMul64HiKernel(unsigned long long int *uA, unsigned long long int *uB, unsigned long long int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = __umul64hi(uA[tid], uB[tid]); 
}

__global__
void mul24Kernel(int *uA, int *uB, int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = __mul24(uA[tid], uB[tid]); 
}

__global__
void uMul24Kernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = __umul24(uA[tid], uB[tid]); 
}

int main(int argv, char **argc) {
  int hiA[NUM] = {1, 2, 3, 4};
  int hiB[NUM] = {1, 2, 3, 4};
  int hiC[NUM] = {1, 2, 3, 4};

  unsigned hA[NUM] = {1, 2, 3, 4};
  unsigned hB[NUM] = {1, 2, 3, 4};
  unsigned hC[NUM] = {1, 2, 3, 4};

  long long int hllA[NUM] = {1, 2, 3, 4};
  long long int hllB[NUM] = {1, 2, 3, 4};
  long long int hllC[NUM] = {1, 2, 3, 4};

  unsigned long long int hullA[NUM] = {1, 2, 3, 4};
  unsigned long long int hullB[NUM] = {1, 2, 3, 4};
  unsigned long long int hullC[NUM] = {1, 2, 3, 4};

#ifdef _SYM
  klee_make_symbolic(hiA, sizeof(int) * NUM, "hiA"); 
  klee_make_symbolic(hiB, sizeof(int) * NUM, "hiB"); 
  klee_make_symbolic(hiC, sizeof(int) * NUM, "hiC"); 

  klee_make_symbolic(hA, sizeof(unsigned) * NUM, "hA"); 
  klee_make_symbolic(hB, sizeof(unsigned) * NUM, "hB"); 
  klee_make_symbolic(hC, sizeof(unsigned) * NUM, "hC"); 

  klee_make_symbolic(hllA, sizeof(long long int) * NUM, "hllA"); 
  klee_make_symbolic(hllB, sizeof(long long int) * NUM, "hllB"); 
  klee_make_symbolic(hllC, sizeof(long long int) * NUM, "hllC"); 

  klee_make_symbolic(hullA, sizeof(unsigned long long int) * NUM, "hullA"); 
  klee_make_symbolic(hullB, sizeof(unsigned long long int) * NUM, "hullB"); 
  klee_make_symbolic(hullC, sizeof(unsigned long long int) * NUM, "hullC"); 
#else
  srand ( time(NULL) );
  for (unsigned i = 0; i < NUM; i++) {
    hiA[i] = rand();
    hiB[i] = rand();
    hiC[i] = rand();
    printf("hiA[%u]: %d, hiB[%u]: %d, hiC[%u]: %d\n",
           i, hiA[i], i, hiB[i], i, hiC[i]);
    hA[i] = (unsigned)rand();
    hB[i] = (unsigned)rand();
    hC[i] = (unsigned)rand();
    printf("hA[%u]: %u, hB[%u]: %u, hC[%u]: %u\n",
           i, hA[i], i, hB[i], i, hC[i]);
    hllA[i] = (long long int)rand();
    hllB[i] = (long long int)rand();
    hllC[i] = (long long int)rand();
    printf("hllA[%u]: %lld, hllB[%u]: %lld, hllC[%u]: %llu\n",
           i, hllA[i], i, hllB[i], i, hllC[i]);
    hullA[i] = (unsigned long long int)rand();
    hullB[i] = (unsigned long long int)rand();
    hullC[i] = (unsigned long long int)rand();
    printf("hullA[%u]: %llu, hullB[%u]: %llu, hullC[%u]: %llu\n",
           i, hullA[i], i, hullB[i], i, hullC[i]);
  }
#endif

  int *diA, *diB, *diC;
  cudaMalloc((void**)&diA, sizeof(int)*NUM);
  cudaMalloc((void**)&diB, sizeof(int)*NUM);
  cudaMalloc((void**)&diC, sizeof(int)*NUM);

  cudaMemcpy(diA, hiA, sizeof(int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(diB, hiB, sizeof(int)*NUM, cudaMemcpyHostToDevice);

  mulHiKernel<<<1, NUM>>>(diA, diB, diC);

  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);

  printf("After mulHi intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiC[%u]: %d\n", i, hiC[i]);
  } 

  unsigned *dA, *dB, *dC;
  cudaMalloc((void**)&dA, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dB, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dC, sizeof(unsigned)*NUM);

  cudaMemcpy(dA, hA, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);

  uMulHiKernel<<<1, NUM>>>(dA, dB, dC);

  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);

  printf("After uMulHi intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hC[%u]: %u\n", i, hC[i]);
  } 

  long long int *dllA, *dllB, *dllC;
  cudaMalloc((void**)&dllA, sizeof(long long int)*NUM);
  cudaMalloc((void**)&dllB, sizeof(long long int)*NUM);
  cudaMalloc((void**)&dllC, sizeof(long long int)*NUM);

  cudaMemcpy(dllA, hllA, sizeof(long long int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dllB, hllB, sizeof(long long int)*NUM, cudaMemcpyHostToDevice);

  mul64HiKernel<<<1, NUM>>>(dllA, dllB, dllC);

  cudaMemcpy(hllC, dllC, sizeof(long long int)*NUM, cudaMemcpyDeviceToHost);

  printf("After mul64Hi intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hllC[%u]: %lld\n", i, hllC[i]);
  }

  unsigned long long int *dullA, *dullB, *dullC;
  cudaMalloc((void**)&dullA, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullB, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullC, sizeof(unsigned long long int)*NUM);

  cudaMemcpy(dullA, hullA, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dullB, hullB, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);

  uMul64HiKernel<<<1, NUM>>>(dullA, dullB, dullC);

  cudaMemcpy(hullC, dullC, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);

  printf("After uMul64Hi intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hullC[%u]: %llu\n", i, hullC[i]);
  }

  mul24Kernel<<<1, NUM>>>(diA, diB, diC);

  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);

  printf("After mul24 intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiC[%u]: %d\n", i, hiC[i]);
  } 

  uMul24Kernel<<<1, NUM>>>(dA, dB, dC);

  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);

  printf("After umul24 intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiC[%u]: %u\n", i, hC[i]);
  }

  cudaFree(diA);
  cudaFree(diB);
  cudaFree(diC);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  cudaFree(dllA);
  cudaFree(dllB);
  cudaFree(dllC);

  cudaFree(dullA);
  cudaFree(dullB);
  cudaFree(dullC);
}
