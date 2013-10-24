#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM 4

__global__
void atomicIntMinKernel(int *uA, int *uB, int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicMin(uA+tid, uB[tid]); 
}

__global__
void atomicUnsignedMinKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicMin(uA+tid, uB[tid]); 
}

__global__
void atomicIntMaxKernel(int *uA, int *uB, int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicMax(uA+tid, uB[tid]); 
}

__global__
void atomicUnsignedMaxKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicMax(uA+tid, uB[tid]); 
}

__global__
void atomicIncKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicInc(uA+tid, uB[tid]); 
}

__global__
void atomicDecKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicDec(uA+tid, uB[tid]); 
}

__global__
void atomicIntCasKernel(int *uA, int *uB, int *uC, int *uD) {
  unsigned tid = threadIdx.x;
  uD[tid] = atomicCAS(uA+tid, uB[tid], uC[tid]); 
}

__global__
void atomicUnsignedCasKernel(unsigned *uA, unsigned *uB, unsigned *uC, unsigned *uD) {
  unsigned tid = threadIdx.x;
  uD[tid] = atomicCAS(uA+tid, uB[tid], uC[tid]); 
}

__global__
void atomicULLCasKernel(unsigned long long int *uA, unsigned long long int *uB, 
                        unsigned long long int *uC, unsigned long long int *uD) {
  unsigned tid = threadIdx.x;
  uD[tid] = atomicCAS(uA+tid, uB[tid], uC[tid]); 
}

int main(int argv, char **argc) {
  int hiA[NUM] = {1, 2, 3, 4}; 
  int hiB[NUM] = {1, 2, 3, 4}; 
  int hiC[NUM] = {0, 0, 0, 0};
  int hiD[NUM] = {0, 0, 0, 0};

  unsigned hA[NUM] = {1, 2, 3, 4}; 
  unsigned hB[NUM] = {1, 2, 3, 4}; 
  unsigned hC[NUM] = {0, 0, 0, 0};
  unsigned hD[NUM] = {0, 0, 0, 0};

  unsigned long long int hullA[NUM] = {1, 2, 3, 4}; 
  unsigned long long int hullB[NUM] = {1, 2, 3, 4}; 
  unsigned long long int hullC[NUM] = {0, 0, 0, 0};
  unsigned long long int hullD[NUM] = {0, 0, 0, 0};

#ifdef _SYM
  klee_make_symbolic(hiA, sizeof(int)*NUM, "hiA");
  klee_make_symbolic(hiB, sizeof(int)*NUM, "hiB");

  klee_make_symbolic(hA, sizeof(unsigned)*NUM, "hA");
  klee_make_symbolic(hB, sizeof(unsigned)*NUM, "hB");

  klee_make_symbolic(hullA, sizeof(unsigned long long int)*NUM, "hullA");
  klee_make_symbolic(hullB, sizeof(unsigned long long int)*NUM, "hullB");
#else
  srand ( time(NULL) );
  for (unsigned i = 0; i < NUM; i++) {
    hiA[i] = rand();
    hiB[i] = rand(); 
    hiC[i] = rand(); 
    hiD[i] = rand(); 
    printf("hiA[%u]: %d, hiB[%u]: %d, hiC[%u]: %d, hiD[%u]: %d\n", 
           i, hiA[i], i, hiB[i], i, hiC[i], i, hiD[i]);
    hA[i] = (unsigned)rand();
    hB[i] = (unsigned)rand(); 
    hC[i] = (unsigned)rand(); 
    hD[i] = (unsigned)rand(); 
    printf("hA[%u]: %u, hB[%u]: %u, hC[%u]: %u, hD[%u]: %u\n", 
           i, hA[i], i, hB[i], i, hC[i], i, hD[i]);
    hullA[i] = (unsigned long long int)rand();
    hullB[i] = (unsigned long long int)rand(); 
    hullC[i] = (unsigned long long int)rand(); 
    hullD[i] = (unsigned long long int)rand(); 
    printf("hullA[%u]: %llu, hullB[%u]: %llu, hullC[%u]: %llu, hullD[%u]: %llu\n", 
           i, hullA[i], i, hullB[i], i, hullC[i], i, hullD[i]);
  }
#endif

  int *diA, *diB, *diC, *diD; 
  cudaMalloc((void**)&diA, sizeof(int)*NUM);
  cudaMalloc((void**)&diB, sizeof(int)*NUM);
  cudaMalloc((void**)&diC, sizeof(int)*NUM);
  cudaMalloc((void**)&diD, sizeof(int)*NUM);

  unsigned *dA, *dB, *dC, *dD; 
  cudaMalloc((void**)&dA, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dB, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dC, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dD, sizeof(unsigned)*NUM);

  unsigned long long int *dullA, *dullB, *dullC, *dullD; 
  cudaMalloc((void**)&dullA, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullB, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullC, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullD, sizeof(unsigned long long int)*NUM);

  cudaMemcpy(diA, hiA, sizeof(int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(diB, hiB, sizeof(int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(diC, hiC, sizeof(int)*NUM, cudaMemcpyHostToDevice);

  cudaMemcpy(dA, hA, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);

  cudaMemcpy(dullA, hullA, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dullB, hullB, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dullC, hullC, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);

  atomicIntMinKernel<<<1, NUM>>>(diA, diB, diC);

  atomicUnsignedMinKernel<<<1, NUM>>>(dA, dB, dC); 

  cudaMemcpy(hiA, diA, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiB, diB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic Min intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiA[%u]: %d\n", i, hiA[i]);
    printf("hiB[%u]: %d\n", i, hiB[i]);
    printf("hiC[%u]: %d\n", i, hiC[i]);
    printf("****** \n");
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
  } 

  atomicIntMaxKernel<<<1, NUM>>>(diA, diB, diC); 

  atomicUnsignedMaxKernel<<<1, NUM>>>(dA, dB, dC); 

  cudaMemcpy(hiA, diA, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiB, diB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic Max intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiA[%u]: %d\n", i, hiA[i]);
    printf("hiB[%u]: %d\n", i, hiB[i]);
    printf("hiC[%u]: %d\n", i, hiC[i]);
    printf("****** \n");
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
  } 

  atomicIncKernel<<<1, NUM>>>(dA, dB, dC); 

  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic Inc intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
  } 

  atomicDecKernel<<<1, NUM>>>(dA, dB, dC);

  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic Dec intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
  } 

  atomicIntCasKernel<<<1, NUM>>>(diA, diB, diC, diD); 
  atomicUnsignedCasKernel<<<1, NUM>>>(dA, dB, dC, dD); 
  atomicULLCasKernel<<<1, NUM>>>(dullA, dullB, dullC, dullD); 

  cudaMemcpy(hiA, diA, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiB, diB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiD, diD, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hD, dD, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hullA, dullA, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hullB, dullB, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hullC, dullC, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hullD, dullD, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic CAS intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiA[%u]: %d\n", i, hiA[i]);
    printf("hiB[%u]: %d\n", i, hiB[i]);
    printf("hiC[%u]: %d\n", i, hiC[i]);
    printf("hiD[%u]: %d\n", i, hiD[i]);
    printf("****** \n");
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
    printf("hD[%u]: %u\n", i, hD[i]);
    printf("****** \n");
    printf("hullA[%u]: %llu\n", i, hullA[i]);
    printf("hullB[%u]: %llu\n", i, hullB[i]);
    printf("hullC[%u]: %llu\n", i, hullC[i]);
    printf("hullD[%u]: %llu\n", i, hullD[i]);
    printf("****** \n");
  }

  cudaFree(diA);
  cudaFree(diB);
  cudaFree(diC);
  cudaFree(diD);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dD);
  cudaFree(dullA);
  cudaFree(dullB);
  cudaFree(dullC);
  cudaFree(dullD);
}
