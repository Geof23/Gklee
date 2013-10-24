#include <cuda.h>
#include <stdio.h>

#define NUM 4

__global__
void atomicIntSubKernel(int *uA, int *uB, int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicSub(uA+tid, uB[tid]); 
}

__global__
void atomicUnsignedSubKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicSub(uA+tid, uB[tid]); 
}

__global__
void atomicIntExchKernel(int *uA, int *uB, int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicExch(uA+tid, uB[tid]); 
}

__global__
void atomicUnsignedExchKernel(unsigned *uA, unsigned *uB, unsigned *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicExch(uA+tid, uB[tid]); 
}

__global__
void atomicULLExchKernel(unsigned long long int *uA, unsigned long long int *uB, 
                         unsigned long long int *uC) {
  unsigned tid = threadIdx.x;
  uC[tid] = atomicExch(uA+tid, uB[tid]); 
}

__global__
void atomicFloatExchKernel(float *fA, float *fB, float *fC) {
  unsigned tid = threadIdx.x;
  fC[tid] = atomicExch(fA+tid, fB[tid]); 
}

int main(int argv, char **argc) {
  int hiA[NUM] = {1, 2, 3, 4}; 
  int hiB[NUM] = {1, 2, 3, 4}; 
  int hiC[NUM] = {0, 0, 0, 0};

  unsigned hA[NUM] = {1, 2, 3, 4}; 
  unsigned hB[NUM] = {1, 2, 3, 4}; 
  unsigned hC[NUM] = {0, 0, 0, 0};

  unsigned long long int hullA[NUM] = {1, 2, 3, 4}; 
  unsigned long long int hullB[NUM] = {1, 2, 3, 4}; 
  unsigned long long int hullC[NUM] = {0, 0, 0, 0};

  float hfA[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  float hfB[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  float hfC[NUM] = {0.0, 0.0, 0.0, 0.0}; 

  //klee_make_symbolic(hiA, sizeof(int)*NUM, "hiA");
  //klee_make_symbolic(hiB, sizeof(int)*NUM, "hiB");

  //klee_make_symbolic(hA, sizeof(unsigned)*NUM, "hA");
  //klee_make_symbolic(hB, sizeof(unsigned)*NUM, "hB");

  //klee_make_symbolic(hullA, sizeof(unsigned long long int)*NUM, "hullA");
  //klee_make_symbolic(hullB, sizeof(unsigned long long int)*NUM, "hullB");

  //klee_make_symbolic(hfA, sizeof(float)*NUM, "hfA");
  //klee_make_symbolic(hfB, sizeof(float)*NUM, "hfB");

  int *diA, *diB, *diC; 
  cudaMalloc((void**)&diA, sizeof(int)*NUM);
  cudaMalloc((void**)&diB, sizeof(int)*NUM);
  cudaMalloc((void**)&diC, sizeof(int)*NUM);

  unsigned *dA, *dB, *dC; 
  cudaMalloc((void**)&dA, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dB, sizeof(unsigned)*NUM);
  cudaMalloc((void**)&dC, sizeof(unsigned)*NUM);

  unsigned long long int *dullA, *dullB, *dullC; 
  cudaMalloc((void**)&dullA, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullB, sizeof(unsigned long long int)*NUM);
  cudaMalloc((void**)&dullC, sizeof(unsigned long long int)*NUM);

  float *dfA, *dfB, *dfC; 
  cudaMalloc((void**)&dfA, sizeof(float)*NUM);
  cudaMalloc((void**)&dfB, sizeof(float)*NUM);
  cudaMalloc((void**)&dfC, sizeof(float)*NUM);

  cudaMemcpy(diA, hiA, sizeof(int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(diB, hiB, sizeof(int)*NUM, cudaMemcpyHostToDevice);

  cudaMemcpy(dA, hA, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(unsigned)*NUM, cudaMemcpyHostToDevice);

  cudaMemcpy(dullA, hullA, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dullB, hullB, sizeof(unsigned long long int)*NUM, cudaMemcpyHostToDevice);

  cudaMemcpy(dfA, hfA, sizeof(float)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dfB, hfB, sizeof(float)*NUM, cudaMemcpyHostToDevice);

  atomicIntSubKernel<<<1, NUM>>>(diA, diB, diC);

  atomicUnsignedSubKernel<<<1, NUM>>>(dA, dB, dC); 

  cudaMemcpy(hiA, diA, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiB, diB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic Sub intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiA[%u]: %d\n", i, hiA[i]);
    printf("hiB[%u]: %d\n", i, hiB[i]);
    printf("hiC[%u]: %d\n", i, hiC[i]);
    printf("****** \n");
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
  } 

  atomicIntExchKernel<<<1, NUM>>>(diA, diB, diC); 

  atomicUnsignedExchKernel<<<1, NUM>>>(dA, dB, dC); 

  atomicULLExchKernel<<<1, NUM>>>(dullA, dullB, dullC); 

  atomicFloatExchKernel<<<1, NUM>>>(dfA, dfB, dfC); 

  cudaMemcpy(hiA, diA, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiB, diB, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hiC, diC, sizeof(int)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hA, dA, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB, dB, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC, dC, sizeof(unsigned)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hullA, dullA, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hullB, dullB, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hullC, dullC, sizeof(unsigned long long int)*NUM, cudaMemcpyDeviceToHost);
 
  cudaMemcpy(hfA, dfA, sizeof(float)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hfB, dfB, sizeof(float)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hfC, dfC, sizeof(float)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After atomic Exch intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hiA[%u]: %d\n", i, hiA[i]);
    printf("hiB[%u]: %d\n", i, hiB[i]);
    printf("hiC[%u]: %d\n", i, hiC[i]);
    printf("****** \n");
    printf("hA[%u]: %u\n", i, hA[i]);
    printf("hB[%u]: %u\n", i, hB[i]);
    printf("hC[%u]: %u\n", i, hC[i]);
    printf("****** \n");
    printf("hullA[%u]: %llu\n", i, hullA[i]);
    printf("hullB[%u]: %llu\n", i, hullB[i]);
    printf("hullC[%u]: %llu\n", i, hullC[i]);
    printf("****** \n");
    printf("hfA[%u]: %f\n", i, hfA[i]);
    printf("hfB[%u]: %f\n", i, hfB[i]);
    printf("hfC[%u]: %f\n", i, hfC[i]);
    printf("------ \n");
  } 

  cudaFree(diA);
  cudaFree(diB);
  cudaFree(diC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dullA);
  cudaFree(dullB);
  cudaFree(dullC);
  cudaFree(dfA);
  cudaFree(dfB);
  cudaFree(dfC);
}
