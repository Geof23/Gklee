#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM 4

__global__
void sinKernel(float *hfA, float *hfB) {
  unsigned tid = threadIdx.x;
  hfB[tid] = __sinf(hfA[tid]); 
  hfB[tid] = sinf(hfB[tid]); 
}

__global__
void sindKernel(double *hdA, double *hdB) {
  unsigned tid = threadIdx.x;
  hdB[tid] = sin(hdA[tid]); 
}

__global__
void cosKernel(float *hfA, float *hfB) {
  unsigned tid = threadIdx.x;
  hfB[tid] = __cosf(hfA[tid]);
  hfB[tid] = cosf(hfB[tid]);
}

__global__
void cosdKernel(double *hdA, double *hdB) {
  unsigned tid = threadIdx.x;
  hdB[tid] = cos(hdA[tid]);
}

__global__
void tanKernel(float *hfA, float *hfB) {
  unsigned tid = threadIdx.x;
  hfB[tid] = __tanf(hfA[tid]);
  hfB[tid] = tanf(hfB[tid]);
}

__global__
void tandKernel(double *hdA, double *hdB) {
  unsigned tid = threadIdx.x;
  hdB[tid] = tan(hdA[tid]);
}

__global__
void fdivideKernel(float *hfA, float *hfB, float *hfC) {
  unsigned tid = threadIdx.x;
  hfC[tid] = fdividef(hfA[tid], hfB[tid]);
  hfC[tid] = __fdividef(hfA[tid], hfB[tid]);
}

__global__
void ddivideKernel(double *hdA, double *hdB, double *hdC) {
  unsigned tid = threadIdx.x;
  hdC[tid] = fdivide(hdA[tid], hdB[tid]);
}

__global__
void sincosfKernel(float *hfA, float *hfB, float *hfC) {
  unsigned tid = threadIdx.x;
  __sincosf(hfA[tid], &hfB[tid], &hfC[tid]);
  sincosf(hfA[tid], &hfB[tid], &hfC[tid]);
}

__global__
void sincosdKernel(double *hdA, double *hdB, double *hdC) {
  unsigned tid = threadIdx.x;
  sincos(hdA[tid], hdB + tid, hdC + tid); 
}

__global__
void sinhKernel(double *hdA, double *hdB) {
  unsigned tid = threadIdx.x;
  hdB[tid] = sinh(hdA[tid]);
}

__global__
void asinhKernel(double *hdA, double *hdB) {
  unsigned tid = threadIdx.x;
  hdB[tid] = asinh(hdA[tid]);
}

int main(int argv, char **argc) {
  float hfA[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  float hfB[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  float hfC[NUM] = {1.0, 2.0, 3.0, 4.0};

  double hdA[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  double hdB[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  double hdC[NUM] = {1.0, 2.0, 3.0, 4.0};

#ifdef _SYM
  klee_make_symbolic(hfA, sizeof(float)*NUM, "hfA");
  klee_make_symbolic(hfB, sizeof(float)*NUM, "hfB");
  klee_make_symbolic(hfC, sizeof(float)*NUM, "hfC");

  klee_make_symbolic(hdA, sizeof(double)*NUM, "hdA");
  klee_make_symbolic(hdB, sizeof(double)*NUM, "hdB");
  klee_make_symbolic(hdC, sizeof(double)*NUM, "hdC");
#else
  srand ( time(NULL) );
  for (unsigned i = 0; i < NUM; i++) {
    hfA[i] = (float)rand();  
    hfB[i] = (float)rand();  
    hfC[i] = (float)rand();  

    hdA[i] = (double)rand();  
    hdB[i] = (double)rand();  
    hdC[i] = (double)rand();  
  }
#endif

  float *dfA, *dfB, *dfC; 
  cudaMalloc((void**)&dfA, sizeof(float)*NUM);
  cudaMalloc((void**)&dfB, sizeof(float)*NUM);
  cudaMalloc((void**)&dfC, sizeof(float)*NUM);

  cudaMemcpy(dfA, hfA, sizeof(float)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(dfB, hfB, sizeof(float)*NUM, cudaMemcpyHostToDevice);

  sinKernel<<<1, NUM>>>(dfA, dfB);

  cudaMemcpy(hfB, dfB, sizeof(float)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After float sin intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hfB[%u]: %f\n", i, hfB[i]);
  } 

  cosKernel<<<1, NUM>>>(dfA, dfB); 
  cudaMemcpy(hfB, dfB, sizeof(float)*NUM, cudaMemcpyDeviceToHost);

  printf("After float cos intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hfB[%u]: %f\n", i, hfB[i]);
  }

  tanKernel<<<1, NUM>>>(dfA, dfB);
  cudaMemcpy(hfB, dfB, sizeof(float)*NUM, cudaMemcpyDeviceToHost);

  printf("After float tan intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hfB[%u]: %f\n", i, hfB[i]);
  } 

  double *ddA, *ddB, *ddC; 
  cudaMalloc((void**)&ddA, sizeof(double)*NUM);
  cudaMalloc((void**)&ddB, sizeof(double)*NUM);
  cudaMalloc((void**)&ddC, sizeof(double)*NUM);

  cudaMemcpy(ddA, hdA, sizeof(double)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(ddB, hdB, sizeof(double)*NUM, cudaMemcpyHostToDevice);

  sindKernel<<<1, NUM>>>(ddA, ddB);

  cudaMemcpy(hdB, ddB, sizeof(double)*NUM, cudaMemcpyDeviceToHost);
 
  printf("After double sin intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdB[%u]: %f\n", i, hdB[i]);
  } 

  cosdKernel<<<1, NUM>>>(ddA, ddB); 
  cudaMemcpy(hdB, ddB, sizeof(double)*NUM, cudaMemcpyDeviceToHost);

  printf("After double cos intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdB[%u]: %f\n", i, hdB[i]);
  } 

  tandKernel<<<1, NUM>>>(ddA, ddB); 
  cudaMemcpy(hdB, ddB, sizeof(double)*NUM, cudaMemcpyDeviceToHost);

  printf("After double tan intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdB[%u]: %f\n", i, hdB[i]);
  } 

  fdivideKernel<<<1, NUM>>>(dfA, dfB, dfC); 

  cudaMemcpy(hfC, dfC, sizeof(float)*NUM, cudaMemcpyDeviceToHost);

  printf("After float dividef intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hfC[%u]: %f\n", i, hfC[i]);
  } 

  ddivideKernel<<<1, NUM>>>(ddA, ddB, ddC); 

  cudaMemcpy(hdC, ddC, sizeof(double)*NUM, cudaMemcpyDeviceToHost);

  printf("After double dividef intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdC[%u]: %f\n", i, hdC[i]);
  } 

  sincosfKernel<<<1, NUM>>>(dfA, dfB, dfC); 

  cudaMemcpy(hfB, dfB, sizeof(float)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hfC, dfC, sizeof(float)*NUM, cudaMemcpyDeviceToHost);

  printf("After float sincos intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hfB[%u]: %f\n", i, hfB[i]);
    printf("hfC[%u]: %f\n", i, hfC[i]);
  } 

  sincosdKernel<<<1, NUM>>>(ddA, ddB, ddC); 

  cudaMemcpy(hdB, ddB, sizeof(double)*NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(hdC, ddC, sizeof(double)*NUM, cudaMemcpyDeviceToHost);

  printf("After double sincos intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdB[%u]: %f\n", i, hdB[i]);
    printf("hdC[%u]: %f\n", i, hdC[i]);
  } 

  /*hdA[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  hdB[NUM] = {1.0, 2.0, 3.0, 4.0}; 
  hdC[NUM] = {1.0, 2.0, 3.0, 4.0};*/

  cudaMemcpy(ddA, hdA, sizeof(double)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(ddB, hdB, sizeof(double)*NUM, cudaMemcpyHostToDevice);
  
  sinhKernel<<<1, NUM>>>(ddA, ddB); 

  cudaMemcpy(hdB, ddB, sizeof(double)*NUM, cudaMemcpyDeviceToHost);

  printf("After double sinh intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdB[%u]: %f\n", i, hdB[i]);
  } 

  asinhKernel<<<1, NUM>>>(ddA, ddB); 

  cudaMemcpy(hdB, ddB, sizeof(double)*NUM, cudaMemcpyDeviceToHost);

  printf("After double asinh intrinsic: \n");
  for (unsigned i = 0; i < NUM; i++) {
    printf("hdB[%u]: %f\n", i, hdB[i]);
  } 

  cudaFree(dfA);
  cudaFree(dfB);
  cudaFree(dfC);

  cudaFree(ddA);
  cudaFree(ddB);
  cudaFree(ddC);
}
