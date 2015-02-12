#include <cstdio>

__global__ void iwarp(int* out)
{
  volatile int* vout = out;
  *vout = threadIdx.x;
}

int main()
{
  int* din;
  cudaMalloc((void**)&din, sizeof(int));
  int in = 0;
  cudaMemcpy(din, &in, sizeof(int), cudaMemcpyHostToDevice);
  iwarp<<<1,16>>>(din);
  int output;
  cudaMemcpy(&output, din, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", output);
}