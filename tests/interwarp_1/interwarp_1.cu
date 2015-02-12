#include <cstdio>

#define N 32

__global__ void iwarp(int* out)
{
  __shared__ volatile int smem[32];
  volatile int* vout = out;
  int idx = threadIdx.x;
  smem[idx] = vout[idx];

  if(idx % 2 == 0)
    smem[idx] = 1;
  else
    smem[idx-1] = 0;
  vout[idx] = smem[idx];
}

int main()
{
  int* din;
  cudaMalloc((void**)&din, N*sizeof(int));
  int in[N];
  for(int i = 0; i < N; i++)
    in[i] = 0;
  cudaMemcpy(din, &in, N*sizeof(int), cudaMemcpyHostToDevice);
  iwarp<<<1,N>>>(din);
  int output[N];
  cudaMemcpy(&output, din, N*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < N; i++)
	printf("%d ", output[i]);
  printf("\n");
}