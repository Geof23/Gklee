#include <cstdio>

#define N 32

__global__ void k(volatile int* in)
{
  __shared__ int volatile smem[N];
  __shared__ int volatile tmem[N];


  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  smem[idx] = in[idx];
  tmem[idx] = smem[N-idx-1];
  
  in[idx] = tmem[idx];
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));

  for(int i = 0; i < N; i++)
    in[i]=i;

  int* din;
  cudaMalloc((void**)&din, N*sizeof(int));
  
  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);

  k<<<1,N>>>(din);

  cudaMemcpy(in, din, N*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    printf("%d ", in[i]);
  printf("\n");

  free(in); cudaFree(din);
}