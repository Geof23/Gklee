#include <cstdio>
#include <klee/klee.h>

#define N 8

__global__ void k(int* in)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // Gkleep reports out of bound pointer...
  // Gkleep omits the WW race
  // Causes gkleep to use one flow per thread.
  
  int write_val = N-tid-1;

  for(int i = 0; i <= tid; i++)
    in[tid] = write_val;

  klee_assume(0 <= write_val && write_val < N);
  __syncthreads();

  klee_assume(0 <= in[tid] && in[tid] < N);
  klee_assume(0 <= tid && tid < N);

  for(int i = 0; i < tid; i++)
    if(tid % 2 == 1)
      in[tid] = in[in[tid]];

  __syncthreads();

  for(int i = 0; i <= tid; i++)
    write_val = in[tid];

  __syncthreads();
  for(int i = 0; i < tid; i++)
    in[write_val] = tid;
}

int main()
{
  int* din;
  cudaMalloc((void**) &din, sizeof(int)*N);
  k<<<1,N>>>(din);

  int in[N];
  cudaMemcpy(&in, din, sizeof(int)*N, cudaMemcpyDeviceToHost);
  for(int i = 0; i < N; i++)
    printf("%4d, ", in[i]);
  printf("\n");

  cudaFree(din);
  return 0;
}