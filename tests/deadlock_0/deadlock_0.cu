#include <cstdio>

#define N 50
#define B 2
#define T 32

__global__ void dl(int* in)
{
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < N)
  {
    if(in[tid] % 2 == 0)
      in[tid]++;

    __syncthreads(); // ouch

    int sum = in[tid];
    if(tid > 0)
      sum += in[tid-1];
    if(tid < N - 1)
      sum += in[tid+1];
    in[tid] = sum / 3;
  }
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));
  
  for(int i = 0; i < N; i++)
    in[i] = i;
  
  int* din;
  cudaMalloc((void**)&din, N*sizeof(int));
  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);

  dl<<<B,T>>>(din);

  cudaMemcpy(in, din, N*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    printf("%d ", in[i]);
  printf("\n");
  free(in); cudaFree(din);
}