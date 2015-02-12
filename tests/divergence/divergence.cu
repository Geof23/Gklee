#include <cstdio>

#define N 50
#define T 128
#define B 2

__global__ void div(int* in, int* out)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(tid < N)
  {
    if(tid % 2 == 0)
      out[tid] = in[tid] - 1;
    else
      out[tid] = in[tid] + 1;
  }
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));

  for(int i = 0; i < N; i++)
    in[i] = i;

  int* din, *dout;
  cudaMalloc((void**)&din, N*sizeof(int));
  cudaMalloc((void**)&dout,N*sizeof(int));
  
  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);

  div<<<B,T>>>(din,dout);

  cudaMemcpy(in, dout, N*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < N; i++)
    printf("%d ", in[i]);
  printf("\n");

  free(in); cudaFree(din); cudaFree(dout);
}