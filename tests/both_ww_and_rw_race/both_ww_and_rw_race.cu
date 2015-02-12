// Exhibits a data race (RW, WW) in global memory.
// Gklee and Gkleep both detect.

#include <iostream>

#define N 50
#define T 128
#define B 2

__global__ void colonel(int* in)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  in[tidx%N]++;
}

int main()
{
  int* in = (int*) calloc(N,sizeof(int));
  int* din;
  cudaMalloc((void**)&din,N*sizeof(int));
  
  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);
  colonel<<<B,T>>>(din);
  
  cudaMemcpy(in, din, N*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    std::cout << in[i] << " ";
  std::cout << std::endl;
  free(in); cudaFree(din);
}