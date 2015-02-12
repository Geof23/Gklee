// Tests executing two kernels, with host code between kernel launches.

#include <cstdio>

#define N 100

__global__ void kernel1(int* in, int* out)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < N)
    out[idx] = in[idx] + 1;
}

__global__ void kernel2(int*in, int*out)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx < N)
    out[idx] = in[idx]*in[idx];
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));
  for(int i = 0; i < N; i++)
    in[i] = i;
  int* din;
  int* dout;
  cudaMalloc((void**)&din, N*sizeof(int));
  cudaMalloc((void**)&dout, N*sizeof(int));

  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);
  kernel1<<<1,N>>>(din, dout);

  cudaMemcpy(in, dout, N*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    in[i]--;

  kernel2<<<1,N>>>(din, dout);
  
  cudaMemcpy(in, dout, N*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 1; i < N; i++)
    {
      in[i] = in[i]/i;
      printf("%d ", in[i]);
    }
  printf("\n");
}