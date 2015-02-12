// Exhibits a bank conflict.
// Gklee and Gkleep both detect this.

#include <cstdio>

#define N 32

__global__ void bc(char* in, char* out)
{
  __shared__ int smem[512];

  int tid = threadIdx.x;
  
  smem[tid*2]=in[tid];
  __syncthreads();
  smem[tid*4]=in[tid];
  __syncthreads();
  smem[tid*8]=in[tid];
  __syncthreads();

  int x = smem[tid * 2]; // 2-way bank conflicts
  int y = smem[tid * 4]; // 4-way bank conflicts
  int z = smem[tid * 8]; // 8-way bank conflicts
  
  int m = max(max(x,y),z);
  out[tid] = m;
}

int main()
{
  char* in = (char*) malloc(N*sizeof(char));
  for(int i = 0; i < N; i++)
    in[i] = i;
  
  char* din, * dout;
  cudaMalloc((void**) &din, N*sizeof(char));
  cudaMalloc((void**) &dout, N*sizeof(char));
  
  cudaMemcpy(din, in, N*sizeof(char), cudaMemcpyHostToDevice);
  
  bc<<<1,N>>>(din,dout);
  
  cudaMemcpy(in, dout, N*sizeof(char), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    printf("%d ", in[i]);
  printf("\n");
  free(in); cudaFree(din); cudaFree(dout);
}