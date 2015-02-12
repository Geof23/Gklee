#include <cstdio>
// Gklee does not detect BC, Gkleep does detect.
// Gkleep detects oob access, Gklee does not.
#define N 128

__device__ int p(int n)
{
  if(n < 0)
    n = -1 * n;
  if(n == 0 || n == 1)
    return 0;
  for(int i = 2; i*i<=n;i++)
    if(n%i == 0)
      return 0;
  return 1;
}

__global__ void compact(int* g_in, int* g_out)
{
  volatile __shared__ int data[N];
  volatile __shared__ int out[N];
  volatile __shared__ int idx[N];
  volatile __shared__ int flag[N];
  unsigned int offset, d, left, right, temp;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Copy input data to shared data
  data[tid] = g_in[tid];
  out[tid] = 0;
  idx[tid] = 0;
  __syncthreads();

  flag[tid] = p(data[tid]);
  __syncthreads();

  if(tid < N/2){
    idx[2*tid] = flag[2*tid];
    idx[2*tid+1] = flag[2*tid+1];
  }

  // upsweep
  offset = 1;
  for(d = N/2; d > 0; d /= 2) {
    __syncthreads();
    if(tid < d) {
      left = offset * (2 * tid + 1) - 1;
      right = offset * (2 * tid + 2) - 1;
      idx[right] += idx[left];
    }
    offset *= 2;
  }
  
  // Downsweep
  if(tid == 0)
    idx[N-1] = 0;

  for( d = 1; d < N; d *= 2) {
    offset /= 2;
    __syncthreads();
    if(tid < d) {
      left = offset * (2 * tid + 1) - 1;
      right = offset * (2 * tid + 2) - 1;
      temp = idx[left];
      idx[left] = idx[right];
      idx[right] += temp;
    }
  }
  __syncthreads();

  if(tid < N && flag[tid] == 1)
    out[idx[tid]] = data[tid];

  __syncthreads();

  if(tid < N)
    g_out[tid] = out[tid];
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));
  for(int i = 0; i < N; i++)
    in[i] = i;
  
  int* din, *dout;
  cudaMalloc((void**) &din, N*sizeof(int));
  cudaMalloc((void**) &dout, N*sizeof(int));
  
  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);
  
  compact<<<1,N>>>(din, dout);

  cudaMemcpy(in, dout, N*sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    printf("%d ", in[i]);
    printf("\n");
  free(in); cudaFree(din); cudaFree(dout);
}