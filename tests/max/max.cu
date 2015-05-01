#include <cstdio>
#include <cstdlib>

#define N 8

__global__ void mmax(int* in, int* out)
{
  // ask Mark, I have no idea - Ian
  int idx = threadIdx.x + blockDim.x*blockIdx.x;

  __shared__ int rslt[N];
  rslt[idx] = in[idx];
  
  int lim = N/2;
  int temp = 0;
  while (lim > 0) {
    __syncthreads();
    if(idx < lim) {
      temp = max(rslt[2*idx], rslt[2*idx + 1]);
    }
 
    __syncthreads();
    rslt[idx] = temp;
    lim /= 2;
  }
  if (idx == 0)
    *out = rslt[0];
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));
  
  for(int i = 0; i < N; i++) {
    in[i] = rand() % 256;
    printf("%d ", in[i]);
  }
  printf("\n");
  
  int* din;
  cudaMalloc((void**) &din, N*sizeof(int));
  cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice);
  int* dout;
  cudaMalloc((void**) &dout, sizeof(int));
  mmax<<<1,N>>>(din,dout);

  int rslt;
  cudaMemcpy(&rslt, dout, sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d\n", rslt);

  cudaFree(din); cudaFree(dout); free(in);
}