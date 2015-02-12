#include <cstdio>

#define N 64
#define B 1
#define T 64

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
     if (abort) exit(code);
   }
}

__global__ void dl(int* in)
{
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // The warps in this block take different paths; the synctreads calls
  // will cause a deadlock.
  if(tid > 31)
  {
    if(in[tid] % 2 == 0)
      in[tid]++;

    __syncthreads();

  }
  else {
    if(in[tid] % 2 == 1)
      in[tid]--;
    
    __syncthreads();
  }
/*  int sum = in[tid];
  if(tid > 0)
    sum += in[tid-1];
  if(tid < N - 1)
      sum += in[tid+1];
      in[tid] = sum / 3; */
}

int main()
{
  int* in = (int*) malloc(N*sizeof(int));
  
  for(int i = 0; i < N; i++)
    in[i] = i;
  
  int* din;
  gpuErrchk(cudaMalloc((void**)&din, N*sizeof(int)));
  gpuErrchk(cudaMemcpy(din, in, N*sizeof(int), cudaMemcpyHostToDevice));

  dl<<<B,T>>>(din);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(in, din, N*sizeof(int), cudaMemcpyDeviceToHost));

  for(int i = 0; i < N; i++)
    printf("%d ", in[i]);
  printf("\n");
  free(in); cudaFree(din);
}