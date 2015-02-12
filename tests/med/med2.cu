#include <cstdlib>
#include <iostream>
#include <time.h>

#define DIM1 3
#define DIM2 3

__global__ void avg(float* in, float* out, int radius)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < DIM1 * DIM2)
    {
      int x = tid / DIM1;
      int y = tid % DIM2;
      
      float count = 0;
      float val = 0;

      for(int i = -1 * radius; i <= radius; i++)
	{
	  int nx = i + x;
	  if(nx >= 0 && nx < DIM1)
	    for(int j = -1 * radius; j <= radius; j++)
	      {
		int ny = j + y;
		if(i*i + j*j <= radius * radius && ny >= 0 && ny < DIM2)
		  val += in[nx * DIM1 + ny], count++;
	      }
	}
      out[tid] = val/count;
    }  
}

int main()
{
  float* in = (float*)malloc(DIM1*DIM2*sizeof(float));
  srand(time(NULL));

  for(int i = 0; i < DIM1*DIM2; i++)
    in[i] = rand()%1000;

 /* std::cout << "Original:" << std::endl;
  for(int i = 0; i < DIM1; ++i) {
    for(int j = 0; j < DIM2; ++j)
      std::cout << in[i*DIM1 + j] << " ";
    std::cout << std::endl;
  }*/


  float* din;
  float* dout;
  cudaMalloc((void**)&din, DIM1 * DIM2 * sizeof(float));
  cudaMalloc((void**)&dout, DIM1 * DIM2 * sizeof(float));

  cudaMemcpy(din, in, DIM1 * DIM2 * sizeof(float), cudaMemcpyHostToDevice);
  int TPB = 9;
  avg<<<(DIM1*DIM2 + TPB - 1)/TPB, TPB>>>(din, dout, 2);
  
  float* out = (float*)malloc(DIM1*DIM2*sizeof(float));

  cudaMemcpy(out, dout, DIM1 * DIM2 * sizeof(float), cudaMemcpyDeviceToHost);


  std::cout << "Averaged:" << std::endl;

  for(int x = 0; x < DIM1; x++)
  {
    if (x % 10 == 0)
      {
	for(int y = 0; y < DIM2; y++)
	  if(y % 10 == 0)
	    std::cout << out[x*DIM1+y] << " ";
	std::cout << std::endl;
      }
  }

  cudaFree(din); cudaFree(dout);

  free(in); free(out);
}
