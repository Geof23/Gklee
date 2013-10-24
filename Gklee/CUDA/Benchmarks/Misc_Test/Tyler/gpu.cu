//Tyler Sorensen
//tylersorensen3221@hotmail.com
//University of Utah

/***********************************************************
 *Really strange behavior from Cuda program
 *
 *This is a simple contrived example I came up with
 *to test GKLEE's -pure-canonical-schedule flag. I made a simple
 *kernel that wouldn't race with lock-step warp behavior, but would
 *under the canonical schedule.
 *
 *Basic overview: 
 *kernel takes 2 vectors of size 64 (launched with 1 block, 64 threads,
 *one thread per element). It adds
 *x and y together and stores the result in y. Then it checks
 *to see if it is a thread at the boundery of a warp (e.g. thread 31,
 *thread 63 etc). If it is not, then it stores the arbitrary flag
 *value 1111 in y at a location one spot ahead of it's original thread's
 *location. Obvious this races under cononical schedule, but should
 *be okay with lock-step scheduling
 *
 *Results: When I run this on my laptop's GT 540M, I simply get 
 *the results of x+y stored in y, no sign of the flag value 1111 at all.
 *However GKLEE reports no race under
 *warp scheduling.
 *If it is executing lock step, then everything should be 1111 except for
 *the thread bounderies (31 and 63). If I put a __syncthreads(); between the two
 *store instructions, then it outputs the expected value. Also when I tried
 *to run it in debug to step through (compiled with -g -G), 
 *I get correct values. The same behavior
 *is reported on the GTX 480 on Formal. Both are using CUDA 4.1
 *When Formal had CUDA 4.0 it was displaying the correct values.
 *
 *What do you guys think?
 */

#include <iostream>
using namespace std;
#define SIZE 64

//Kernel
__global__ void kernel(int* x, int* y)
{
  //Get the index (64 threads, arrays are 64 long,
  //one thread per index)
  int index = threadIdx.x;
  y[index] = x[index] + y[index];

  //Will output the expected value if this is included,
  //but even without it, gklee reports NO RACE
  //__syncthreads();
  
  //Make sure we aren't a warp boundery, then set
  //the flag value
  if (index != 63 && index != 31)
    y[index+1] = 1111;
       
}

int main( void ) {



  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  //Make sure we do actually have 32 threads in a warp
  cout << "Threads in a warp: " << prop.warpSize << endl;
  cout << "compute cap: " << prop.major << " " << prop.minor << endl;
  
  //All the vectors
  int *vector_hx, *vector_dx;
  int *vector_hy, *vector_dy;
  
  //Allocating Memory on the host and device
  cudaMalloc( (void**)&vector_dx, sizeof(int)*SIZE);

  //All the cout's were checking for errors, there weren't any
  //so I commented them out.
  //cout << er << endl;
  cudaMalloc( (void**)&vector_dy, sizeof(int)*SIZE);
  //cout << er << endl;

  vector_hx = new int[SIZE];
  vector_hy = new int[SIZE];
  
  //Arbitrarily filling the vectors, location 0 is assigned to 0
  //location 1 is assigned to 1 etc. 
  //So x = y = 0,1,2,3,4... and  x + y = 0,2,4,6,8 ... 126
  for (int i = 0; i < SIZE; i++)
    {
      vector_hx[i] = i;
      vector_hy[i] = i;	    
    }

  //Copy memory over to device
  cudaMemcpy(vector_dx, vector_hx, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
  //cout << er << endl;
  cudaMemcpy(vector_dy, vector_hy, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
  //cout << er << endl;

  //Call the kernel
  kernel<<<1,64>>>(vector_dx, vector_dy);

  //Copy memory back over and output results
  cudaMemcpy(vector_hy, vector_dy, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
  //cout << er << endl;
  for (int i = 0; i < SIZE; i++)
    cout << vector_hy[i] << "\n";

  //cleanup
  delete vector_hx;
  delete vector_hy;

  cudaFree(vector_dx);
  //cout << er << endl;
  cudaFree(vector_dy);
  //cout << er << endl;

  return 0;
}
