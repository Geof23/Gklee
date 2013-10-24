// These examples came with the Allinea DDT Tutorials - without any headers or copyright notices.
// Ganesh has copied them over and modified them merely for the purposes of trying to test GKLEE well.
// 12/3/11.
//
// For the symbolic test being conducted, see the last line.

#include "cutil.h"
#include "klee.h"
#include <stdio.h>
#include <stdlib.h>

#define NITEMS 4

bool verify(int data[], int ROM_data[], int length)
{
  // Do a prefix-sum sequentially onto ROM_data
  for (int i = 1; i < length; ++i)
    {
      ROM_data[i] += ROM_data[i-1];
      printf("ROM_data[%d]=%d\n", i, ROM_data[i]);
    }

  // Now, verify
  for (int i = 1 ; i < length; ++i)
    {
      if (data[i] != ROM_data[i] )
	{ printf("error, the results disagree at location %d\n", i); return false; }
    }
  return true;
}

#define DUMP(x) printf("%s %d\n", #x, props.x)

// void dumpCUDAProps(cudaDeviceProp & props)
// {
// 	DUMP(canMapHostMemory);
// 	DUMP(clockRate);
// 	DUMP(computeMode);
// 	DUMP(deviceOverlap);
// 	DUMP(integrated);
// 	DUMP(kernelExecTimeoutEnabled);
// 	DUMP(major);
// 	DUMP(maxGridSize[0]);
// 	DUMP(maxGridSize[1]);
// 	DUMP(maxGridSize[2]);
// 	DUMP(maxThreadsDim[0]);
// 	DUMP(maxThreadsDim[1]);
// 	DUMP(maxThreadsDim[2]);
// 	DUMP(maxThreadsPerBlock);
// 	DUMP(memPitch);
// 	DUMP(minor);
// 	DUMP(multiProcessorCount);
// 	printf("name %s\n", props.name);
// 	DUMP(regsPerBlock);
// 	DUMP(sharedMemPerBlock);
// 	DUMP(textureAlignment);
// 	DUMP(totalConstMem);
// 	DUMP(totalGlobalMem);
// 	DUMP(warpSize);
// 
// }

//#define BLOCK_SIZE 64 
#define BLOCK_SIZE 32 

__global__ void prefixsumblock(int *in, int *out, int length)
{
	int x = threadIdx.x + blockIdx.x * BLOCK_SIZE;

	if (x < length)
		out[x] = in[x];

	__syncthreads();

	for ( int i = 1; i < BLOCK_SIZE; i <<= 1)
	{
		if (threadIdx.x + i < BLOCK_SIZE && x + i < length) 
		{
			out[x + i] = in[x] + in[x + i];
		}
		__syncthreads();

		if (x < length)
			in[x] = out[x];

		__syncthreads();
	}
}

__global__ void correctsumends(int *ends, int *in, int *out)
{
	int x = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	int end = ends[blockIdx.x];
	out[x] = in[x] + end;
}


__global__ void gathersumends(int *in, int *out)
{
	int x = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (x > 0)
		out[x] = in[x * BLOCK_SIZE - 1];
	else 
		out[x] = 0;
}

__global__ void zarro(int *data, int length)
{
	int x = threadIdx.x + blockIdx.x * BLOCK_SIZE;

	if (x < length)
		data[x] = 0;
}


void prefixsum(int* in, int *out, int length)
{
	int blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;


	//dim3 dimGrid(blocks, 1, 1);
	__modify_Grid(blocks, 1);

	//dim3 dimBlock(BLOCK_SIZE, 1, 1);
	__modify_Block(BLOCK_SIZE, 1, 1);

	__begin_GPU();
	zarro(out, length);
	__end_GPU();

	__begin_GPU();
	prefixsumblock(in, out, length);
	__end_GPU();

	if (blocks > 1) {
		int *devEnds;
		int *devTmpEnds;

		cudaMalloc((void**) &devEnds, blocks * sizeof(int));
		cudaMalloc((void**) &devTmpEnds, blocks * sizeof(int));

		int subblocks = (blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

		//dim3 subgrid(subblocks, 1, 1);
		__modify_Grid(subblocks, 1);

		//dim3 subblock(BLOCK_SIZE, 1, 1);
		__modify_Block(BLOCK_SIZE, 1);

		__begin_GPU();
		gathersumends(out, devEnds);
		__end_GPU();

		prefixsum(devEnds, devTmpEnds, blocks);

		cudaFree(devEnds);

		__begin_GPU();		
		correctsumends(devTmpEnds, in, out);
		__end_GPU();

		cudaFree(devTmpEnds);
	}
}


void cudasummer(int data[], int length)
{

	int *devIn, *devOut;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	cudaMalloc((void**) &devIn, length * sizeof(int));
	cudaMalloc((void**) &devOut, length * sizeof(int));
	cudaMemcpy(devIn, data, length * sizeof(int), cudaMemcpyHostToDevice);

	prefixsum(devIn, devOut, length);

	cudaMemcpy(data, devOut, length * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(devIn);
	cudaFree(devOut);

	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);

	float t;

	//cudaEventElapsedTime(&t, start, stop);

	printf("Elapsed time %3fms\n", t);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

}

// void devicesDump()
// {
// 
// 	int deviceCount;
// 	cudaGetDeviceCount(&deviceCount);
// 	int device;
// 	for (device = 0; device < deviceCount; ++device) {
// 		cudaDeviceProp deviceProp;
// 		cudaGetDeviceProperties(&deviceProp, device);
// 		dumpCUDAProps(deviceProp);
// 	}
// }


int main(int argc, char *argv[]) 
{

	int length;
	if (argc < 2) {
	  length = NITEMS;
	}	
	else length = atoi(argv[1]);

	int *data = (int*) malloc(length * sizeof(int));
	int *ROM_data = (int*) malloc(length * sizeof(int)); 

	klee_make_symbolic(data, NITEMS * sizeof(int), "data_symb");

	//	for (int i = 0; i < length-1; ++i) {
	//		data[i] = i;                      //could be rand(); later
	//		ROM_data[i] = i;
	//	}

	//	devicesDump();

	klee_assume(data[0] != data[1]);
 
	// Copy all the symbolic stuff in!
	for (int i = 0; i < length; ++i)
	  {
	    ROM_data[i] = data[i];
	  }

	if (data[0] < data[1])
	  { printf("a\n");
	    cudasummer(data, length); 
	  }
	else
	  { printf("b\n");
	    cudasummer(data, length);
	  }

	if (length < 1000) 
		for (int i = 0 ; i < length; ++i)
		{
			printf("%d\n", data[i]);
		}
	verify(data, ROM_data, length);
}

// prefix5.C : Making final location of initial array symbolic - but I also fixed the verify function !!
// Forcing choices other than 0



