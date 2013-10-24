
#ifndef FWT_KERNEL_CUH
#define FWT_KERNEL_CUH
#ifndef fwt_kernel_cuh
#define fwt_kernel_cuh

#include "stdio.h"

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory 
// combined radix-2 + radix-4 Fast Walsh Transform 
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ void fwtBatch1Kernel(int *d_Output, int *d_Input, int log2N){
    const int    N = 1 << log2N;
    const int base = blockIdx.x << log2N;

    //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ int s_data[];
    int *d_Src = d_Input  + base;
    int *d_Dst = d_Output + base;

    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        s_data[pos] = d_Src[pos];

    //Main radix-4 stages
    const int pos = threadIdx.x;
    for(int stride = N >> 2; stride > 0; stride >>= 2){
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        __syncthreads();
        int D0 = s_data[i0];
        int D1 = s_data[i1];
        int D2 = s_data[i2];
        int D3 = s_data[i3];

        int T;
        T = D0; D0         = D0 + D2; D2         = T - D2;
        T = D1; D1         = D1 + D3; D3         = T - D3;
        T = D0; s_data[i0] = D0 + D1; s_data[i1] = T - D1;
        T = D2; s_data[i2] = D2 + D3; s_data[i3] = T - D3;
    }

    //Do single radix-2 stage for odd power of two
    if(log2N & 1){
        __syncthreads();
        for(int pos = threadIdx.x; pos < N / 2; pos += blockDim.x){
            int i0 = pos << 1;
            int i1 = i0 + 1;

            int D0 = s_data[i0];
            int D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        d_Dst[pos] = s_data[pos];
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void fwtBatch2Kernel(
    int *d_Output,
    int *d_Input,
    int stride
){
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    int *d_Src = d_Input  + blockIdx.y * N;
    int *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    int D0 = d_Src[i0];
    int D1 = d_Src[i1];
    int D2 = d_Src[i2];
    int D3 = d_Src[i3];

    int T;
    T = D0; D0        = D0 + D2; D2        = T - D2;
    T = D1; D1        = D1 + D3; D3        = T - D3;
    T = D0; d_Dst[i0] = D0 + D1; d_Dst[i1] = T - D1;
    T = D2; d_Dst[i2] = D2 + D3; d_Dst[i3] = T - D3;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(int *d_Data, int M, int log2N){
    const int THREAD_N = 256;

    int N = 1 << log2N;
    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);
    for(; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2){
        fwtBatch2Kernel<<<grid, THREAD_N>>>(d_Data, d_Data, N / 4);
        cutilCheckMsg("fwtBatch2Kernel() execution failed\n");
    }

    fwtBatch1Kernel<<<M, N / 4, N * sizeof(int)>>>(
        d_Data,
        d_Data,
        log2N
    );
    cutilCheckMsg("fwtBatch1Kernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void modulateKernel(int *d_A, int *d_B, int N){
    int        tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    int     rcpN = 1.0f / (int)N;

    for(int pos = tid; pos < N; pos += numThreads)
        d_A[pos] *= d_B[pos] * rcpN;
}

//Interface to modulateKernel()
void modulateGPU(int *d_A, int *d_B, int N){
    modulateKernel<<<128, 256>>>(d_A, d_B, N);
}

///////////////////////////////////////////////////////////////////////////////
// Straightforward Walsh Transform: used to test both CPU and GPU FWT
// Slow. Uses doubles because of straightforward accumulation
///////////////////////////////////////////////////////////////////////////////
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N){
    const int N = 1 << log2N;

    for(int i = 0; i < N; i++){
        double sum = 0;

        for(int j = 0; j < N; j++){
            //Walsh-Hadamar quotent
            double q = 1.0;
            for(int t = i & j; t != 0; t >>= 1)
                if(t & 1) q = -q;

            sum += q * h_Input[j];
        }

        h_Output[i] = (float)sum;
    }
}

//***********************************************************************
//! The Driver
//***********************************************************************

int main() {
  unsigned int d_Histogram[BIN_COUNT];
  unsigned int d_Data[DATA_N];
  
  unsigned int h_result[BIN_COUNT];

  unsigned int data[10];
  klee_make_symbolic(data, sizeof(data), "input");
  // klee_make_symbolic(d_Data, sizeof(d_Data), "input");
  for (int i = 0; i < 5; i++)
    d_Data[i] = data[i];

  // const int histogramCount = iDivUp(DATA_N / 4, THREAD_N * 63);
  histogram64Kernel(d_Histogram, (unsigned int *)d_Data, DATA_N / 4);
  
  // post-condition
  printf("Now check the post-condition. \n");
  histogram64CPU(h_result, (unsigned int *)d_Data, DATA_N / 4);
  
  for (int i = 0; i < BIN_COUNT; i++) {
    if (h_result[i] != d_Histogram[i]) {
      printf("Incorrect when i = %d: %d != %d\n", 
	     i, h_result[i], d_Histogram[i]);
    }
  }

  // histogram64Kernel<<<histogramCount, THREAD_N>>>(d_Histogram,
  // 						  (unsigned int *)d_Data,
  // 						  dataN / 4
  // 						  );
}

#endif
#endif
