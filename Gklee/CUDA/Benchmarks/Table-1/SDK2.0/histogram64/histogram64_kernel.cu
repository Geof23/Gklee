#include "stdio.h"

#ifdef REPLAY

#include "histogram64_kernel.cpp"

#else

//Threads block size for histogram64Kernel()
//Preferred to be a multiple of 64 (refer to the supplied whitepaper)
#define BIN_COUNT 4 

#define THREAD_N  32

//Total number of possible data values
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))

const int  DATA_N = THREAD_N * BIN_COUNT;
const int  DATA_SIZE = DATA_N * sizeof(unsigned char);

////////////////////////////////////////////////////////////////////////////////
// GPU-specific definitions
////////////////////////////////////////////////////////////////////////////////
//Fast mul on G8x / G9x / G100
#define IMUL(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// If threadPos == threadIdx.x, there are always  4-way bank conflicts,
// since each group of 16 threads (half-warp) accesses different bytes,
// but only within 4 shared memory banks. Having shuffled bits of threadIdx.x 
// as in histogram64GPU(), each half-warp accesses different shared memory banks
// avoiding any bank conflicts at all.
// Refer to the supplied whitepaper for detailed explanations.
////////////////////////////////////////////////////////////////////////////////
inline __device__ void addData64(unsigned char *s_Hist, int threadPos, unsigned int data) {
    s_Hist[threadPos + IMUL(data, THREAD_N)]++;
}

//Per-thread histogram storage
__shared__ unsigned char s_Hist[THREAD_N * BIN_COUNT];

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
__global__ void histogram64Kernel(unsigned int *d_Result, unsigned int *d_Data, int dataN){
    //Encode thread index in order to avoid bank conflicts in s_Hist[] access:
    //each half-warp accesses consecutive shared memory banks
    //and the same bytes within the banks
    const int threadPos = 
        //[31 : 6] <== [31 : 6]
        ((threadIdx.x & (~63)) >> 0) |
        //[5  : 2] <== [3  : 0]
        ((threadIdx.x &    15) << 2) |
        //[1  : 0] <== [5  : 4]
        ((threadIdx.x &    48) >> 4);
    // threadPos: 0 for thread 0, and 32 for thread 8
    //Flush shared memory
    for(int i = 0; i < BIN_COUNT / 4; i++)
      ((unsigned int *)s_Hist)[threadIdx.x + i * THREAD_N] = 0;

    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////
    // Cycle through current block, update per-thread histograms
    // Since only 64-bit histogram of 8-bit input data array is calculated,
    // only highest 6 bits of each 8-bit data element are extracted,
    // leaving out 2 lower bits.
    ////////////////////////////////////////////////////////////////////////////
    for(int pos = IMUL(blockIdx.x, blockDim.x) + threadIdx.x; pos < dataN; pos += IMUL(blockDim.x, gridDim.x)){
        unsigned int data4 = d_Data[pos];
        // d_Data[0] (symbolic) for thread 0, and d_Data[8] (symbolic) for thread 8 ...  
        // s_Hist[threadPos + IMUL(data, THREAD_N)]++;
        // constraint: 0 + ((d_Data[0] << 3) & 2016) = 32 + ((d_Data[8] << 3) & 2016) 
        // d_Data[0] is \x50\xa4\xb8\x84, d_Data[8] is \x4c\xa0\xb4\x80
        addData64(s_Hist, threadPos, (data4 >>  2) & 0x3FU);
        addData64(s_Hist, threadPos, (data4 >> 10) & 0x3FU);
        addData64(s_Hist, threadPos, (data4 >> 18) & 0x3FU);
        addData64(s_Hist, threadPos, (data4 >> 26) & 0x3FU);
    }

    __syncthreads();
    ////////////////////////////////////////////////////////////////////////////
    // Merge per-thread histograms into per-block and write to global memory.
    // Start accumulation positions for half-warp each thread are shifted
    // in order to avoid bank conflicts. 
    // See supplied whitepaper for detailed explanations.
    ////////////////////////////////////////////////////////////////////////////
    if(threadIdx.x < BIN_COUNT){
        unsigned int sum = 0;
        const int value = threadIdx.x;

        const int valueBase = IMUL(value, THREAD_N);
        const int  startPos = (threadIdx.x & 15) * 4;

        //Threads with non-zero start positions wrap around the THREAD_N border
        for(int i = 0, accumPos = startPos; i < THREAD_N; i++){
            sum += s_Hist[valueBase + accumPos];
            if(++accumPos == THREAD_N) accumPos = 0;
        }

	// printf("d_Result[%d] = %d\n", blockIdx.x * BIN_COUNT + value, sum);

        // #ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
        //     atomicAdd(d_Result + value, sum);
        // #else
        d_Result[blockIdx.x * BIN_COUNT + value] = sum;
        // #endif
    }
}

#endif

////////////////////////////////////////////////////////////////////////////////
// Merge blockN histograms into gridDim.x histograms
// blockDim.x == BIN_COUNT
// gridDim.x  == BLOCK_N2
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADS 256

__global__ void mergeHistogram64Kernel(
    unsigned int *d_Histogram,
    unsigned int *d_PartialHistograms,
    unsigned int blockN
){
    __shared__ unsigned int data[MERGE_THREADS];

    unsigned int sum = 0;
    for(unsigned int i = threadIdx.x; i < blockN; i += MERGE_THREADS)
        sum += d_PartialHistograms[blockIdx.x + i * BIN_COUNT];
    data[threadIdx.x] = sum;

    for(unsigned int stride = MERGE_THREADS / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(threadIdx.x < stride)
            data[threadIdx.x] += data[threadIdx.x + stride];
    }

    if(threadIdx.x == 0)
        d_Histogram[blockIdx.x] = data[0];
}



////////////////////////////////////////////////////////////////////////////////
// CPU interface to GPU histogram calculator
////////////////////////////////////////////////////////////////////////////////
//histogram64Kernel() results buffer
unsigned int *d_PartialHistograms;

//Maximum block count for histogram64kernel()
//Limits input data size to 756MB
const int MAX_BLOCK_N = 16384;

void initHistogram64GPU(void){
    #ifdef CUDA_NO_SM_11_ATOMIC_INTRINSICS
        cutilSafeCall( cudaMalloc((void **)&d_PartialHistograms, MAX_BLOCK_N * HISTOGRAM_SIZE) );
    #endif
}

//Internal memory deallocation
void closeHistogram64GPU(void){
    #ifdef CUDA_NO_SM_11_ATOMIC_INTRINSICS
        cutilSafeCall( cudaFree(d_PartialHistograms) );
    #endif
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


//***********************************************************************
//! The CPU version
//***********************************************************************

extern "C" 
void histogram64CPU(
    unsigned int *h_Result,
    unsigned int *h_Data,
    int dataN
){
    int i;
    unsigned int data4;

    for(i = 0; i < BIN_COUNT; i++)
        h_Result[i] = 0;

    for (i = 0; i < dataN; i++){
        data4 = h_Data[i];
        h_Result[(data4 >>  2) & 0x3F]++;
        h_Result[(data4 >> 10) & 0x3F]++;
        h_Result[(data4 >> 18) & 0x3F]++;
        h_Result[(data4 >> 26) & 0x3F]++;
    }
}

//***********************************************************************
//! The Driver
//***********************************************************************

int main() {
  unsigned int *h_Histogram = (unsigned int *)malloc(sizeof(unsigned int) * BIN_COUNT);
  unsigned int *h_result = (unsigned int *)malloc(sizeof(unsigned int) * BIN_COUNT);
  unsigned char *h_Data = (unsigned char *)malloc(sizeof(unsigned char) * DATA_N);
  klee_make_symbolic(h_Data, sizeof(unsigned char) * DATA_N, "input");

  unsigned int *d_Histogram;
  unsigned char *d_Data;
  cudaMalloc((void**)&d_Data, sizeof(unsigned char) * DATA_N);
  cudaMalloc((void**)&d_Histogram, sizeof(unsigned int) * BIN_COUNT);
  cudaMemcpy(d_Data, h_Data, sizeof(unsigned char) * DATA_N, cudaMemcpyHostToDevice);

  // const int histogramCount = iDivUp(DATA_N / 4, THREAD_N * 63);
  histogram64Kernel<<<BIN_COUNT/4, THREAD_N>>>(d_Histogram, (unsigned int *)d_Data, DATA_N / 4);
  cudaMemcpy(h_Histogram, d_Histogram, sizeof(unsigned int) * BIN_COUNT, cudaMemcpyDeviceToHost);

#ifndef _SYM  
  // post-condition
  printf("Now check the post-condition. \n");
  histogram64CPU(h_result, (unsigned int *)d_Data, DATA_N / 4);
  
  printf("After cpu version\n");
  for (int i = 0; i < BIN_COUNT; i++) {
    if (h_result[i] != h_Histogram[i]) {
      printf("Incorrect when i = %d: %d != %d\n",
             i, h_result[i], h_Histogram[i]);
    }
  }
#endif

  cudaFree(d_Data);
  cudaFree(d_Histogram);
  free(h_Data);
  free(h_Histogram);
  free(h_result);
}

