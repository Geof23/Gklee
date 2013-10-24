
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define GRIDSIZE_X 1
#define GRIDSIZE_Y 1

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8 
#endif

#define DIM_X GRIDSIZE_X * BLOCK_SIZE
#define DIM_Y GRIDSIZE_Y * BLOCK_SIZE
#define CN DIM_X * DIM_Y
#define P 32  //must be the case that BLOCKSIZE | P
#define AN DIM_Y * P
#define BN P * DIM_X

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((unsigned*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((unsigned*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////

// Declaration of the shared memory array As used to
// store the sub-matrix of A
__shared__ unsigned As[BLOCK_SIZE][BLOCK_SIZE];

// Declaration of the shared memory array Bs used to
// store the sub-matrix of B
__shared__ unsigned Bs[BLOCK_SIZE][BLOCK_SIZE];

__global__ void
matrixMul( unsigned* C, unsigned* A, unsigned* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    unsigned Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += AS(ty, k) * BS(k, tx);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


// CPU program for testing
////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
computeGold(unsigned* C, const unsigned* A, const unsigned* B, 
	    unsigned int hA, unsigned int wA, unsigned int wB) {
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      int sum = 0;
      for (unsigned int k = 0; k < wA; ++k) {
	unsigned a = A[i * wA + k];
	unsigned b = B[k * wB + j];
	sum += a * b;
      }
      C[i * wB + j] = sum;
    }
}


//***********************************************************************
//! The Driver
// The number of threads must be not less than wA * wB
//***********************************************************************

int main() {
  unsigned hA[AN] = {0,1,2,3};
  unsigned hB[BN] = {0,1,2,3};
  klee_make_symbolic(hA, sizeof(hA), "hA");
  klee_make_symbolic(hB, sizeof(hB), "hB");

  unsigned hC[CN];

  int wA = P;
  int wB = DIM_X;

  unsigned *dA, *dB, *dC;
  cudaMalloc((void**)&dA, sizeof(unsigned)*AN);
  cudaMalloc((void**)&dB, sizeof(unsigned)*BN);
  cudaMalloc((void**)&dC, sizeof(unsigned)*CN);
  cudaMemcpy(dA, hA, sizeof(unsigned)*AN, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(unsigned)*BN, cudaMemcpyHostToDevice);

  // now execute the kernel
  dim3 grid(GRIDSIZE_X, GRIDSIZE_Y);
  dim3 block(BLOCK_SIZE/2, BLOCK_SIZE);
  matrixMul<<<grid, block>>>(dC, dA, dB, wA, wB);
  cudaMemcpy(hC, dC, sizeof(unsigned)*CN, cudaMemcpyDeviceToHost);

  unsigned hC1[CN];
  int hhA = DIM_Y;

#ifndef _SYM
  // post-condition
  computeGold(hC1, hA, hB, hhA, wA, wB);

  for (int i = 0; i < hhA; i++) 
    for (int j = 0; j < wB; j++) {
      int k = i * wB + j;
      // printf("C[%d] = %u ", k, C[k]);
      if (hC1[k] != hC[k]) {
	printf("Incorrect: hC1[k] = %u, hC[k] = %u \n", hC1[k], hC[k]); 
	return 0;
      };
    }
#endif
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
