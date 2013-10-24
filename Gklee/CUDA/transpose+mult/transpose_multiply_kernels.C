#include <stdlib.h>
#include <stdio.h>
#include <klee.h>
#include "cutil.h"


#define GRIDSIZE_X 1
#define GRIDSIZE_Y 1
#define BLOCKSIZE  8 //blocks are BLOCKSIZE x BLOCKSIZE
#define DIM_X GRIDSIZE_X * BLOCKSIZE
#define DIM_Y GRIDSIZE_Y * BLOCKSIZE
#define CN DIM_X * DIM_Y 
#define P 16  //must be the case that BLOCKSIZE | P
#define AN P*DIM_X
#define BN DIM_Y*P
#define VAL_CEILING 5

__shared__ int As[BLOCKSIZE][BLOCKSIZE];
__shared__ int Bs[BLOCKSIZE][BLOCKSIZE];

__global__ void
matrixMul( int* A, int* B, int *C, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCKSIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCKSIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCKSIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCKSIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    unsigned Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    printf("bx: %d, by: %d, tx: %d, ty: %d\n", bx, by, tx, ty);
    printf("aBegin: %d, aEnd: %d, aStep: %d\n", aBegin, aEnd, aStep);
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCKSIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCKSIZE * by + BLOCKSIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void
MatTrans(int *gi_data, int *go_data){
  unsigned int x, y, i, j;
  //get cartesian coords in grid
  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  i = x + y * blockDim.x * gridDim.x;
  j = y + x * blockDim.y * gridDim.y;
  
  go_data[j] = gi_data[i];
}

void
printMatrix(int *matrix, int xDim, int yDim){
  for(int y = 0; y < yDim; ++y){
    for(int x = 0; x < xDim; ++x){
      printf("%u\t",matrix[x + y*xDim]);
    }
    printf("\n");
  }
}

bool
matricesEquiv(int *matrix_a, int *matrix_b, int n){
  for(int i = 0; i < n; ++i){
    if(matrix_a[i] != matrix_b[i]){
      return false;
    }
  }
  return true;
}

int 
main(int argc, char* argv[]){
  // const unsigned int seed = 99;
  // Make the input 'hA' as symbolic...
  int *hA = (int *)malloc(sizeof(int) * AN);
  klee_make_symbolic(hA, sizeof(int) * AN, "A_var");
  // Make the input 'hB' as symbolic...
  int *hB = (int *)malloc(sizeof(int) * BN);
  klee_make_symbolic(hB, sizeof(int) * BN, "B_var");
  int *hC = (int *)malloc(sizeof(int) * CN);
  int *hC_P = (int *)malloc(sizeof(int) * CN);

  // A^T ...
  int *A, *AT; // A: [16 * 8] 
  cudaMalloc((void **)&A, sizeof(int) * AN);
  cudaMalloc((void **)&AT, sizeof(int) * AN);
  cudaMemcpy(A, hA, sizeof(int) * AN, cudaMemcpyHostToDevice);
  __modify_Grid(GRIDSIZE_X, P/BLOCKSIZE);// (1, 2)
  __modify_Block(BLOCKSIZE, BLOCKSIZE);// (8, 8)
  __begin_GPU();
  MatTrans<<<>>>(A, AT);
  __end_GPU();
  printf("After A's transpose!\n");

  // B^T ...
  int *B, *BT; // B: [8 * 16]
  cudaMalloc((void **)&B, sizeof(int) * BN);
  cudaMalloc((void **)&BT, sizeof(int) * BN);
  cudaMemcpy(B, hB, sizeof(int) * BN, cudaMemcpyHostToDevice);
  __modify_Grid(P/BLOCKSIZE, GRIDSIZE_Y); // (2, 1)
  __modify_Block(BLOCKSIZE, BLOCKSIZE); // (8, 8)
  __begin_GPU();
  MatTrans(B, BT);
  __end_GPU();
  printf("After B's transpose!\n");

  // A^T * B^T = C...
  int *C;
  cudaMalloc((void **)&C, sizeof(int) * CN);
  __modify_Grid(GRIDSIZE_Y, GRIDSIZE_X); // (1, 1)
  __modify_Block(BLOCKSIZE, BLOCKSIZE); // (8, 8)
  __begin_GPU();
  matrixMul(AT, BT, C, P, DIM_Y);
  __end_GPU();
  printf("After AT and BT multiplication !\n");


  // B * A = T 
  int *T;
  cudaMalloc((void **)&T, sizeof(int) * CN);
  __modify_Grid(GRIDSIZE_Y, GRIDSIZE_X);
  __modify_Block(BLOCKSIZE, BLOCKSIZE);
  __begin_GPU();
  matrixMul(B, A, T, P, DIM_X);
  __end_GPU();

  // T^T = C'
  int *C_P;
  cudaMalloc((void **)&C_P, sizeof(int) * CN);
  __modify_Grid(GRIDSIZE_X, GRIDSIZE_Y);
  __modify_Block(BLOCKSIZE, BLOCKSIZE);
  __begin_GPU();
  MatTrans(T, C_P);
  __end_GPU();
   
  // copy back 
  cudaMemcpy(hC, C, sizeof(int) * CN, cudaMemcpyDeviceToHost);
  cudaMemcpy(hC_P, C_P, sizeof(int) * CN, cudaMemcpyDeviceToHost);

  if (!matricesEquiv(hC, hC_P, CN)) {
    printf("post condition fails!\n");
  } else {
    printf("post condition succeeds!\n");
  }

  cudaFree(C_P);   
  cudaFree(T);
  cudaFree(C);
  cudaFree(B);
  cudaFree(BT);
  cudaFree(A);
  cudaFree(AT);
  free(hA);
  free(hB);
  free(hC);
  free(hC_P);
}
