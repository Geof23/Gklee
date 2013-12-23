
/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_KERNEL_SMALL_H_
#define _BISECT_KERNEL_SMALL_H_

// includes, project
#include "structs.h"
#include <stdio.h>

// additional kernel
#include "bisect_util.cpp"

////////////////////////////////////////////////////////////////////////////////
//! Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
//! @param  g_d  diagonal elements in global memory
//! @param  g_s  superdiagonal elements in global elements (stored so that the 
//!              element *(g_s - 1) can be accessed an equals 0
//! @param  n   size of matrix
//! @param  lg  lower bound of input interval (e.g. Gerschgorin interval)
//! @param  ug  upper bound of input interval (e.g. Gerschgorin interval)
//! @param  lg_eig_count  number of eigenvalues that are smaller than \a lg
//! @param  lu_eig_count  number of eigenvalues that are smaller than \a lu
//! @param  epsilon  desired accuracy of eigenvalues to compute
////////////////////////////////////////////////////////////////////////////////
__global__
void
bisectKernel( float* g_d, float* g_s, const unsigned int n,
              float* g_left, float* g_right, 
              unsigned int* g_left_count, unsigned int* g_right_count,
              const float lg, const float ug,
              const unsigned int lg_eig_count, const unsigned int ug_eig_count,
              float epsilon
             ) 
{
  // intervals (store left and right because the subdivision tree is in general 
  // not dense
  __shared__  float  s_left[MAX_THREADS_BLOCK_SMALL_MATRIX];
  __shared__  float  s_right[MAX_THREADS_BLOCK_SMALL_MATRIX];

  // number of eigenvalues that are smaller than s_left / s_right 
  // (correspondence is realized via indices)
  __shared__  unsigned int  s_left_count[MAX_THREADS_BLOCK_SMALL_MATRIX];
  __shared__  unsigned int  s_right_count[MAX_THREADS_BLOCK_SMALL_MATRIX];
  
  // helper for stream compaction
  __shared__  unsigned int  
    s_compaction_list[MAX_THREADS_BLOCK_SMALL_MATRIX + 1];

  // state variables for whole block
  // if 0 then compaction of second chunk of child intervals is not necessary
  // (because all intervals had exactly one non-dead child)
  __shared__  unsigned int compact_second_chunk;
  __shared__  unsigned int all_threads_converged;

  // number of currently active threads
  __shared__  unsigned int num_threads_active;

  // number of threads to use for stream compaction
  __shared__  unsigned int num_threads_compaction;

  // helper for exclusive scan
  unsigned int* s_compaction_list_exc = s_compaction_list + 1;


  // variables for currently processed interval
  // left and right limit of active interval
  float  left = 0.0f;
  float  right = 0.0f;
  unsigned int left_count = 0;
  unsigned int right_count = 0;
  // midpoint of active interval
  float  mid = 0.0f;
  // number of eigenvalues smaller then mid
  unsigned int mid_count = 0;
  // affected from compaction
  unsigned int  is_active_second = 0;

  s_compaction_list[threadIdx.x] = 0;
  s_left[threadIdx.x] = 0;
  s_right[threadIdx.x] = 0;
  s_left_count[threadIdx.x] = 0;
  s_right_count[threadIdx.x] = 0;

  __syncthreads();

  // set up initial configuration
  if( 0 == threadIdx.x) {
    s_left[0] = lg;
    s_right[0] = ug;
    s_left_count[0] = lg_eig_count;
    s_right_count[0] = ug_eig_count;

    compact_second_chunk = 0;
    num_threads_active = 1;
    num_threads_compaction = 1;
  }

  printf("here 1\n");

  // for all active threads read intervals from the last level 
  // the number of (worst case) active threads per level l is 2^l
  while( true) {

    all_threads_converged = 1;
    __syncthreads();

    is_active_second = 0;
    subdivideActiveInterval( threadIdx.x, 
                             s_left, s_right, s_left_count, s_right_count,
                             num_threads_active,
                             left, right, left_count, right_count,
                             mid, all_threads_converged);

    __syncthreads();
    printf("here 2, all_threads_converged: %u\n", all_threads_converged);


    // check if done
    if( 1 == all_threads_converged) {
      printf("break here\n");
      break;
    }

    printf("all_threads_converged: %d\n", all_threads_converged);
    __syncthreads();

    // compute number of eigenvalues smaller than mid
    // use all threads for reading the necessary matrix data from global 
    // memory
    // use s_left and s_right as scratch space for diagonal and
    // superdiagonal of matrix
    mid_count = computeNumSmallerEigenvals( g_d, g_s, n, mid, 
                                            threadIdx.x, num_threads_active,
                                            s_left, s_right,
                                            (left == right) );

#if __DEVICE_EMULATION__
    if(    ((mid_count < left_count) || (mid_count > right_count)) 
        && (left != right) ) {
      printf( "%f / %f / %f :: %i / %i / %i\n",
              left, mid, right, left_count, mid_count, right_count);
      cutilCondition( 0);
    }
#endif

    __syncthreads();

    printf("here 3\n");

    // store intervals 
    // for all threads store the first child interval in a continuous chunk of
    // memory, and the second child interval -- if it exists -- in a second
    // chunk; it is likely that all threads reach convergence up to 
    // \a epsilon at the same level; furthermore, for higher level most / all
    // threads will have only one child, storing the first child compactly will
    // (first) avoid to perform a compaction step on the first chunk, (second)
    // make it for higher levels (when all threads / intervals have 
    // exactly one child)  unnecessary to perform a compaction of the second 
    // chunk
    if( threadIdx.x < num_threads_active) {

      if(left != right) {

        // store intervals
        storeNonEmptyIntervals( threadIdx.x, num_threads_active,
                                s_left, s_right, s_left_count, s_right_count,
                                left, mid, right, 
                                left_count, mid_count, right_count,
                                epsilon, compact_second_chunk,
                                s_compaction_list_exc,
                                is_active_second ); 
      }
      else {

        storeIntervalConverged( s_left, s_right, s_left_count, s_right_count,
                                left, mid, right, 
                                left_count, mid_count, right_count, 
                                s_compaction_list_exc, compact_second_chunk,
                                num_threads_active,
                                is_active_second );
      }
    }

    // necessary so that compact_second_chunk is up-to-date
    __syncthreads();

    printf("here 4\n");

    // perform compaction of chunk where second children are stored
    // scan of (num_threads_active / 2) elements, thus at most
    // (num_threads_active / 4) threads are needed
    if( compact_second_chunk > 0) {

      createIndicesCompaction( s_compaction_list_exc, num_threads_compaction);

      compactIntervals( s_left, s_right, s_left_count, s_right_count,
                        mid, right, mid_count, right_count,
                        s_compaction_list, num_threads_active,
                        is_active_second );
    } 

    __syncthreads();

    if( 0 == threadIdx.x) {

      // update number of active threads with result of reduction
      num_threads_active += s_compaction_list[num_threads_active];

      num_threads_compaction = ceilPow2( num_threads_active);

      compact_second_chunk = 0;
    }

    __syncthreads();

  }

  __syncthreads();

  printf("here 5\n");

  // write resulting intervals to global mem
  // for all threads write if they have been converged to an eigenvalue to 
  // a separate array
 
  // at most n valid intervals
  if( threadIdx.x < n) {

    // intervals converged so left and right limit are identical
    g_left[threadIdx.x]  = s_left[threadIdx.x]; 
    // left count is sufficient to have global order
    g_left_count[threadIdx.x]  = s_left_count[threadIdx.x]; 
  }

}

//***********************************************************************
//! Initialization by the GPU
//***********************************************************************

////////////////////////////////////////////////////////////////////////////////
//! Initialize the input data to the algorithm
//! @param input  handles to the input data
//! @param exec_path  path where executable is run (argv[0])
//! @param mat_size  size of the matrix
//! @param user_defined  1 if the matrix size has been requested by the user, 
//!                      0 if the default size
////////////////////////////////////////////////////////////////////////////////

void
initInputData( InputData& input, const unsigned int mat_size) 
{   
  // allocate memory
  input.a = (float*) malloc( sizeof(float) * mat_size);
  input.b = (float*) malloc( sizeof(float) * mat_size);
  
  // // initialize diagonal and superdiagonal entries with random values
#ifndef _SYM
  srand( 278217421);
  srand( clock());
  for( unsigned int i = 0; i < mat_size; ++i) {
    input.a[i] = (float) (2.0 * (((double)rand() 
  				    / (double) RAND_MAX) - 0.5));
    input.b[i] = (float) (2.0 * (((double)rand() 
   				    / (double) RAND_MAX) - 0.5));
  } 
#else
  klee_make_symbolic(input.a, sizeof(float) * mat_size, "input_a");
  klee_make_symbolic(input.b, sizeof(float) * mat_size, "input_b");
#endif
  
  // the first element of s is used as padding on the device (thus the 
  // whole vector is copied to the device but the kernels are launched
  // with (s+1) as start address
  input.b[0] = 0.0f;
    
  // // allocate device memory for input
  cutilSafeCall( cudaMalloc( (void**) &(input.g_a)    , sizeof(float) * mat_size));
  cutilSafeCall( cudaMalloc( (void**) &(input.g_b_raw), sizeof(float) * mat_size));
  
  // copy data to device
  cutilSafeCall( cudaMemcpy( input.g_a    , input.a, sizeof(float) * mat_size, cudaMemcpyHostToDevice ));
  cutilSafeCall( cudaMemcpy( input.g_b_raw, input.b, sizeof(float) * mat_size, cudaMemcpyHostToDevice ));
  
  input.g_b = input.g_b_raw + 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////
void 
initResultSmallMatrix( ResultDataSmall& result, const unsigned int mat_size) {

  result.mat_size_f = sizeof(float) * mat_size;
  result.mat_size_ui = sizeof(unsigned int) * mat_size;

  result.eigenvalues = (float*) malloc( result.mat_size_f);

  // helper variables
  result.zero_f = (float*) malloc( result.mat_size_f);
  result.zero_ui = (unsigned int*) malloc( result.mat_size_ui);
  for( unsigned int i = 0; i < mat_size; ++i) {

    result.zero_f[i] = 0.0f;
    result.zero_ui[i] = 0;
    
    result.eigenvalues[i] = 0.0f;
  }

  cutilSafeCall( cudaMalloc( (void**) &result.g_left, result.mat_size_f));
  cutilSafeCall( cudaMalloc( (void**) &result.g_right, result.mat_size_f));
  
  cutilSafeCall( cudaMalloc( (void**) &result.g_left_count, 
			     result.mat_size_ui));
  cutilSafeCall( cudaMalloc( (void**) &result.g_right_count, 
                            result.mat_size_ui));
  
  // initialize result memory
  cutilSafeCall( cudaMemcpy( result.g_left, result.zero_f, result.mat_size_f,
			     cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMemcpy( result.g_right, result.zero_f, result.mat_size_f,
			     cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMemcpy( result.g_right_count, result.zero_ui, 
			     result.mat_size_ui,
			     cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMemcpy( result.g_left_count, result.zero_ui, 
			     result.mat_size_ui,
			     cudaMemcpyHostToDevice));
}


//***********************************************************************
//! The Driver
//***********************************************************************

// // the original driver
//     dim3  blocks( 1, 1, 1);
//     dim3  threads( MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

//     bisectKernel<<< blocks, threads >>>( input.g_a, input.g_b, mat_size,
//                                          result.g_left, result.g_right, 
//                                          result.g_left_count, 
//                                          result.g_right_count,
//                                          lg, ug, 0, mat_size, 
//                                          precision 
//                                        );

#define FLT_MAX 1

int main() {

  unsigned int mat_size = 128;

  InputData input;
  initInputData(input, mat_size);

  // desired precision of eigenvalues
  float precision = 0.00001f;

  float lg = FLT_MAX;
  float ug = -FLT_MAX;
  // computeGerschgorin( input.a, input.b+1, mat_size, lg, ug);

  ResultDataSmall result;
  initResultSmallMatrix( result, mat_size);

  dim3  blocks( 1, 1, 1);
  dim3  threads( MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

  bisectKernel<<< blocks, threads >>>( input.g_a, input.g_b, mat_size,
                                       result.g_left, result.g_right,
                                       result.g_left_count,
                                       result.g_right_count,
                                       lg, ug, 0, mat_size,
                                       precision);
}


#endif // #ifndef _BISECT_KERNEL_SMALL_H_
