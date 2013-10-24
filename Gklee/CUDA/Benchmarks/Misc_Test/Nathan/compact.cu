#include <cassert>
#include <cstdio>

#define N 4

#define PREDICATE(x) (((x & 1) == 0) ? 1 : 0)

__global__ void compact(int *out, int*in) {
  __shared__ unsigned flag[N];
  __shared__ unsigned idx[N];

  unsigned t = threadIdx.x;

  // (i) test each element with predicate p
  // flag = 1 if keeping element
  //        0 otherwise
  printf("before predicate\n");
  flag[t] = PREDICATE(in[t]);

  // (ii) compute indexes for scatter
  //      using an exclusive prefix sum
  __syncthreads();
  if (t < N/2) {
    idx[2*t]   = flag[2*t];
    idx[2*t+1] = flag[2*t+1];
  }
  // (a) upsweep
  int offset = 1;
  for (unsigned d = N/2; d > 0; d /= 2) {
    __syncthreads();
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      idx[bi] += idx[ai];
    }
    offset *= 2;
  }
  // (b) downsweep
  if (t == 0) idx[N-1] = 0;
  for (unsigned d = 1; d < N; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      int temp = idx[ai];
      idx[ai] = idx[bi];
      idx[bi] += temp;
    }
  }
  __syncthreads();
  // end of exclusive prefix sum of flag into idx

  // (iii) scatter
  if (flag[t]) out[idx[t]] = in[t];
}

int main(int argc, char **argv) {
  // test data
  size_t ArraySize = N * sizeof(int);
  int *in  = (int *)malloc(ArraySize);
  int *out = (int *)malloc(ArraySize);
  klee_make_symbolic(in, ArraySize, "in");

  // create some memory objects on the device
  int *d_in;
  int *d_out;
  cudaMalloc((void **)&d_in, ArraySize);
  cudaMalloc((void **)&d_out, ArraySize);

  // memcpy into these objects
  cudaMemcpy(d_in, in, ArraySize, cudaMemcpyHostToDevice);

  // run the kernel
  compact<<<1,N>>>(d_out, d_in);

  printf("finish GPU mode\n");
  // memcpy back the result
  cudaMemcpy(out, d_out, ArraySize, cudaMemcpyDeviceToHost);

#ifndef _SYM
  // check results
  unsigned idx = 0;
  for (unsigned i=0; i<N; ++i) {
    if (PREDICATE(in[i])) {
      assert(out[idx] == in[i]);
      idx++;
    }
  }
  printf("TEST PASSED\n");
#endif

  // cleanup
  free(in);
  free(out);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
