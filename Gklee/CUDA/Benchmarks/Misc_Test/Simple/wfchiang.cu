#include <stdio.h>

__device__ void testKern(){
  __shared__ int tvs[4];
  __shared__ int tv;
  tvs[tv] = threadIdx.x;
  if (threadIdx.x == 0) {
    tv = tv + 1;
  }
}

int main(){
  testKern<<<1, 4>>>();
}

