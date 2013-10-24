#include <stdio.h>
#include "reduction_kernel.cu"

//***********************************************************************
//! The Driver
// The number of threads must be not less than wA * wB
//***********************************************************************

int main() {
  int input[NUM * 2];
  int output[1];
  klee_make_symbolic(input, sizeof(input), "input");

  int *dinput, *doutput;
  cudaMalloc((void**)&dinput, sizeof(int)*NUM*2);
  cudaMalloc((void**)&doutput, sizeof(int));
  cudaMemcpy(dinput, input, sizeof(int)*NUM*2, cudaMemcpyHostToDevice);

#if defined RED0
  FUNC(reduce0)<<<1, NUM>>>(dinput, doutput, NUM);
#elif defined RED1 
  FUNC(reduce1)<<<1, NUM>>>(dinput, doutput, NUM);
#elif defined RED2
  FUNC(reduce2)<<<1, NUM>>>(dinput, doutput, NUM);
#elif defined RED3
  FUNC(reduce3)<<<1, NUM>>>(dinput, doutput, NUM);
#elif defined RED4
  FUNC(reduce4)<<<1, NUM>>>(dinput, doutput, NUM, NUM);
#elif defined RED5
  FUNC(reduce5)<<<1, NUM>>>(dinput, doutput, NUM, NUM);
#else
  FUNC(reduce6)<<<1, NUM>>>(dinput, doutput, NUM, NUM, false);
#endif

  cudaMemcpy(output, doutput, sizeof(int), cudaMemcpyDeviceToHost);

#ifndef _SYM
  // post-condition
  int sum = 0;
  for (int i = 0; i < NUM; i++) 
    sum += input[i];
  if (sum != output[0]) {
    printf("Incorrect: sum = %d, output[0] = %d \n", sum, output[0]); 
    return 0;
  }
  return 1;
#endif
}
