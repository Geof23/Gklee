#include <stdio.h>

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  // stop out of bounds access
  if (my_index < num_elements) {

    if (my_index%2 == 1) {
      // even threads write their index to their array entry
      input_array[my_index] = my_index;
    } else {
      // odd threads copy their value from the next array entry
      input_array[my_index] = input_array[my_index+1];
    }
  }
}

int main(void) {
  // how big our array for interfacing with the GPU will be
  int num_elements = 100;
  int num_bytes = sizeof(unsigned int) * num_elements;
    
  // pointers for the interfacing arrays
  unsigned int *host_array = 0;
  unsigned int *device_array = 0;
 
  // malloc for host and device
  host_array = (unsigned int*) malloc(num_bytes);
  cudaMalloc((void **) &device_array, num_bytes);

  // check the mallocs
  if (host_array == 0) {
    printf("Unable to allocate memory on host");
    return 1;
  }

  if (device_array == 0) {
    printf("Unable to allocate memory on device");
    return 1;
  }

  // set host array values
  for (int i = 0; i<num_elements; i++) {
    host_array[i] = 1;
  }

  // copy them to the GPU
  cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyHostToDevice);

  // define block and grid sizes
  int block_size = 128;
  int grid_size = (num_elements + block_size - 1) / block_size;

  // run GPU code
  device_global<<<grid_size, block_size>>>(device_array, num_elements);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  // copy output to host
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print any information
  for (int i=0; i<num_elements; i++) {
    printf("%03u, ", host_array[i]);
    if (i%10 == 9) {
      printf(" \n");
    }
  }

  // free memory
  free(host_array);
  cudaFree(device_array);
}