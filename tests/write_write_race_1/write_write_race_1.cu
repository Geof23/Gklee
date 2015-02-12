#include <stdio.h>

__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  input_array[0] = my_index;
  }


int main(void) {
  // how big our array for interfacing with the GPU will be
  int num_elements = 1;
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
    host_array[i] = 0;
  }

  // copy them to the GPU
  cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyHostToDevice);

  // define block and grid sizes
  int block_size = 32;
  int grid_size = (num_elements + block_size - 1) / block_size;

  // run GPU code
  device_global<<<grid_size, block_size>>>(device_array, num_elements);

  // copy output to host
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print any information
  printf("host_array[0] = %u \n", host_array[0]);

  // free memory
  free(host_array);
  cudaFree(device_array);
}