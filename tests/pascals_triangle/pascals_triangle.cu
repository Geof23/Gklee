#include <stdio.h>
#define THREADS 1024
#define ELEMENTS 1024<<4

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

void print_array(unsigned int* host_array, int num_elements) {
  // print any information
  for (int i=0; i<num_elements; i++) {
    if (host_array[i] != 0) {
      printf("%u, ", host_array[i]);
    }
  }
  printf("\n");
}

__global__ void device_global(unsigned int *input_array, unsigned int *output_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (my_index != 0) {
    int my_val = input_array[my_index];
    int prev_val = input_array[my_index-1];
  }

  __syncthreads();

  if (my_index != 0) {
    output_array[my_index] = my_val+prev_val;
  }
}


int main(void) {
  // how big our array for interfacing with the GPU will be
  int num_elements = ELEMENTS;
  int num_bytes = sizeof(unsigned int) * num_elements;
    
  // pointers for the interfacing arrays
  unsigned int *host_array = 0;
  unsigned int *device_array_a = 0;
  unsigned int *device_array_b = 0;
 
  // malloc for host and device
  host_array = (unsigned int*) malloc(num_bytes);
  cudaMalloc((void **) &device_array_a, num_bytes);
  cudaMalloc((void **) &device_array_b, num_bytes);

  // check the mallocs
  if (host_array == 0) {
    printf("Unable to allocate memory on host");
    return 1;
  }

  if (device_array_a == 0 || device_array_b == 0) {
    printf("Unable to allocate memory on device");
    return 1;
  }

  // set host array values
  for (int i = 0; i<num_elements; i++) {
    host_array[i] = 0;
  }
  host_array[1] = 1;

  // copy them to the GPU
  cudaMemcpy(device_array_a, host_array, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_array_b, host_array, num_bytes, cudaMemcpyHostToDevice);

  // define block and grid sizes
  int block_size = THREADS;
  int grid_size = (num_elements + block_size - 1) / block_size;

  // run GPU code
  print_array(host_array, num_elements);

  for (int j=(num_elements-2)/2; j>5; j -= 2) {
    device_global<<<grid_size, block_size>>>(device_array_a, device_array_b, num_elements);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(host_array, device_array_b, num_bytes, cudaMemcpyDeviceToHost);
    print_array(host_array, num_elements);

    device_global<<<grid_size, block_size>>>(device_array_b, device_array_a, num_elements);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(host_array, device_array_a, num_bytes, cudaMemcpyDeviceToHost);
    print_array(host_array, num_elements);
  }


  // free memory
  free(host_array);
  cudaFree(device_array_a);
  cudaFree(device_array_b);
}
