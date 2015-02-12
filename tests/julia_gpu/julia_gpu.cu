#define DIM 30

__device__ int julia( int x, int y ) {
  if(x > y*y)
    return 1;
  return 0;
}

__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // Break symmetry...
    for(int i = 0; i < x/8; i++)
      ptr[offset*4 + 3] = 255;

    // now calculate the value at that position
    int juliaValue = julia( x, y );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    unsigned char    *dev_bitmap;

    cudaMalloc( (void**)&dev_bitmap, 4*DIM*DIM*sizeof(char) );
    data.dev_bitmap = dev_bitmap;

    dim3    grid(DIM,DIM);
    kernel<<<grid,1>>>( dev_bitmap );
}

