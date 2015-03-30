#include <cuda.h>
#include<stdio.h>

#define SIZE 10
#define RAD 12

__global__ void set_sphere(int cx, int cy, int cz, int r, bool* out)
{                                                                   
  int x  =threadIdx.x + blockIdx.x*blockDim.x - r + cx;             
  int y = threadIdx.y + blockIdx.y*blockDim.y - r + cy;             
  int z = threadIdx.z + blockIdx.z*blockDim.z - r + cz;             
                                                                    
  bool inside = false, outside = false;                             
  int idx = threadIdx.z*blockDim.x*blockDim.y                       
    + threadIdx.y*blockDim.x + threadIdx.x;                
	out[ idx ] = 0;
                                                                    
  // Test if block is on surface of sphere                                                    
  for(int dx = 0; dx < 2; ++dx)                                     
    for(int dy = 0; dy < 2; ++dy)                                   
      for(int dz = 0; dz < 2; ++dz)                                 
      {
        int vertex_dist =(x+dx - cx)*(x+dx - cx) +                                
                                     (y+dy - cy)*(y+dy - cy) +                                
                                     (z+dz - cz)*(z+dz - cz);                                                     
				if(vertex_dist <= r*r)                          
          inside = true;                                            
        else                                                        
          outside = true;                                           
      }    
                                                         
  out[idx] = inside && outside;                                     
} 


int
main(){
	bool out[SIZE*SIZE*SIZE];
	bool *dev;
	dim3 blockDim(SIZE, SIZE, SIZE);
	cudaMalloc( (void**) &dev, sizeof(bool) * SIZE*SIZE*SIZE);
	set_sphere<<<1, blockDim>>>(SIZE, SIZE, SIZE, RAD, dev);
	cudaMemcpy(out, dev, sizeof(bool)*SIZE*SIZE*SIZE, cudaMemcpyDeviceToHost);

 int x,y,z;
 printf("showing the results of 'out':\n");
 for(z = 0; z < SIZE; ++z){
 	 printf("\nhere's the surface for z = %d\n", z);
 	 for(y = 0; y < SIZE; ++y){
 		 for(x = 0; x < SIZE; ++x){
 			 printf(" %d ", out[x + y * SIZE + z * SIZE * SIZE] == true? 1 : 0);
 		 }
 		 printf("\n");
 	 }      
 }
}
