#include <stdio.h>

__global__ void dumbkernel(bool *input){
   // if( input[threadIdx.x] ){
   //    printf("we made it to dumbkernel\n");
   // }
}

#define SZ 25

int main(){
    bool *devDummy;
    cudaMalloc( (void**) &devDummy, sizeof(bool) * SZ);
    dumbkernel<<<1, 32>>>(devDummy);
}