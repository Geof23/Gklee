#include <stdio.h>

__global__ void dumbkernel(){
   printf("we made it to dumbkernel\n");
}


int main(){
    dumbkernel<<<1, 1>>>();
}