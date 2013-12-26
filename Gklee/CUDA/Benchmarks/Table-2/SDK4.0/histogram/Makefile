PROG64=histogram64_kernel
PROG256=histogram256_kernel

SRC64=$(PROG64).cu
SRC256=$(PROG256).cu

NVCC_FLAGS=-D_SYM
P_FLAGS=--symbolic-config # Parametric

all: compile run para-run

compile:
	gklee-nvcc $(SRC64) -o $(PROG64).o
	gklee-nvcc $(SRC256) -o $(PROG256).o
	klee-l++ histogram_main.cpp -D_SYM -o histogram_main.o 
	llvm-link -o histogram histogram_main.o $(PROG64).o $(PROG256).o
run:
	gklee histogram 
para-run:
	gklee $(P_FLAGS)$ histogram 
clean:
	rm -rf klee* $(PROG64).o $(PROG256).o kernelSet.txt
	rm -f histogram histogram_main.o $(PROG64).cpp $(PROG256).cpp 
