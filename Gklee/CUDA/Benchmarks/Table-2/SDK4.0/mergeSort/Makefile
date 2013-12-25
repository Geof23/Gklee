KERNEL=mergeSort
SRCK=$(KERNEL).cu
PROG=mergeSort_main
SRC=$(PROG).cu

NVCC_FLAGS=-D_SYM
P_FLAGS=--symbolic-config # Parametric

all: compile run para-run

compile:
	gklee-nvcc $(NVCC_FLAGS) $(SRC) -o $(PROG).o
	gklee-nvcc $(NVCC_FLAGS) $(SRCK) -o $(KERNEL)_kernel.o
	llvm-link -o $(KERNEL) $(PROG).o $(KERNEL)_kernel.o
run:
	gklee $(KERNEL)
para-run:
	gklee $(P_FLAGS)$ $(PROG) 
clean:
	rm -rf klee* $(KERNEL) $(PROG).o $(KERNEL)_kernel.o kernelSet.txt
	rm -f $(KERNEL).cpp $(PROG).cpp
