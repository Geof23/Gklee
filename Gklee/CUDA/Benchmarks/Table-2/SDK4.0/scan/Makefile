PROG=scan_kernel
SRC=$(PROG).cu

NVCC_FLAGS=-D_SYM
P_FLAGS=--symbolic-config # Parametric

all: compile run para-run

compile:
	klee-l++ -o main.o main.cpp -D_SYM
	gklee-nvcc $(NVCC_FLAGS) $(SRC) -o $(PROG).o
	llvm-link -o scan main.o $(PROG).o 
run:
	gklee scan
para-run:
	gklee $(P_FLAGS)$ scan

clean:
	rm -rf klee* scan main.o kernelSet.txt
	rm scan_kernel.o scan_kernel.cpp
