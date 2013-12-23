PROG=bisect_kernel_small
PROGL=bisect_kernel_large
SRC=$(PROG).cu
SRCL=$(PROGL).cu

NVCC_FLAGS=-D_SYM
P_FLAGS=--symbolic-config # Parametric

all: compile run para-run

compile:
	gklee-nvcc $(NVCC_FLAGS) $(SRC)
	gklee-nvcc $(NVCC_FLAGS) $(SRCL)
run:
	gklee $(PROG)
	gklee $(PROGL)
para-run:
	gklee $(P_FLAGS)$ $(PROG) 
	gklee $(P_FLAGS)$ $(PROGL) 
clean:
	rm -rf klee* $(PROG) $(PROGL) kernelSet.txt
	rm -f $(PROG).cpp $(PROGL).cpp
