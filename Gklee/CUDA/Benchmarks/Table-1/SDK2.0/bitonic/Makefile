1=small
2=large
PROG=bitonic
SRC=$(PROG).cu

NVCC_FLAGS=-D_SYM
P_FLAGS=--symbolic-config # Parametric

all: compile run para-run

compile:
	gklee-nvcc $(NVCC_FLAGS) $(SRC)
run:
	gklee $(PROG)
para-run:
	gklee $(P_FLAGS)$ $(PROG) 
clean:
	rm -rf klee* $(PROG) kernelSet.txt
	rm -f $(PROG).cpp
