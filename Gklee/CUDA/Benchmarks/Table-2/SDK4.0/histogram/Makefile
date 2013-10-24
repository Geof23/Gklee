1=64
2=256
PROG=histogram$(2)_kernel
SIZE=$(2)

#argument in PROG and SIZE should be same (both 1 or both 2)

SRC=$(PROG).cu

P= #--symbolic-config # Parametric

R=--check-level=2
WD=--check-WD=1
BC=--check-BC=1
MC=--check-MC=1

FLAGS= $(R) $(WD) $(BC) $(MC) $(P)

NVCC_FLAGS= #-D_SYM

KLEE_LPP=-DHIST$(SIZE)

all: compile run

compile:
	gklee-nvcc $(NVCC_FLAGS) $(SRC)
	klee-l++ histogram_main.cpp $(KLEE_LPP)
	llvm-link -o $(PROG).bin histogram_main.o $(PROG)
run:
	gklee $(FLAGS) $(PROG).bin
clean:
	rm -fR klee* $(PROG)
	rm -f $(PROG).bin histogram_main.o histogram_main 