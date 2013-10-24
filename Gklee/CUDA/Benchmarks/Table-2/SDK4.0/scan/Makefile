P= #--symbolic-config # Parametric

R=--check-level=2
WD=--check-WD=1
BC=--check-BC=1
MC=--check-MC=1

FLAGS= $(R) $(WD) $(BC) $(MC) $(P)

NVCC_FLAGS= #-D_SYM

all:
	$(info use make small, runsmall, large, runlarge)

small:
	gklee-nvcc scan_kernel.cu 
	klee-l++ -c main.cpp -DSCANSHORT
	llvm-link -o scan_short main.o scan_kernel 
runsmall: small
	gklee scan_short
large:
	gklee-nvcc scan_kernel.cu
	klee-l++ -c main.cpp -DSCANLARGE
	llvm-link -o scan_large main.o scan_kernel
runlarge: large
	gklee scan_large
clean:
	rm -fR klee* scan_short *.o
	rm scan_kernel