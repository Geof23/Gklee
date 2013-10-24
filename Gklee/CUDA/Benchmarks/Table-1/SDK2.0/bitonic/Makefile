1=small
2=large
PROG=bitonic
SRC=$(PROG).cu

P= #--symbolic-config # Parametric

R=--check-level=2
WD=--check-WD=1
BC=--check-BC=1
MC=--check-MC=1

FLAGS= $(R) $(WD) $(BC) $(MC) $(P)

NVCC_FLAGS=-D_SYM

all: compile run

compile:
	gklee-nvcc $(NVCC_FLAGS) $(SRC)
run:
	gklee $(FLAGS) $(PROG)
clean:
	rm -fR klee* $(PROG)
	rm -f $(PROG).cpp