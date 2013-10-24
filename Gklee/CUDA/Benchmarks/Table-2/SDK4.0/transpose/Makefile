1=small
2=large
PROG=transpose_main
SRC=$(PROG).cu

P= #--symbolic-config # Parametric

R=--check-level=2
WD=--check-WD=1
BC=--check-BC=1
MC=--check-MC=1

FLAGS= $(R) $(WD) $(BC) $(MC) $(P)

NVCC_FLAGS=
CONF1=-Dtranpose0
CONF2=-Dtranpose1
CONF3=-Dtranpose2
CONF3=-Dtranpose4
CONF3=-Dtranpose5
CONF=$(CONF5)

all: compile run

compile:
	gklee-nvcc $(CONF) $(SRC) $(NVCC_FLAGS)
run:
	gklee $(FLAGS) $(PROG)
clean:
	rm -fR klee* $(PROG)
	rm -f $(PROG).cpp