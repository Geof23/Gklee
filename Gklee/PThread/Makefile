
all: t1.c
	cc t1.c -o t1 -lpthread

sym: t1.c
	klee-lcc t1.c
