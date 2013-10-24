// RUN: echo "x" > %t1.res
// RUN: echo "x" >> %t1.res
// RUN: echo "x" >> %t1.res
// RUN: echo "x" >> %t1.res
// RUN: %llvmgcc %s -emit-llvm -O0 -c -o %t1.bc
// RUN: %klee %t1.bc > %t1.log
// RUN: diff %t1.res %t1.log

#include <stdio.h>

unsigned klee_urange(unsigned start, unsigned end) {
  unsigned x;
  klee_make_symbolic(&x, sizeof x);
  if (x-start>=end-start) klee_silent_exit(0);
  return x;
}

int *make_int(int i) {
  int *x = malloc(sizeof(*x));
  *x = i;
  return x;
}

int main() {
  int *buf[4];
  int i,s,t;

  for (i=0; i<4; i++)
    buf[i] = make_int((i+1)*2);

  s = klee_urange(0,4);

  int x = *buf[s];

  if (x == 4)
    if (s!=1)
      abort();

  printf("x\n");
  fflush(stdout);

  return 0;
}
