
#ifndef __FLA_KLEE__
#define __FLA_KLEE__

#include <klee.h>

#ifdef __cplusplus
extern "C" {
#endif

void __attribute__((noinline))
klee_make_symbolic_str(char *addr, unsigned nbytes, const char *name) {
  klee_make_symbolic(addr, nbytes, name);
  int i;
  for (i = 0; i < nbytes - 1; i++) 
    klee_assume(addr[i] != 0);     // no '\0' in the middle
  klee_assume(addr[i] == 0);        // end with '\0'
}

#ifdef __cplusplus
}
#endif

#endif
