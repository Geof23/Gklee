
#ifndef __FLA_KLEE__
#define __FLA_KLEE__

#include <string>
#include <klee.h>

#ifdef __cplusplus
extern "C" {
#endif


  /**************************************************************
   Functions to access klee's internal facilities.
   Note that the bodies of these functions do not matter because
   they will be reinterpreted by KLEE++ at runtime.
  ***************************************************************/

// unsigned FLA_klee_dummy() {
//   printf("dummy function \n");
// }

// // compute the maximum possible value
// unsigned __attribute__((noinline))
//   FLA_klee_get_max_value(unsigned i) {
//   printf("compute the maximum possible value \n");

//   // trikcy here: need this to avoid LLVM-GCC's over-optimization
//   return FLA_klee_dummy();
// }


// // make an ite expression
// unsigned __attribute__((noinline)) 
//   FLA_klee_make_ite(bool c, unsigned l, unsigned r) {
//   return c ? l : r;
// }

// // print an expression
// void __attribute__((noinline)) FLA_klee_print_expr(unsigned i) {
//   printf("print an integer expression \n");
// }


/***************************************************
 Functions extending KLEE's basic interface 
 **************************************************/

// // nbytes may be a symbolic expression
// void FLA_klee_make_symbolic(string *addr, unsigned nbytes, const char *name) {

//   // compute the maximum possible value of the length
//   unsigned max = FLA_klee_get_max_value(nbytes);
//   printf("max = %d \n", max);

//   // create a char array and make it symbolic
//   char tmp_s[max];
//   klee_make_symbolic(tmp_s, max, name);

//   // copy the char array to the input string
//   addr->clear();
//   for (int i = 0; i < max; i++) { 
//     *addr += tmp_s[i];
//   }

//   // record the symbolic length
//   addr->symlen = nbytes;
// }


  void FLA_klee_make_symbolic(std::string *addr, unsigned nbytes, const char *name) {
  // create a char array and make it symbolic
  char tmp_s[nbytes];
  klee_make_symbolic(tmp_s, nbytes, name);

  // copy the char array to the input string
  addr->clear();
  for (int i = 0; i < nbytes; i++) { 
    *addr += tmp_s[i];
  }
}


// nbytes here is the maximum size; the body won't be executed
  void FLA_klee_make_symbolic_length(std::string *addr, unsigned min, unsigned max, const char *name) {
  FLA_klee_make_symbolic(addr, max, name);
}


// void FLA_klee_make_symbolic(void *addr, unsigned nbytes, const char *name) {
//   klee_make_symbolic(addr, nbytes, name);
// }


#ifdef __cplusplus
}
#endif

#endif
