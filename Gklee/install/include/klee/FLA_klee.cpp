
#include <string>
#include <klee.h>

#ifndef __FLA_KLEE__

#define __FLA_KLEE__

using namespace std;


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


// // nbytes here is assumed to be a concrete value specifying 
// // the maximum length of the string
// void FLA_klee_make_symbolic2(string *addr, unsigned nbytes, const char *name) {

//   // make a symbolic variable for the length
//   string len_name = name;
//   len_name += ".len";
  
//   char len;
//   klee_make_symbolic(&len, sizeof(len), len_name.c_str());
//   klee_assume(len <= nbytes);   // don't miss this one

//   addr->symlen = len;

//   // create a char array and make it symbolic
//   char tmp_s[nbytes];
//   klee_make_symbolic(tmp_s, nbytes, name);

//   // copy the char array to the input string
//   addr->clear();
//   for (int i = 0; i < nbytes; i++) { 
//     *addr += tmp_s[i];
//   }

// }


void FLA_klee_make_symbolic(string *addr, unsigned nbytes, const char *name) {
  // create a char array and make it symbolic
  char tmp_s[nbytes];
  klee_make_symbolic(tmp_s, nbytes, name);

  // copy the char array to the input string
  addr->clear();
  for (int i = 0; i < nbytes; i++) { 
    *addr += tmp_s[i];
  }
}


// nbytes here is the maximum size
void FLA_klee_make_symbolic_length(string *addr, unsigned min, unsigned max, const char *name) {
  FLA_klee_make_symbolic(addr, max, name);
}


// void FLA_klee_make_symbolic(void *addr, unsigned nbytes, const char *name) {
//   klee_make_symbolic(addr, nbytes, name);
// }


#ifdef __cplusplus
}
#endif

#endif
