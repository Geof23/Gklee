//=== logging.h --------------------------- C++ ====================//
//
// GKLEE logging support.  Distributed under MIT license, unless
// top level LICENSE.txt file indicates otherwise.
// 
// Instantiate an instance with location of log file and then call
// static methods to make log entries.  Call Stack oriented.
//------------------------------------------------------------------//

#ifndef GKLEE_LOGGING_H
#define GKLEE_LOGGING_H

#include <string>
#include <fstream>
#include <exception>

#include <llvm/Value.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/Casting.h>
#include <llvm/Instruction.h>

#include "klee/Expr.h"


namespace Gklee {

/* class loggingException : std::exception { */
/*   const char* what() const noexcept {return "You may only have one instance of Logging\n";} */
/*   }; */

class Logging{
 public:
  Logging( const std::string& logFile, size_t maxDepth);
  ~Logging();
  static void enterFunc( const std::string& fName, const std::string& data );
  static void enterFunc( const std::string& fName, const klee::ref<klee::Expr>& cond );  
  static void outItem( const std::string& name, const std::string& data );
  static void outItem( const std::string& name, const klee::ref<klee::Expr>& cond );
  template <typename V>
  static void outLLVMObj( const V& val );
  static void outInstruction( const llvm::Instruction& val );
  static void exitFunc();
 private:
  static std::ofstream lstream;
  static size_t level;
  static void tab();
  static size_t maxDepth;
  static bool first;
  static size_t count;
};
// used for LLVM objects (i.e. Value and Type) that have print methods
/* template <typename V> */
/* void  */
/* Logging::outLLVMObj( const V& val ){ */
/*   if( level <= maxDepth ){ */
/*     assert( lstream.is_open() && "You must instantiate Logging before calling its methods"); */
/*     lstream << "," << std::endl; */
/*     tab(); */
/*     lstream << "\"Instruction_" << count++ << "\": " << "\""; */
/*     if( llvm::isa< llvm::Instruction >( val ) ){ */
/*       llvm::raw_os_ostream roo( lstream ); */
/*       val.print( *(dynamic_cast< llvm::raw_ostream* >( &roo )), (llvm::AssemblyAnnotationWriter*)NULL); */
/*     }else{ */
/*       val.print( lstream ); */
/*     } */
/*     lstream << "\""; */
/*   } */
/* } */


}
	  
#endif
