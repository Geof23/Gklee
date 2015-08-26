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
#include <stack>
#include <unordered_map>
#include <functional>

#include <llvm/Value.h>

#include "klee/Expr.h"

#include "FlowGraph.h"

namespace Gklee {

class Logging{
 public:
  Logging( const std::string& logFile );
  ~Logging();
  template<typename T>
    static void enterFunc( T const& data, const std::string& fName );
  template < typename T >
    static void enterFunc( T const& data1,
			   T const& data2, const std::string& fName );
  template < typename T > 
    static void outItem( T const& data, const std::string& name );
  template < typename V >
    static void outLLVMObj( const V& val );
  static void exitFunc();
  /* typedef struct{ */
  /*   bool operator() ( std::string a, std::string b ) const { return a.find( b ) != std::string::npos || */
  /* 	b.find( a ) != std::string::npos; } */
  /* } str_eq; */
  template < typename T >
  static void fgInfo( const std::string& type,
		      const T& data,
		      const klee::ref<klee::Expr>& cond = klee::ConstantExpr::alloc(1, klee::Expr::Bool) );

 private:
  static std::string getCondString( const klee::ref<klee::Expr>& cond );
  typedef std::unordered_map< std::string, std::string > mapType;
  static  mapType Funcs;
  static std::stack< std::string > CallStack;
  static void outInstruction( const llvm::Value& val );
  static std::string getInstString( const llvm::Value& val );
  static bool initLeadComma( const std::string& fun = std::string() );
  static std::ofstream lstream;
  static size_t level;
  static void tab();
  static bool first;
  static size_t count;
  static FlowGraph fg;
};

}
	  
#endif
