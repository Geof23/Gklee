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
#include <map>

#include <llvm/Value.h>

namespace Gklee {

class Logging{
 public:
  Logging( const std::string& logFile, size_t maxDepth, 
	   bool startNow = true );
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
  static void start();
  static void stop();
 private:
  static std::map< std::string, std::string > Funcs;
  static std::stack< std::string > CallStack;
  static void outInstruction( const llvm::Value& val );
  static bool initLeadComma( const std::string& fun = std::string(), 
			     bool = false );
  static std::ofstream lstream;
  static size_t level;
  static void tab();
  static size_t maxDepth;
  static bool first;
  static size_t count;
  static bool paused;
  static size_t start_level; //this is for recursive start/stop calls, start -> start_level++, stop -> start_level--, enabled -> start_level > 0
};

}
	  
#endif
