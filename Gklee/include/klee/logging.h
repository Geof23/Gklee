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

namespace Gklee {

class loggingException : std::exception {
  const char* what() const noexcept {return "You may only have one instance of Logging\n";}
  };

class Logging{
 public:
  Logging( std::string logFile, size_t maxDepth);
  ~Logging();
  static void enterFunc( std::string fName, std::string data );
  static void outItem( std::string name, std::string data );
  static void outInstruction( llvm::Value& val );
  static void exitFunc();
 private:
  static std::ofstream lstream;
  static size_t level;
  static void tab();
  static size_t maxDepth;
};
}
	  
#endif
