//=== logging.cpp-------------------------- C++ ====================//
//
// GKLEE logging support.  Distributed under MIT license, unless
// top level LICENSE.txt file indicates otherwise.
// 
// Instantiate an instance with location of log file and then call
// static methods to make log entries.  Call Stack oriented. 
// JSON format, hierarchical by call graph
//------------------------------------------------------------------//
#include <cassert>
#include "klee/logging.h"

//using namespace llvm;
//using namespace klee;
using namespace Gklee;

std::ofstream Logging::lstream;
size_t Logging::maxDepth;
size_t Logging::level;


Logging::Logging( std::string logFile, size_t maxDepth ){ 
  //  std::string logFile( "log.txt" );
  level = 0;
  Logging::maxDepth = maxDepth;
  if( maxDepth > 0 ){
    assert( ! lstream.is_open() && "You may only have one instance of Logging" );
    // if( lstream.is_open() ) throw loggingException();
    lstream.open( logFile, std::ofstream::out |
		  std::ofstream::trunc);
    lstream << "{";
  }
}

Logging::~Logging(){
  if( lstream.is_open() ){ 
    lstream << std::endl << "}" << std::endl;
    lstream.close();
  }
}
	 
inline
void
Logging::tab(){
  lstream.width( level );
  lstream.fill( '\t' );
}

void
Logging::enterFunc(  std::string fName, 
		     std::string data ){
  if( level < maxDepth ){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    if( level > 0 ) lstream << ",";
    lstream << std::endl;
    ++level;
    tab();
    lstream << "\"" << fName << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"" << data << "\"";
  }
}

void
Logging::outItem(std::string name,
		 std::string data){
  if( level < maxDepth){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << "," << std::endl;
    tab();
    lstream << "\"" << name << "\": " << "\"" << data << "\"";
  }
}

void 
Logging::outInstruction( llvm::Value& val ){
    if( level < maxDepth){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << "," << std::endl;
    tab();
    lstream << "\"Instruction\": " << "\"";
    //    val.print( lstream );
    lstream << "\"";
  }
}

void
Logging::exitFunc(){
  if( level < maxDepth ){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << std::endl;
    --level;
    tab();
    lstream << "}" << std::endl;
  }
}
