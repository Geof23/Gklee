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
bool Logging::first = true;
size_t Logging::count = 0;

Logging::Logging( const std::string& logFile, size_t maxDepth ){ 
  //  std::string logFile( "log.txt" );
  level = 0;
  Logging::maxDepth = maxDepth;
  if( maxDepth > 0 ){
    assert( ! lstream.is_open() && "You may only have one instance of Logging" );
    // if( lstream.is_open() ) throw loggingException();
    lstream.open( logFile, std::ofstream::out |
		  std::ofstream::trunc);
    lstream << "{";
    ++level;
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
  for( size_t w = 0; w < level; ++w ){
    lstream << '\t';
  }
}

void
Logging::enterFunc( const std::string& fName, 
		    const std::string& data ){
  if( level <= maxDepth ){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    if( !first ){
      lstream << ",";
    }
    first = false;
    lstream << std::endl;
    tab();
    ++level;
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"" << data << "\"";
  }
}

void
Logging::enterFunc( const std::string& fName, 
		    const klee::ref<klee::Expr>& cond ){
  if( level <= maxDepth ){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    if( !first ){
      lstream << ",";
    }
    first = false;
    lstream << std::endl;
    tab();
    ++level;
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"";
    if( !cond.isNull() ){
      cond->print( lstream );
    }
    lstream << "\"";
  }
}

// void
// Logging::outList( const std::string& name,
		  

void
Logging::outItem( const std::string& name,
		  const std::string& data){
  if( level <= maxDepth){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << "," << std::endl;
    tab();
    lstream << "\"" << name << "_" << count++ << "\": " << "\"" << data << "\"";
  }
}

void
Logging::outItem( const std::string& name,
		  const klee::ref<klee::Expr>& cond ){
  if( level <= maxDepth){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << "," << std::endl;
    tab();
    lstream << "\"" << name << "_" << count++ << "\": " << "\"";
    if( !cond.isNull() ){
      cond->print( lstream );
    }
    lstream << "\"";
  }
}

void 
Logging::outInstruction( const llvm::Instruction& val ){
  
  if( level <= maxDepth ){
    assert( lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << "," << std::endl;
    tab();
    lstream << "\"Instruction_" << count++ << "\": " << "\"";
      llvm::raw_os_ostream roo( lstream );
      val.print( *(dynamic_cast< llvm::raw_ostream* >( &roo )), (llvm::AssemblyAnnotationWriter*)NULL);
    lstream << "\"";
  }
}

void
Logging::exitFunc(){
  if( level <= maxDepth ){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << std::endl;
    --level;
    tab();
    lstream << "}";
  }
}
