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
namespace Gklee {

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


///
/// Returns true if the Logging object is accepting output,
/// also performs propper indentation prior to output line and leading comma
inline
bool
Logging::initLeadComma( bool newCall ){
  bool retVal = true;
  if( level <= maxDepth ){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    if( !first ){
      lstream << ",";
    }
    first = false;
    lstream << std::endl;
    tab();
    if( newCall ){
      ++level;
    }
  }else{
    retVal = false;
  }
  return retVal;
}

template <>
void
Logging::enterFunc( const klee::KFunction& kfunc,
		    const std::string& fName ){
  if( initLeadComma( true )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"frameName\": \"";
    lstream << kfunc.function->getName().str() << "\"";
    //<< data << "\"";
  }
}

template <>
void
Logging::enterFunc( const std::string& data, 
		    const std::string& fName ){
  if( initLeadComma( true )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"" << data << "\"";
  }
}

template <>
void
Logging::enterFunc( const llvm::Instruction& i,
		    const std::string& fName ){
  if( initLeadComma( true )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"";
    outInstruction( i );
    lstream << "\"";
  }
}

template <>
void
Logging::enterFunc( const klee::ref<klee::Expr>& cond,
		    const std::string& fName ){
  if( initLeadComma( true )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"";
    if( !cond.isNull() ){
      cond->print( lstream );
    }
    lstream << "\"";
  }
}

template <>
void
Logging::enterFunc( const llvm::Instruction& i1,
		    const llvm::Instruction& i2,
		    const std::string& fName ){
  if( initLeadComma( true )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"";
    outInstruction( i1 );
    lstream << ":";
    outInstruction( i2 );
    lstream << "\"";
  }
}

template<>
void
Logging::outItem( const std::string& data,
		  const std::string& name ){
  if( initLeadComma()){
    lstream << "\"" << name << "_" << count++ << "\": " << "\"" << data << "\"";
  }
}

template <>
void
Logging::outItem( const klee::KFunction& kfunc,
		  const std::string& name ){
  if( initLeadComma()){
    lstream << "\"" << name << "_" << count++ << "\": " << "\"" << kfunc.function->getName().str()  << "\"";
  }
}

template <>
void
Logging::outItem( const klee::ref<klee::Expr>& cond,
		  const std::string& name ){
  if( initLeadComma()){
    lstream << "\"" << name << "_" << count++ << "\": " << "\"";
    if( !cond.isNull() ){
      cond->print( lstream );
    }
    lstream << "\"";
  }
}

void 
Logging::outInstruction( const llvm::Instruction& val ){
  llvm::raw_os_ostream roo( lstream );
  val.print( *(dynamic_cast< llvm::raw_ostream* >( &roo )), 
	     (llvm::AssemblyAnnotationWriter*)NULL);
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

}
