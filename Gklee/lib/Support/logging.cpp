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
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Function.h>
#include "klee/Internal/Module/KModule.h"

#include "../Core/Memory.h"
#include "../Core/AddressSpace.h"

#include "klee/Expr.h"
#include "klee/logging.h"

namespace Gklee {

std::ofstream Logging::lstream;
size_t Logging::maxDepth;
size_t Logging::level;
bool Logging::first = true;
size_t Logging::count = 0;
bool Logging::paused = false;
size_t Logging::start_level;
Logging::mapType Logging::Funcs = {{ "void klee::Executor::executeInstruction(klee::ExecutionState&, klee::KInstruction*)", "" }};
// std::map< std::string, std::string > Logging::Funcs = {{ "Executor::executeInstruction", "" }};
std::stack< std::string > Logging::CallStack;

Logging::Logging( const std::string& logFile, size_t maxDepth,
		    bool startNow) { 
  //  std::string logFile( "log.txt" );
  
  level = 0;
  paused = !startNow;
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
/// also performs proper indentation prior to output line and leading comma
inline
bool
Logging::initLeadComma( const std::string& fun ){
  bool retVal = true;
  bool newCall;
  std::string _fun;
  if( !fun.empty()){
    CallStack.push( fun );
    _fun = fun;
    newCall = true;
  }else{
    _fun = CallStack.top();
    newCall = false;
  }

  if( !paused && level < maxDepth && Funcs.find( _fun ) 
      != Funcs.end()){
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
Logging::enterFunc( const std::string& data, 
		    const std::string& fName ){
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"" << data << "\"";
  }
}

template <>
void
Logging::enterFunc( const klee::MemoryObject& mo,
		    const std::string& fName ){
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"allocInfo\": \"";
    std::string ai;
    mo.getAllocInfo( ai );
    lstream << ai << "\"";
  }
}

template <>
void
Logging::enterFunc( const klee::KFunction& kfunc,
		    const std::string& fName ){
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"frameName\": \"";
    lstream << kfunc.function->getName().str() << "\"";
    //<< data << "\"";
  }
}

template <>
void
Logging::enterFunc( const llvm::Instruction& i,
		    const std::string& fName ){
  if( initLeadComma( fName )){
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
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"data\": \"";
    if( !cond.isNull() ){
      cond->print( lstream, true );
    }
    lstream << "\"";
  }
}

template <>
void
Logging::enterFunc( const std::vector<klee::ref<klee::Expr>>& conds,
		    const std::string& fName ){
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    size_t cnt = 0;
    for(auto cond: conds){
      if( !cond.isNull() ){
	lstream << " \"cond_" << cnt++ << ":";
	cond->print( lstream, true );
	lstream << "\"";
      }
    }
  }
}

template <>
void
Logging::enterFunc( const llvm::Instruction& i1,
		    const llvm::Instruction& i2,
		    const std::string& fName ){
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"";
    outInstruction( i1 );
    lstream << ":";
    outInstruction( i2 );
    lstream << "\"";
  }
}

template <>
void
Logging::enterFunc( const klee::ref< klee::Expr >& e1,
		    const klee::ref< klee::Expr >& e2,
		    const std::string& fName ){
  if( initLeadComma( fName )){
    lstream << "\"" << fName << "_" << count++ << "\":" << " {" << std::endl;
    tab();
    lstream << "\"";
    e1->print( lstream, true );
    lstream << ":";
    e2->print( lstream, true );
    lstream << "\"";
  }
}

template<>
void
Logging::outItem( const klee::HierAddressSpace& as,
		  const std::string& name ){
  if( initLeadComma()){
    lstream << "\"" << name << "_" << count++ << "\": \"cpu:" << std::endl;
    tab();
    as.cpuMemory.dump( false, lstream, level+1 );
    lstream << std::endl;
    tab();
    lstream << " device:" << std::endl;
    tab();
    as.deviceMemory.dump( false, lstream, level+1 );
    lstream << std::endl;
    tab();
    lstream << " shared:" << std::endl;
    tab();
    for(size_t i = 0; i < as.sharedMemories.size(); ++i){
      lstream << "(" << i << "):" << std::endl;
      tab();
      as.sharedMemories[i].dump( false, lstream, level+1 );
    }
    lstream << std::endl;
    tab();
    lstream << " local (tid):" << std::endl;
    tab();
    for(size_t i = 0; i < as.localMemories.size(); ++i){
      lstream << "(" << i << "):" << std::endl;
      tab();
      as.localMemories[i].dump( false, lstream, level+1 );
    }
    lstream << "\"";
  }
}

template<>
void
Logging::outItem( const klee::MemoryObject& mo,
		  const std::string& name ){
  if( initLeadComma()){
    std::string ai;
    mo.getAllocInfo( ai );
    lstream << "\"" << name << "_" << count++ << "\": " << "\"" << ai << "\"";
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

template<>
void
Logging::outItem( const llvm::Value& data,
		  const std::string& name ){
  if( initLeadComma()){
    lstream << "\"" << name << "_" << count++ << "\": " << "\"";
    outInstruction( data );
    lstream << "\"";
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
      cond->print( lstream, true );
    }
    lstream << "\"";
  }
}

void 
//Logging::outInstruction( const llvm::Instruction& val ){
Logging::outInstruction( const llvm::Value& val ){ //instruction is 2nd order subclass of value
  llvm::raw_os_ostream roo( lstream );
  val.print( *(dynamic_cast< llvm::raw_ostream* >( &roo )), 
	     (llvm::AssemblyAnnotationWriter*)NULL);
}

void
Logging::exitFunc(){
  std::string fun = CallStack.top();
  CallStack.pop();
  if( !paused && level < maxDepth && Funcs.find( fun ) 
      != Funcs.end()){
    assert(lstream.is_open() && "You must instantiate Logging before calling its methods");
    lstream << std::endl;
    --level;
    tab();
    lstream << "}";
  }
}

void
Logging::start(){
  ++start_level;
  paused = false;
}

void
Logging::stop(){
  assert( start_level > 0 && "logging stop called without start" );
  --start_level;
  if( start_level == 0 ){
    paused = true;
  }
}

}
