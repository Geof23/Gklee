//===-- ModuleUtil.cpp ----------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "klee/Internal/Support/ModuleUtil.h"
#include "klee/Config/Version.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Linker/Linker.h"
#include "llvm/IR/Module.h"
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 8)
#include "llvm/Assembly/AsmAnnotationWriter.h"
#else
#include "llvm/IR/AssemblyAnnotationWriter.h"
#endif
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/ValueTracking.h"
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 9)
#include "llvm/System/Path.h"
#else
#include "llvm/Support/Path.h"
#endif
//#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/ErrorOr.h"

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <memory>

using namespace llvm;
using namespace klee;

Module *klee::linkWithLibrary(Module *module, 
                              const std::string &libraryName) {
  // Linker linker("klee", module, false);
  Linker linker(module);
	//	std::unique_ptr<MemoryBuffer> BufferPtr;
  // OwningPtr<MemoryBuffer> BufferPtr;
	auto BufferPtr = MemoryBuffer::getFileOrSTDIN( libraryName );
	//	std::error_code ec=MemoryBuffer::getFileOrSTDIN( libraryName, BufferPtr );
  if (std::error_code ec = BufferPtr.getError()) {
    // klee_error("error loading library '%s': %s", libraryName,
    //            ec.message().c_str());
		assert( 0 && ec.message().c_str() );
  }
  
  Module *libModule = getLazyBitcodeModule( BufferPtr, 
					   getGlobalContext());
  if (std::error_code ec = libModule->getError()){
		assert( 0 && ec.message().c_str());
  if( linker.linkInModule( libModule ) ){
    assert(0 && "linking in library failed!");
  }

  return linker.getModule();
    
  // llvm::sys::Path libraryPath(libraryName);
  // bool native = false;
    
  // if (linker.LinkInFile(libraryPath, native)) {
  //   assert(0 && "linking in library failed!");
  // }
    
  // return linker.releaseModule();
}

Function *klee::getDirectCallTarget(CallSite cs) {
  Value *v = cs.getCalledValue();
  if (Function *f = dyn_cast<Function>(v)) {
    return f;
  } else if (llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(v)) {
    if (ce->getOpcode()==Instruction::BitCast)
      if (Function *f = dyn_cast<Function>(ce->getOperand(0)))
        return f;

    // NOTE: This assert may fire, it isn't necessarily a problem and
    // can be disabled, I just wanted to know when and if it happened.
    assert(0 && "FIXME: Unresolved direct target for a constant expression.");
  }
  
  return 0;
}

static bool valueIsOnlyCalled(const Value *v) {
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 8)
  for (Value::use_const_iterator it = v->use_begin(), ie = v->use_end();
       it != ie; ++it) {
#else
  for (Value::const_use_iterator it = v->use_begin(), ie = v->use_end();
       it != ie; ++it) {
#endif
    if (const Instruction *instr = dyn_cast<Instruction>(*it)) {
      if (instr->getOpcode()==0) continue; // XXX function numbering inst
      if (!isa<CallInst>(instr) && !isa<InvokeInst>(instr)) return false;
      
      // Make sure that the value is only the target of this call and
      // not an argument.
      for (unsigned i=1,e=instr->getNumOperands(); i!=e; ++i)
        if (instr->getOperand(i)==v)
          return false;
    } else if (const llvm::ConstantExpr *ce = 
               dyn_cast<llvm::ConstantExpr>(*it)) {
      if (ce->getOpcode()==Instruction::BitCast)
        if (valueIsOnlyCalled(ce))
          continue;
      return false;
    } else if (const GlobalAlias *ga = dyn_cast<GlobalAlias>(*it)) {
      // XXX what about v is bitcast of aliasee?
      if (v==ga->getAliasee() && !valueIsOnlyCalled(ga))
        return false;
    } else {
      return false;
    }
  }

  return true;
}

bool klee::functionEscapes(const Function *f) {
  return !valueIsOnlyCalled(f);
}
