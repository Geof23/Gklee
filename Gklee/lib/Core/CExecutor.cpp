//===-- CExecutor.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "Executor.h"

#include "Context.h"
#include "CoreStats.h"
#include "ExternalDispatcher.h"
#include "ImpliedValue.h"
#include "Memory.h"
#include "MemoryManager.h"
#include "PTree.h"
#include "Searcher.h"
#include "SeedInfo.h"
#include "SpecialFunctionHandler.h"
#include "StatsTracker.h"
#include "TimingSolver.h"
#include "UserSearcher.h"
#include "../Solver/SolverStats.h"

#include "klee/ExecutionState.h"
#include "klee/Expr.h"
#include "klee/Interpreter.h"
#include "klee/TimerStatIncrementer.h"
#include "klee/util/Assignment.h"
#include "klee/util/ExprPPrinter.h"
#include "klee/util/ExprUtil.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "klee/Config/Version.h"
#include "klee/Internal/ADT/KTest.h"
#include "klee/Internal/ADT/RNG.h"
#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "klee/Internal/Support/FloatEvaluation.h"
#include "klee/Internal/System/Time.h"

#include "llvm/Attributes.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#if LLVM_VERSION_CODE >= LLVM_VERSION(2, 7)
#include "llvm/LLVMContext.h"
#endif
#include "llvm/Module.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 9)
#include "llvm/System/Process.h"
#else
#include "llvm/Support/Process.h"
#endif
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif

#include <cassert>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <sys/mman.h>

#include <errno.h>
#include <cxxabi.h>

#include "CUDA.h"
// ***

using namespace llvm;
using namespace klee;

namespace {

  cl::opt<bool>
  SimplifySymIndices("simplify-sym-indices",
                     cl::init(false));

  cl::opt<unsigned>
  MaxSymArraySize("max-sym-array-size",
                  cl::init(0));

  cl::opt<bool>
  UseAsmAddresses("use-asm-addresses",
                  cl::init(false));
}

namespace runtime {
  cl::opt<bool>
  UseSymbolicConfig("symbolic-config", 
                    cl::desc("test whether to apply symbolic configuration"),
                    cl::init(false));
  cl::opt<bool>
  AvoidOOBCheck("avoid-oob-check", 
                cl::desc("avoid out of bound check under parametric flow"),
                cl::init(false));
}

using namespace runtime; 

ref<klee::ConstantExpr> Executor::evalConstant(const Constant *c) {
  if (const llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(c)) {
    ref<klee::ConstantExpr> expr = evalConstantExpr(ce);
    return expr;
  } else {
    if (const ConstantInt *ci = dyn_cast<ConstantInt>(c)) {
      return ConstantExpr::alloc(ci->getValue());
    } else if (const ConstantFP *cf = dyn_cast<ConstantFP>(c)) {
      return ConstantExpr::alloc(cf->getValueAPF().bitcastToAPInt());
    } else if (const GlobalValue *gv = dyn_cast<GlobalValue>(c)) {
      return globalAddresses.find(gv)->second;
    } else if (isa<ConstantPointerNull>(c)) {
      return Expr::createPointer(0);
    } else if (isa<UndefValue>(c) || isa<ConstantAggregateZero>(c)) {
      return ConstantExpr::create(0, getWidthForLLVMType(c->getType()));
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
    } else if (const ConstantDataSequential *cds =
                 dyn_cast<ConstantDataSequential>(c)) {
      std::vector<ref<Expr> > kids;
      for (unsigned i = 0, e = cds->getNumElements(); i != e; ++i) {
        ref<Expr> kid = evalConstant(cds->getElementAsConstant(i));
        kids.push_back(kid);
      }
      ref<Expr> res = ConcatExpr::createN(kids.size(), kids.data());
      return cast<ConstantExpr>(res);
#endif
    } else if (const ConstantStruct *cs = dyn_cast<ConstantStruct>(c)) {
      const StructLayout *sl = kmodule->targetData->getStructLayout(cs->getType());
      llvm::SmallVector<ref<Expr>, 4> kids;
      for (unsigned i = cs->getNumOperands(); i != 0; --i) {
        unsigned op = i-1;
        ref<Expr> kid = evalConstant(cs->getOperand(op));

        uint64_t thisOffset = sl->getElementOffsetInBits(op),
                 nextOffset = (op == cs->getNumOperands() - 1)
                              ? sl->getSizeInBits()
                              : sl->getElementOffsetInBits(op+1);
        if (nextOffset-thisOffset > kid->getWidth()) {
          uint64_t paddingWidth = nextOffset-thisOffset-kid->getWidth();
          kids.push_back(ConstantExpr::create(0, paddingWidth));
        }

        kids.push_back(kid);
      }
      ref<Expr> res = ConcatExpr::createN(kids.size(), kids.data());
      return cast<ConstantExpr>(res);
    } else if (const ConstantArray *ca = dyn_cast<ConstantArray>(c)){
      llvm::SmallVector<ref<Expr>, 4> kids;
      for (unsigned i = ca->getNumOperands(); i != 0; --i) {
        unsigned op = i-1;
        ref<Expr> kid = evalConstant(ca->getOperand(op));
        kids.push_back(kid);
      }
      ref<Expr> res = ConcatExpr::createN(kids.size(), kids.data());
      return cast<ConstantExpr>(res);
    } else {
      // Constant{Vector}
      assert(0 && "invalid argument to evalConstant()");
      return 0; // Fake returning
    }
  }
}

void Executor::initializeGlobalObject(ExecutionState &state, ObjectState *os,
                                      const Constant *c, 
                                      unsigned offset) {

  DataLayout *targetData = kmodule->targetData;
  if (const ConstantVector *cp = dyn_cast<ConstantVector>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cp->getType()->getElementType());
    for (unsigned i=0, e=cp->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cp->getOperand(i), 
			     offset + i*elementSize);
  } else if (isa<ConstantAggregateZero>(c)) {
    unsigned i, size = targetData->getTypeStoreSize(c->getType());
    for (i=0; i<size; i++)
      os->write8(offset+i, (uint8_t) 0);
  } else if (const ConstantArray *ca = dyn_cast<ConstantArray>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(ca->getType()->getElementType());
    for (unsigned i=0, e=ca->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, ca->getOperand(i), 
			     offset + i*elementSize);
  } else if (const ConstantStruct *cs = dyn_cast<ConstantStruct>(c)) {
    const StructLayout *sl =
      targetData->getStructLayout(cast<StructType>(cs->getType()));
    for (unsigned i=0, e=cs->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cs->getOperand(i), 
			     offset + sl->getElementOffset(i));
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
  } else if (const ConstantDataSequential *cds =
               dyn_cast<ConstantDataSequential>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cds->getElementType());
    for (unsigned i=0, e=cds->getNumElements(); i != e; ++i)
      initializeGlobalObject(state, os, cds->getElementAsConstant(i),
                             offset + i*elementSize);
#endif
  } else {
    unsigned StoreBits = targetData->getTypeStoreSizeInBits(c->getType());
    ref<ConstantExpr> C = evalConstant(c);

    // Extend the constant if necessary;
    assert(StoreBits >= C->getWidth() && "Invalid store size!");
    if (StoreBits > C->getWidth())
      C = C->ZExt(StoreBits);

    os->write(offset, C);
  }
}

void Executor::handleBuiltInVariablesAsSymbolic(ExecutionState &state, MemoryObject *mo, 
                                                std::string vname) {
  // starting from a single symbolic thread ..
  
  if (vname == "threadIdx") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::LOCAL;
    mo->is_builtin = true;
    mo->setName("tid");
    state.tinfo.thread_id_mo = mo;
    builtInSet.insert(vname);

    for (unsigned i = 0; i < GPUConfig::sym_num_threads; i++) {
      std::string tidName = "tid_arr_k" + llvm::utostr(state.kernelNum) + "_" + llvm::utostr(i);
      const Array *threadArray = new Array(tidName, mo->size);
      ObjectState *os = new ObjectState(mo, threadArray);
      state.addressSpace.getAddressSpace(GPUConfig::LOCAL, i).bindObject(mo, os); 
    }
  }
  else if (vname == "blockIdx") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::SHARED;
    mo->is_builtin = true;
    mo->setName("bid");
    state.tinfo.block_id_mo = mo;
    builtInSet.insert(vname);

    for (unsigned i = 0; i < GPUConfig::sym_num_blocks; i++) {
      std::string bidName = "bid_arr_k" + llvm::utostr(state.kernelNum) + "_" + llvm::utostr(i);
      const Array *blockArray = new Array(bidName, mo->size);
      ObjectState* os = new ObjectState(mo, blockArray);
      state.addressSpace.getAddressSpace(GPUConfig::SHARED, i).bindObject(mo, os);
    }
  }
  else if (vname == "blockDim") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::HOST;
    mo->is_builtin = true;
    mo->setName("bdim");
    ObjectState* os = new ObjectState(mo);
    os->write32(0, GPUConfig::BlockSize[0]);
    os->write32(4, GPUConfig::BlockSize[1]);
    os->write32(8, GPUConfig::BlockSize[2]);
    state.addressSpace.getAddressSpace(GPUConfig::HOST).bindObject(mo, os);
    state.tinfo.block_size_os = os;
    builtInSet.insert(vname);

    // construct the sym blockDim
    MemoryObject *symBDimMo = memory->allocateSymBlockDim();
    state.tinfo.sym_bdim_mo = symBDimMo;
    std::string name = "sym_bdim";
    const Array *symBDimArray = new Array(name, symBDimMo->size);
    ObjectState *bdimos = new ObjectState(state.tinfo.sym_bdim_mo, symBDimArray);
    state.addressSpace.getAddressSpace(GPUConfig::HOST).bindObject(state.tinfo.sym_bdim_mo, bdimos);
    state.addSymbolic(state.tinfo.sym_bdim_mo, symBDimArray);
  }
  else if (vname == "gridDim") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::HOST;
    mo->is_builtin = true;
    mo->setName("gdim");
    ObjectState* os = new ObjectState(mo);
    os->write32(0, GPUConfig::GridSize[0]);
    os->write32(4, GPUConfig::GridSize[1]);
    os->write32(8, GPUConfig::GridSize[2]);
    state.addressSpace.getAddressSpace(GPUConfig::HOST).bindObject(mo, os);
    state.tinfo.grid_size_os = os;
    builtInSet.insert(vname);

    // construct the sym gridDim
    MemoryObject *symGDimMo = memory->allocateSymGridDim();
    state.tinfo.sym_gdim_mo = symGDimMo;
    std::string name = "sym_gdim";
    const Array *symGDimArray = new Array(name, symGDimMo->size);
    ObjectState *gdimos = new ObjectState(state.tinfo.sym_gdim_mo, symGDimArray);
    state.addressSpace.getAddressSpace(GPUConfig::HOST).bindObject(state.tinfo.sym_gdim_mo, gdimos);
    state.addSymbolic(state.tinfo.sym_gdim_mo, symGDimArray);
  }
}

void Executor::handleBuiltInVariables(ExecutionState &state, MemoryObject* mo, 
				      std::string vname) {
  
  if (GPUConfig::verbose > 0 && 
      vname.compare(0, 5, "llvm.") && vname.compare(0, 4, ".str")) {   // don't print LLVM variables
    llvm::errs() << "Global: " << vname << " : " 
		 << CUDAUtil::getCTypeStr(mo->ctype) << "\n"; 
  }

  if (vname == "threadIdx") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::LOCAL;
    mo->is_builtin = true;
    mo->setName("tid");
    state.tinfo.thread_id_mo = mo;
    builtInSet.insert(vname);

    for (unsigned gz = 0; gz < GPUConfig::GridSize[2]; gz++)
      for (unsigned gy = 0; gy < GPUConfig::GridSize[1]; gy++)
        for (unsigned gx = 0; gx < GPUConfig::GridSize[0]; gx++) {
          unsigned g = (GPUConfig::GridSize[0] * (GPUConfig::GridSize[1] * gz + gy) + gx) * GPUConfig::block_size;
          for (unsigned z = 0; z < GPUConfig::BlockSize[2]; z++) 
            for (unsigned y = 0; y < GPUConfig::BlockSize[1]; y++) 
	      for (unsigned x = 0; x < GPUConfig::BlockSize[0]; x++) {
	  	unsigned k = g + GPUConfig::BlockSize[0] * (GPUConfig::BlockSize[1] * z + y) + x;
		ObjectState* os = new ObjectState(mo);
		os->write32(0, x);
		os->write32(4, y);
		os->write32(8, z);
		state.addressSpace.getAddressSpace(GPUConfig::LOCAL, k).bindObject(mo, os);
	      }
        }
  }
  else if (vname == "blockIdx") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::SHARED;
    mo->is_builtin = true;
    mo->setName("bid");
    state.tinfo.block_id_mo = mo;
    builtInSet.insert(vname);

    for (unsigned z = 0; z < GPUConfig::GridSize[2]; z++) 
      for (unsigned y = 0; y < GPUConfig::GridSize[1]; y++) 
        for (unsigned x = 0; x < GPUConfig::GridSize[0]; x++) {
	  unsigned k = GPUConfig::GridSize[0] * (GPUConfig::GridSize[1] * z + y)+ x;
	  ObjectState* os = new ObjectState(mo);
	  os->write32(0, x);
	  os->write32(4, y);
	  os->write32(8, z);
	  state.addressSpace.getAddressSpace(GPUConfig::SHARED, k).bindObject(mo, os);
	}
  }
  else if (vname == "blockDim") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::HOST;
    mo->is_builtin = true;
    mo->setName("bdim");
    ObjectState* os = new ObjectState(mo);
    os->write32(0, GPUConfig::BlockSize[0]);
    os->write32(4, GPUConfig::BlockSize[1]);
    os->write32(8, GPUConfig::BlockSize[2]);
    state.addressSpace.getAddressSpace(GPUConfig::HOST).bindObject(mo, os);
    state.tinfo.block_size_os = os;
    builtInSet.insert(vname);
  }
  else if (vname == "gridDim") {
    if (GPUConfig::verbose > 0)
      GKLEE_INFO << "Found built-in variable: " << vname << "\n";
    mo->ctype = GPUConfig::HOST;
    mo->is_builtin = true;
    mo->setName("gdim");
    ObjectState* os = new ObjectState(mo);
    os->write32(0, GPUConfig::GridSize[0]);
    os->write32(4, GPUConfig::GridSize[1]);
    os->write32(8, GPUConfig::GridSize[2]);
    state.addressSpace.getAddressSpace(GPUConfig::HOST).bindObject(mo, os);
    state.tinfo.grid_size_os = os;
    builtInSet.insert(vname);
  }
}

void Executor::initializeMissedBuiltInVariables(ExecutionState &state) {
  std::vector<std::string> totalVec;
  totalVec.push_back("gridDim");
  totalVec.push_back("blockDim");
  totalVec.push_back("blockIdx");
  totalVec.push_back("threadIdx");

  for (unsigned i = 0; i < totalVec.size(); i++) {
    if (builtInSet.find(totalVec[i]) == builtInSet.end()) {
      // Specific builtin variable not found ... 
      GPUConfig::CTYPE ctype = GPUConfig::UNKNOWN;
      if (totalVec[i].compare("gridDim") == 0 
          || totalVec[i].compare("blockDim") == 0) {
        ctype = GPUConfig::HOST;
      } else if (totalVec[i].compare("blockIdx") == 0) {
        ctype = GPUConfig::SHARED;
      } else {
        ctype = GPUConfig::LOCAL;
      }
      MemoryObject *mo = memory->allocateMissedBuiltInMO(ctype, totalVec[i]);

      if (UseSymbolicConfig)
        handleBuiltInVariablesAsSymbolic(state, mo, totalVec[i]);
      else
        handleBuiltInVariables(state, mo, totalVec[i]);
      
      builtInSet.insert(totalVec[i]);
    } 
  }
}

extern void *__dso_handle __attribute__ ((__weak__));

// Handle extern __shared__ case ...
void Executor::initializeExternalSharedGlobals(ExecutionState &state) {
  Module *m = kmodule->module;

  for (Module::const_global_iterator gi = m->global_begin(),
       e = m->global_end(); gi != e; ++gi) {
    // most of the non-local variables go here
    LLVM_TYPE_Q Type *ty = gi->getType()->getElementType();
    uint64_t size = kmodule->targetData->getTypeStoreSize(ty);
    const llvm::GlobalValue* gv = dyn_cast<llvm::GlobalValue>(gi);
    if (gv->hasSection()) {
      std::string s = gv->getSection();
      // Remove the case 'extern __shared__'
      if (size == 0 && s == "__shared__") {
        MemoryObject *oldMo = globalObjects.find(gi)->second;
        ref<Expr> oldAddr = oldMo->getBaseExpr();
        std::vector<unsigned> cVec;
        for (unsigned i = 0; i < kmodule->constants.size(); ++i) {
          Cell &c = kmodule->constantTable[i];
          if (oldAddr == c.value)
            cVec.push_back(i);
        }
        state.addressSpace.unbindObject(oldMo); 

        MemoryObject *mo = memory->allocate(state.maxKernelSharedSize, false, true, 
                                            false, is_GPU_mode, &*gi);
        mo->setName(gi->getName().str());
        // Replace the old mo with the new one...  
        globalObjects.find(gi)->second = mo; 
        globalAddresses.find(gi)->second = mo->getBaseExpr();
        ObjectState *os = bindObjectInState(state, mo, false); 
        if (!gi->hasInitializer())
          os->initializeToRandom();
        else {
          ObjectState *wos = state.addressSpace.getWriteable(mo, os);
          initializeGlobalObject(state, wos, gi->getInitializer(), 0);
        }

        // Update the constantTable w.r.t. the new constant
        for (unsigned i = 0; i < cVec.size(); i++) {
          unsigned n = cVec[i];
          Cell &c = kmodule->constantTable[n];
          c.value = mo->getBaseExpr(); 
          
          // insert the extern shared variable set 
          if (externSharedSet.size() == 0) {
            ExternSharedVarVec vec;
            externSharedSet.push_back(vec);
          } else {
            unsigned size = externSharedSet.size();
            ExternSharedVar &var = externSharedSet[size-1][0];
            if (var.kernelNum != state.kernelNum) {
              ExternSharedVarVec vec;
              externSharedSet.push_back(vec);
            }
          }
          unsigned size = externSharedSet.size();
          externSharedSet[size-1].push_back(ExternSharedVar(state.kernelNum, n, mo));
        }
      }
    }
  }
}

void Executor::initializeGlobals(ExecutionState &state) {
  Module *m = kmodule->module;

  if (m->getModuleInlineAsm() != "")
    klee_warning("executable has module level assembly (ignoring)");

  assert(m->lib_begin() == m->lib_end() &&
         "XXX do not support dependent libraries");

  // represent function globals using the address of the actual llvm function
  // object. given that we use malloc to allocate memory in states this also
  // ensures that we won't conflict. we don't need to allocate a memory object
  // since reading/writing via a function pointer is unsupported anyway.
  for (Module::iterator i = m->begin(), ie = m->end(); i != ie; ++i) {
    Function *f = i;
    ref<ConstantExpr> addr(0);

    // If the symbol has external weak linkage then it is implicitly
    // not defined in this module; if it isn't resolvable then it
    // should be null.
    if (f->hasExternalWeakLinkage() && 
        !externalDispatcher->resolveSymbol(f->getName())) {
      addr = Expr::createPointer(0);
    } else {
      addr = Expr::createPointer((unsigned long) (void*) f);
      legalFunctions.insert((uint64_t) (unsigned long) (void*) f);
    }
    
    globalAddresses.insert(std::make_pair(f, addr));
  }

  // allocate and initialize globals, done in two passes since we may
  // need address of a global in order to initialize some other one.

  // allocate memory objects for all globals
  for (Module::const_global_iterator i = m->global_begin(),
         e = m->global_end();
       i != e; ++i) {

    // bool is_llvm_var = !(i->getNameStr().compare(0, 5, "llvm."));
    // if (is_llvm_var)   // disregard LLVM variables
    //   continue;

    if (i->isDeclaration()) {
      // FIXME: We have no general way of handling unknown external
      // symbols. If we really cared about making external stuff work
      // better we could support user definition, or use the EXE style
      // hack where we check the object file information.

      LLVM_TYPE_Q Type *ty = i->getType()->getElementType();
      uint64_t size = kmodule->targetData->getTypeStoreSize(ty);

      // XXX - DWD - hardcode some things until we decide how to fix.
#ifndef WINDOWS
      if (i->getName() == "_ZTVN10__cxxabiv117__class_type_infoE") {
        size = 0x2C;
      } else if (i->getName() == "_ZTVN10__cxxabiv120__si_class_type_infoE") {
        size = 0x2C;
      } else if (i->getName() == "_ZTVN10__cxxabiv121__vmi_class_type_infoE") {
        size = 0x2C;
      }
#endif

      if (size == 0) {
        llvm::errs() << "Unable to find size for global variable: " 
                     << i->getName() 
                     << " (use will result in out of bounds access)\n";
      }

      MemoryObject *mo = memory->allocate(size, false, true, false, 
                                          is_GPU_mode, i);
      std::string vname = i->getName().str();

      // handle built-in variables
      if (UseSymbolicConfig)
        handleBuiltInVariablesAsSymbolic(state, mo, vname);
      else
        handleBuiltInVariables(state, mo, vname);
      
      ObjectState *os = 0; 
      if (!mo->is_builtin) {
        os = bindObjectInState(state, mo, false);
      }

      globalObjects.insert(std::make_pair(i, mo));
      globalAddresses.insert(std::make_pair(i, mo->getBaseExpr()));

      // Program already running = object already initialized.  Read
      // concrete value and write it to our copy.
      if (size && !mo->is_builtin) {
        void *addr;
        if (i->getName() == "__dso_handle") {
          addr = &__dso_handle; // wtf ?
        } else {
          addr = externalDispatcher->resolveSymbol(i->getName());
        }
        if (!addr)
          klee_error("unable to load symbol(%s) while initializing globals.", 
                     i->getName().data());

        for (unsigned offset=0; offset<mo->size; offset++)
          os->write8(offset, ((unsigned char*)addr)[offset]);
      }
    } else {  // the variable has a definition
      // most of the non-local variables go here
      LLVM_TYPE_Q Type *ty = i->getType()->getElementType();
      uint64_t size = kmodule->targetData->getTypeStoreSize(ty);
      MemoryObject *mo = 0;

      if (UseAsmAddresses && i->getName()[0]=='\01') {
        char *end;
        uint64_t address = ::strtoll(i->getName().str().c_str()+1, &end, 0);

        if (end && *end == '\0') {
          klee_message("NOTE: allocated global at asm specified address: %#08llx"
                       " (%llu bytes)",
                       (long long) address, (unsigned long long) size);
          mo = memory->allocateFixed(address, size, is_GPU_mode, &*i);
          mo->isUserSpecified = true; // XXX hack;
        }
      }

      if (!mo) {
        mo = memory->allocate(size, false, true, false, 
                              is_GPU_mode, &*i);
	mo->setName(i->getName().str());
      }
      assert(mo && "out of memory");
      std::string vname = i->getName().str();

      // handle built-in variables
      if (UseSymbolicConfig)
        handleBuiltInVariablesAsSymbolic(state, mo, vname);
      else
        handleBuiltInVariables(state, mo, vname);

      if (!mo->is_builtin) {
	ObjectState *os = bindObjectInState(state, mo, false);
	if (!i->hasInitializer())
	  os->initializeToRandom();
      }

      globalObjects.insert(std::make_pair(i, mo));
      globalAddresses.insert(std::make_pair(i, mo->getBaseExpr()));
    }
  }
  
  // link aliases to their definitions (if bound)
  for (Module::alias_iterator i = m->alias_begin(), ie = m->alias_end(); 
       i != ie; ++i) {
    // Map the alias to its aliasee's address. This works because we have
    // addresses for everything, even undefined functions. 
    globalAddresses.insert(std::make_pair(i, evalConstant(i->getAliasee())));
  }

  // once all objects are allocated, do the actual initialization
  for (Module::const_global_iterator i = m->global_begin(),
         e = m->global_end();
       i != e; ++i) {
    if (i->hasInitializer()) {
      MemoryObject *mo = globalObjects.find(i)->second;
      if (!mo->is_builtin) { 
	const ObjectState *os = state.addressSpace.findObject(mo);
	assert(os);
	ObjectState *wos = state.addressSpace.getWriteable(mo, os);
	initializeGlobalObject(state, wos, i->getInitializer(), 0);
	// if(i->isConstant()) os->setReadOnly(true);
      }
    }
  }

  // for debugging
  if (GPUConfig::verbose > 0) {
    std::cout << "\n **************** The contents of the memories (excluding the CPU memory)\n";
    state.addressSpace.dump();
  }
}

template <typename TypeIt>
void Executor::computeOffsets(KGEPInstruction *kgepi, TypeIt ib, TypeIt ie) {
  ref<ConstantExpr> constantOffset =
    ConstantExpr::alloc(0, Context::get().getPointerWidth());
  uint64_t index = 1;
  for (TypeIt ii = ib; ii != ie; ++ii) {
    if (LLVM_TYPE_Q StructType *st = dyn_cast<StructType>(*ii)) {
      const StructLayout *sl = kmodule->targetData->getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(ii.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned) ci->getZExtValue());
      constantOffset = constantOffset->Add(ConstantExpr::alloc(addend,
                                                               Context::get().getPointerWidth()));
    } else {
      const SequentialType *set = cast<SequentialType>(*ii);
      uint64_t elementSize =
        kmodule->targetData->getTypeStoreSize(set->getElementType());
      Value *operand = ii.getOperand();
      if (Constant *c = dyn_cast<Constant>(operand)) {
        ref<ConstantExpr> index =
          evalConstant(c)->SExt(Context::get().getPointerWidth());
        ref<ConstantExpr> addend =
          index->Mul(ConstantExpr::alloc(elementSize,
                                         Context::get().getPointerWidth()));
        constantOffset = constantOffset->Add(addend);
      } else {
        kgepi->indices.push_back(std::make_pair(index, elementSize));
      }
    }
    index++;
  }
  kgepi->offset = constantOffset->getZExtValue();
}

void Executor::bindInstructionConstants(KInstruction *KI) {
  KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(KI);

  if (GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(KI->inst)) {
    computeOffsets(kgepi, gep_type_begin(gepi), gep_type_end(gepi));
  } else if (InsertValueInst *ivi = dyn_cast<InsertValueInst>(KI->inst)) {
    computeOffsets(kgepi, iv_type_begin(ivi), iv_type_end(ivi));
    assert(kgepi->indices.empty() && "InsertValue constant offset expected");
  } else if (ExtractValueInst *evi = dyn_cast<ExtractValueInst>(KI->inst)) {
    computeOffsets(kgepi, ev_type_begin(evi), ev_type_end(evi));
    assert(kgepi->indices.empty() && "ExtractValue constant offset expected");
  }
}

ObjectState *Executor::bindObjectInState(ExecutionState &state, 
                                         const MemoryObject *mo,
                                         bool isLocal,
                                         const Array *array) {
  ObjectState *os = array ? new ObjectState(mo, array) : new ObjectState(mo);

  // bind the object to the right place
  unsigned t_b_index = mo->ctype == GPUConfig::LOCAL ? state.tinfo.get_cur_tid() : state.tinfo.get_cur_bid();
  state.addressSpace.bindObject(mo, os, t_b_index);

  // Its possible that multiple bindings of the same mo in the state
  // will put multiple copies on this list, but it doesn't really
  // matter because all we use this list for is to unbind the object
  // on function return.
  if (isLocal)
    state.getCurStack().back().allocas.push_back(mo);

  return os;
}


ObjectState *Executor::bindObjectInStateToShared(ExecutionState &state, 
                                                 const MemoryObject *mo, unsigned bid) {
  ObjectState *os = new ObjectState(mo);
  // bind the object to the right place
  state.addressSpace.bindObject(mo, os, bid);

  return os;
}

void Executor::resolveExact(ExecutionState &state,
                            ref<Expr> p,
                            ExactResolutionList &results, 
                            const std::string &name, 
			    GPUConfig::CTYPE ctype,  
			    unsigned b_t_index) {

  // XXX we may want to be capping this?
  ResolutionList rl;
  state.addressSpace.resolve(state, solver, p, rl, 0, 0, ctype, b_t_index);
  
  ExecutionState *unbound = &state;

  for (ResolutionList::iterator it = rl.begin(), ie = rl.end(); 
       it != ie; ++it) {
    ref<Expr> inBounds = EqExpr::create(p, it->first->getBaseExpr());
    
    StatePair branches = fork(*unbound, inBounds, true);
    
    if (branches.first)
      results.push_back(std::make_pair(*it, branches.first));

    unbound = branches.second;
    if (!unbound) // Fork failure
      break;
  }

  if (unbound) {
    terminateStateOnError(*unbound,
                          "memory error: invalid pointer: " + name,
                          "ptr.err",
                          getAddressInfo(*unbound, p));
  }
}

void Executor::dumpTmpOutOfBoundConfig(ExecutionState &state, 
                                       ref<Expr> boundExpr, 
                                       ref<Expr> offset) {
  bool benign = false;
  std::vector<SymBlockID_t> symBlockIDs;
  std::vector<SymThreadID_t> symThreadIDs;
  SymBlockDim_t symBlockDim(0, 0, 0);
 
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));

  ref<Expr> oobCond = Expr::createIsZero(boundExpr);
  std::vector< ref<Expr> > offsetVec, cOffsetVec;
  offsetVec.push_back(offset); 
  bool success = getSymbolicConfigSolution(state, oobCond, offsetVec, 
                                           cOffsetVec, 0, 0, 
                                           benign, symBlockIDs, 
                                           symThreadIDs, symBlockDim); 
  if (success) {
    std::cout << "+++++++++++++++" << std::endl;
    std::cout << "OOB cond: " << std::endl;
    oobCond->dump();

    std::cout << "Thread : { <" << symBlockIDs[0].x << ", " 
              << symBlockIDs[0].y << ">" << ", <" << symThreadIDs[0].x
              << ", " << symThreadIDs[0].y << ", " << symThreadIDs[0].z
              << "> }" << " incurs out of bound error!" << std::endl; 

    // concretize the 'offset' expression w.r.t bindings ...
    std::cout << "The concretized offset: " << std::endl;
    cOffsetVec[0]->dump(); 
    std::cout << "+++++++++++++++" << std::endl;
  }
}

void Executor::executeMemoryOperation(ExecutionState &state,
                                      bool isWrite,
                                      ref<Expr> address,
                                      ref<Expr> value /* undef if read */,
                                      KInstruction *target, 
                                      unsigned seqNum, bool isAtomic) {
  Expr::Width type = (isWrite ? value->getWidth() : 
		      getWidthForLLVMType(target->inst->getType()));
  unsigned bytes = Expr::getMinBytesForWidth(type);
  
  if (SimplifySymIndices) {
    if (!isa<ConstantExpr>(address))
      address = state.constraints.simplifyExpr(address);
    if (isWrite && !isa<ConstantExpr>(value))
      value = state.constraints.simplifyExpr(value);
  }

  // fast path: single in-bounds resolution
  ObjectPair op;
  bool success;
  solver->setTimeout(stpTimeout);

  // by Guodong
  HierAddressSpace& addrSpace = state.addressSpace;

  // if target is null, then the memory operation is local
  GPUConfig::CTYPE ctype = GPUConfig::LOCAL;  

  if (address->ctype != GPUConfig::UNKNOWN) {
    ctype = address->ctype;
  } else {
    if (target) {
      llvm::Instruction* inst = target->inst;
      llvm::Value *v = isWrite ? inst->getOperand(1) : inst->getOperand(0);
      ctype = CUDAUtil::getCType(v, is_GPU_mode);
    }
  }
  //std::cout << "execute memory operation ctype: " 
  //          << CUDAUtil::getCTypeStr(ctype) << std::endl;
  //std::cout << "Address CTYPE: " << CUDAUtil::getCTypeStr(address->ctype) << std::endl;
  unsigned b_t_index = ctype == GPUConfig::LOCAL ? state.tinfo.get_cur_tid() : state.tinfo.get_cur_bid();

  if (!addrSpace.resolveOne(state, solver, address, op, success, ctype, b_t_index)) {
    address = toConstant(state, address, "resolveOne failure");
    success = addrSpace.resolveOne(cast<ConstantExpr>(address), op, ctype, b_t_index);
  }

  solver->setTimeout(0);

  if (success) {
    const MemoryObject *mo = op.first;

    if (MaxSymArraySize && mo->size>=MaxSymArraySize) {
      address = toConstant(state, address, "max-sym-array-size");
    }
    
    ref<Expr> offset = mo->getOffsetExpr(address);
   
    bool inBounds;
    solver->setTimeout(stpTimeout);

    bool success = false;
    if (UseSymbolicConfig && 
         state.tinfo.is_GPU_mode) {
      if (!AvoidOOBCheck) {
        ExecutorUtil::copyOutConstraint(state);
        success = solver->mustBeTrue(state, 
                                     mo->getBoundsCheckOffset(offset, bytes),
                                     inBounds);
        if (!inBounds) {
          dumpTmpOutOfBoundConfig(state, 
                                  mo->getBoundsCheckOffset(offset, bytes), 
                                  offset);
        }
        ExecutorUtil::copyBackConstraint(state);
      } else {
        success = true;
        inBounds = true;
      }
    } else {
      success = solver->mustBeTrue(state, 
                                   mo->getBoundsCheckOffset(offset, bytes),
                                   inBounds);
    }

    solver->setTimeout(0);
    if (!success) {
      state.setPC(state.getPrevPC());
      terminateStateEarly(state, "query timed out");
      return;
    }

    if (inBounds) {      // no memory out-of-bound error
      const ObjectState *os = op.second;
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnError(state,
                                "memory error: object read only",
                                "readonly.err");
        } else {
          // memory type inference
          ObjectState *wos = state.addressSpace.getWriteable(mo, os, b_t_index);
          wos->write(offset, value);

          if (!UseSymbolicConfig)
	    state.addressSpace.addWrite(state.tinfo.is_GPU_mode, mo, 
                                        offset, value, type, 
	  			        state.tinfo.get_cur_bid(), 
                                        state.tinfo.get_cur_tid(),
                                        target->inst, seqNum, isAtomic, 
                                        b_t_index);
          else {
            ref<Expr> accessExpr = state.getTDCCondition();
	    state.addressSpace.addWrite(state.tinfo.is_GPU_mode, mo, 
                                        offset, value, type, 
	  			        state.tinfo.get_cur_bid(), 
                                        state.tinfo.get_cur_tid(),
                                        target->inst, seqNum, isAtomic, 
                                        b_t_index, accessExpr);

          }
        }
      } else {
        // memory type inference
        ref<Expr> result = os->read(offset, type);

        if (!UseSymbolicConfig) {
  	  state.addressSpace.addRead(state.tinfo.is_GPU_mode, mo, offset, result, type,
	  			     state.tinfo.get_cur_bid(), state.tinfo.get_cur_tid(), 
                                     target->inst, seqNum, isAtomic, b_t_index);
        } else {
          ref<Expr> accessExpr = state.getTDCCondition();
  	  state.addressSpace.addRead(state.tinfo.is_GPU_mode, mo, offset, result, type,
	  			     state.tinfo.get_cur_bid(), state.tinfo.get_cur_tid(), 
                                     target->inst, seqNum, isAtomic, b_t_index, accessExpr);

        }

        if (interpreterOpts.MakeConcreteSymbolic)
          result = replaceReadWithSymbolic(state, result);
 
        if (!isAtomic)
          bindLocal(target, state, result);
        else
          atomicRes = result;
      }

      return;
    }
  } 

  // we are on an error path (no resolution, multiple resolution, one
  // resolution with out of bounds)
  
  ResolutionList rl;  
  solver->setTimeout(stpTimeout);

  bool incomplete = state.addressSpace.resolve(state, solver, address, rl,
                                               0, stpTimeout, ctype, b_t_index);

  solver->setTimeout(0);
  
  // XXX there is some query wasteage here. who cares?
  ExecutionState *unbound = &state;
  
  for (ResolutionList::iterator i = rl.begin(), ie = rl.end(); i != ie; ++i) {
    const MemoryObject *mo = i->first;
    const ObjectState *os = i->second;
    ref<Expr> inBounds = mo->getBoundsCheckPointer(address, bytes);
    
    StatePair branches = fork(*unbound, inBounds, true);
    ExecutionState *bound = branches.first;
    
    // bound can be 0 on failure or overlapped 
    if (bound) {
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnError(*bound,
                                "memory error: object read only",
                                "readonly.err");
        } else {
          ObjectState *wos = bound->addressSpace.getWriteable(mo, os);
          wos->write(mo->getOffsetExpr(address), value);
        }
      } else {
        ref<Expr> result = os->read(mo->getOffsetExpr(address), type);
        bindLocal(target, *bound, result);
      }
    } 

    unbound = branches.second;
    if (!unbound)
      break;
  }
  // XXX should we distinguish out of bounds and overlapped cases?
  if (unbound) {
    if (incomplete) {
      terminateStateEarly(*unbound, "query timed out (resolve)");
    } else {
      terminateStateOnError(*unbound,
                            "memory error: out of bound pointer",
                            "ptr.err",
                            getAddressInfo(*unbound, address));
    }
  }
}

void Executor::executeNoMemoryCoalescing(ExecutionState &state, ref<Expr> &noMCCond) {
  if (GPUConfig::verbose > 0){
    std::cout << "No memory coalescing condition: " << std::endl;
    noMCCond->dump();
  }
  ExecutionState *anoState = state.branch(); 
  addConstraint(*anoState, noMCCond);
  char suffix[256] = "mc";
  interpreterHandler->processTestCase(*anoState, "execution encounters the performance defect of non-memory coalescing", "mc.err", suffix);
  traceInfo.empty();
  delete anoState;
}

void Executor::executeBankConflict(ExecutionState &state, ref<Expr> &bcCond) {
  if (GPUConfig::verbose > 0){
    std::cout << "Bank conflict condition: " << std::endl;
    bcCond->dump();
  }
  ExecutionState *anoState = state.branch(); 
  addConstraint(*anoState, bcCond);
  char suffix[256] = "bc";
  interpreterHandler->processTestCase(*anoState, "execution encounters the performance defect of bank conflict", "bc.err", suffix);
  traceInfo.empty();
  delete anoState;
}

void Executor::executeVolatileMissing(ExecutionState &state, ref<Expr> &vmCond) {
  if (GPUConfig::verbose > 0) {
    std::cout << "Volatile missing condition: " << std::endl;
    vmCond->dump();
  }
  ExecutionState *anoState = state.branch(); 
  addConstraint(*anoState, vmCond);
  char suffix[256] = "vm";
  interpreterHandler->processTestCase(*anoState, "execution encounters the potential volatile missing", "vm.err", suffix);
  traceInfo.empty();
  delete anoState;
}

void Executor::executeRaceCondition(ExecutionState &state, ref<Expr> &raceCond) {
  bool result = false;
  bool success = false;

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(raceCond)) {
    result = CE->isTrue()? true : false;
    success = true;
  } else {
    raceCond = state.constraints.simplifyExpr(raceCond);
    success = solver->mustBeTrue(state, raceCond, result); 
  }
  
  if (success) {
    if (result)
      terminateStateOnExecError(state, "execution halts on encountering a race");
    else {
      StatePair branches = fork(state, Expr::createIsZero(raceCond), true); 
      ExecutionState *raceState = branches.second;
 
      if (raceState) {
        terminateStateOnExecError(*raceState, "execution halts on encountering a race");
      } 
    }
  }
  else {
    terminateStateOnExecError(state, "execution halts on encountering a race");
  }
}

// CUDA doesn't support dynamic allocation in GPU; thus only stack or host allocation are allowed
void Executor::executeAlloc(ExecutionState &state,
                            ref<Expr> size,
                            bool isLocal,
                            KInstruction *target,
                            bool zeroMemory,
                            const ObjectState *reallocFrom) {
  size = toUnique(state, size);

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(size)) {
    MemoryObject *mo = NULL;
    mo = memory->allocate(CE->getZExtValue(), isLocal, false, state.deviceSet, 
                          is_GPU_mode, state.getPrevPC()->inst);
    //std::cout << "mo alloc: " << mo->address 
    //          << ", size: " << mo->size 
    //          << ", ctype: " << CUDAUtil::getCTypeStr(mo->ctype) << std::endl;
    //mo->name = target->inst->getNameStr();   // this will help debugging the races
    //std::cout << "mo name : " << mo->name << std::endl;

    if (!mo) {   //  null pointer
      bindLocal(target, state, 
                ConstantExpr::alloc(0, Context::get().getPointerWidth()));
    } else {
      ObjectState *os = bindObjectInState(state, mo, isLocal);
      if (zeroMemory) {
        os->initializeToZero();
      } else {
        os->initializeToRandom();
      }
      bindLocal(target, state, mo->getBaseExpr());
      
      if (reallocFrom) {
        unsigned count = std::min(reallocFrom->size, os->size);
        for (unsigned i=0; i<count; i++)
          os->write(i, reallocFrom->read8(i));
	unsigned t_b_index = state.tinfo.get_cur_tid();
        state.addressSpace.unbindObject(reallocFrom->getObject(), t_b_index);
      }
    }
  } else {
    // XXX For now we just pick a size. Ideally we would support
    // symbolic sizes fully but even if we don't it would be better to
    // "smartly" pick a value, for example we could fork and pick the
    // min and max values and perhaps some intermediate (reasonable
    // value).
    // 
    // It would also be nice to recognize the case when size has
    // exactly two values and just fork (but we need to get rid of
    // return argument first). This shows up in pcre when llvm
    // collapses the size expression with a select.

    ref<ConstantExpr> example;
    ExecutorUtil::copyOutConstraintUnderSymbolic(state);

    bool success = solver->getValue(state, size, example);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    
    // Try and start with a small example.
    Expr::Width W = example->getWidth();
    while (example->Ugt(ConstantExpr::alloc(128, W))->isTrue()) {
      ref<ConstantExpr> tmp = example->LShr(ConstantExpr::alloc(1, W));
      bool res;
      bool success = solver->mayBeTrue(state, EqExpr::create(tmp, size), res);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      if (!res)
        break;
      example = tmp;
    }

    ExecutorUtil::copyBackConstraintUnderSymbolic(state);

    StatePair fixedSize = fork(state, EqExpr::create(example, size), true);
    
    if (fixedSize.second) { 
      // Check for exactly two values
      ref<ConstantExpr> tmp;
      bool success = solver->getValue(*fixedSize.second, size, tmp);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      bool res;
      success = solver->mustBeTrue(*fixedSize.second, 
                                   EqExpr::create(tmp, size),
                                   res);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      if (res) {
        executeAlloc(*fixedSize.second, tmp, isLocal,
                     target, zeroMemory, reallocFrom);
      } else {
        // See if a *really* big value is possible. If so assume
        // malloc will fail for it, so lets fork and return 0.
        StatePair hugeSize = 
          fork(*fixedSize.second, 
               UltExpr::create(ConstantExpr::alloc(1<<31, W), size), 
               true);
        if (hugeSize.first) {
          klee_message("NOTE: found huge malloc, returing 0");
          bindLocal(target, *hugeSize.first, 
                    ConstantExpr::alloc(0, Context::get().getPointerWidth()));
        }
        
        if (hugeSize.second) {
          std::ostringstream info;
          ExprPPrinter::printOne(info, "  size expr", size);
          info << "  concretization : " << example << "\n";
          info << "  unbound example: " << tmp << "\n";
          terminateStateOnError(*hugeSize.second, 
                                "concretized symbolic size", 
                                "model.err", 
                                info.str());
        }
      }
    }

    if (fixedSize.first) // can be zero when fork fails
      executeAlloc(*fixedSize.first, example, isLocal, 
                   target, zeroMemory, reallocFrom);
  }
}

void Executor::executeFree(ExecutionState &state,
                           ref<Expr> address,
                           KInstruction *target) {
  StatePair zeroPointer = fork(state, Expr::createIsZero(address), true);
  if (zeroPointer.first) {
    if (target)
      bindLocal(target, *zeroPointer.first, Expr::createPointer(0));
  }
  if (zeroPointer.second) { // address != 0
    ExactResolutionList rl;
    resolveExact(*zeroPointer.second, address, rl, "free", address->ctype);  // free host memory??
    
    for (Executor::ExactResolutionList::iterator it = rl.begin(), 
           ie = rl.end(); it != ie; ++it) {
      const MemoryObject *mo = it->first.first;
      if (mo->isLocal) {
        terminateStateOnError(*it->second, 
                              "free of alloca", 
                              "free.err",
                              getAddressInfo(*it->second, address));
      } else if (mo->isGlobal) {
        terminateStateOnError(*it->second, 
                              "free of global", 
                              "free.err",
                              getAddressInfo(*it->second, address));
      } else {
        it->second->addressSpace.unbindObject(mo);
        if (target)
          bindLocal(target, *it->second, Expr::createPointer(0));
      }
    }
  }
}

