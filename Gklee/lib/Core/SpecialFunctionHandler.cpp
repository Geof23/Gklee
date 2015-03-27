//===-- SpecialFunctionHandler.cpp ----------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "Memory.h"
#include "SpecialFunctionHandler.h"
#include "TimingSolver.h"
// #include "../FLA/StringSolver.h"

#include "klee/ExecutionState.h"

#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"

#include "Executor.h"
#include "CUDA.h"

#include "MemoryManager.h"

#include "llvm/Module.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
#include "llvm/Analysis/DebugInfo.h"
#else
#include "llvm/DebugInfo.h"
#endif

#include <errno.h>

using namespace llvm;
using namespace klee;


namespace runtime {
  extern cl::opt<bool> UseSymbolicConfig;
  extern cl::opt<bool> Emacs;
}

using namespace runtime;

/// \todo Almost all of the demands in this file should be replaced
/// with terminateState calls.

///

struct HandlerInfo {
  const char *name;
  SpecialFunctionHandler::Handler handler;
  bool doesNotReturn; /// Intrinsic terminates the process
  bool hasReturnValue; /// Intrinsic has a return value
  bool doNotOverride; /// Intrinsic should not be used if already defined
};

// FIXME: We are more or less committed to requiring an intrinsic
// library these days. We can move some of this stuff there,
// especially things like realloc which have complicated semantics
// w.r.t. forking. Among other things this makes delayed query
// dispatch easier to implement.
HandlerInfo handlerInfo[] = {
#define add(name, handler, ret) { name, \
                                  &SpecialFunctionHandler::handler, \
                                  false, ret, false }
#define addDNR(name, handler) { name, \
                                &SpecialFunctionHandler::handler, \
                                true, false, false }
  addDNR("__assert_rtn", handleAssertFail),
  addDNR("__assert_fail", handleAssertFail),
  addDNR("_assert", handleAssert),
  addDNR("abort", handleAbort),
  addDNR("_exit", handleExit),
  { "exit", &SpecialFunctionHandler::handleExit, true, false, true },
  addDNR("klee_abort", handleAbort),
  addDNR("klee_silent_exit", handleSilentExit),  
  addDNR("klee_report_error", handleReportError),

  add("calloc", handleCalloc, true),
  add("free", handleFree, false),
  add("klee_assume", handleAssume, false),
  add("klee_check_memory_access", handleCheckMemoryAccess, false),
  add("klee_get_valuef", handleGetValue, true),
  add("klee_get_valued", handleGetValue, true),
  add("klee_get_valuel", handleGetValue, true),
  add("klee_get_valuell", handleGetValue, true),
  add("klee_get_value_i32", handleGetValue, true),
  add("klee_get_value_i64", handleGetValue, true),
  add("klee_get_value", handleGetValue, true),
  add("klee_define_fixed_object", handleDefineFixedObject, false),
  add("klee_get_obj_size", handleGetObjSize, true),
  add("klee_get_errno", handleGetErrno, true),
  add("klee_is_symbolic", handleIsSymbolic, true),
  add("klee_make_symbolic", handleMakeSymbolic, false),
  add("klee_mark_global", handleMarkGlobal, false),
  add("klee_merge", handleMerge, false),
  add("klee_prefer_cex", handlePreferCex, false),
  add("klee_print_expr", handlePrintExpr, false),
  add("klee_print_range", handlePrintRange, false),
  add("klee_set_forking", handleSetForking, false),
  add("klee_stack_trace", handleStackTrace, false),
  add("klee_warning", handleWarning, false),
  add("klee_warning_once", handleWarningOnce, false),
  add("klee_alias_function", handleAliasFunction, false),
  add("malloc", handleMalloc, true),
  add("realloc", handleRealloc, true),

  // operator delete[](void*)
  add("_ZdaPv", handleDeleteArray, false),
  // operator delete(void*)
  add("_ZdlPv", handleDelete, false),

  // operator new[](unsigned int)
  add("_Znaj", handleNewArray, true),
  // operator new(unsigned int)
  add("_Znwj", handleNew, true),

  // FIXME-64: This is wrong for 64-bit long...

  // operator new[](unsigned long)
  add("_Znam", handleNewArray, true),
  // operator new(unsigned long)
  add("_Znwm", handleNew, true),

  // concurrent
  add("__set_CUDAConfig", handleSetCUDAConfiguration, false),
  add("__set_device", handleSetDevice, false),
  add("__clear_device", handleClearDevice, false),
  add("__set_host", handleSetHost, false),
  add("__clear_host", handleClearHost, false),

#undef addDNR
#undef add  
};

SpecialFunctionHandler::SpecialFunctionHandler(Executor &_executor) 
  : executor(_executor) {}


void SpecialFunctionHandler::prepare() {
  unsigned N = sizeof(handlerInfo)/sizeof(handlerInfo[0]);

  for (unsigned i=0; i<N; ++i) {
    HandlerInfo &hi = handlerInfo[i];
    Function *f = executor.kmodule->module->getFunction(hi.name);
    
    // No need to create if the function doesn't exist, since it cannot
    // be called in that case.
  
    if (f && (!hi.doNotOverride || f->isDeclaration())) {
      // Make sure NoReturn attribute is set, for optimization and
      // coverage counting.
      if (hi.doesNotReturn)
        f->addFnAttr(Attributes::NoReturn);

      // Change to a declaration since we handle internally (simplifies
      // module and allows deleting dead code).
      if (!f->isDeclaration())
        f->deleteBody();
    }
  }
}

void SpecialFunctionHandler::bind() {
  unsigned N = sizeof(handlerInfo)/sizeof(handlerInfo[0]);

  for (unsigned i=0; i<N; ++i) {
    HandlerInfo &hi = handlerInfo[i];
    Function *f = executor.kmodule->module->getFunction(hi.name);
    
    if (f && (!hi.doNotOverride || f->isDeclaration()))
      handlers[f] = std::make_pair(hi.handler, hi.hasReturnValue);
  }
}


bool SpecialFunctionHandler::handle(ExecutionState &state, 
                                    Function *f,
                                    KInstruction *target,
                                    std::vector< klee::ref<Expr> > &arguments) {

  handlers_ty::iterator it = handlers.find(f);
  if (it != handlers.end()) {    
    Handler h = it->second.first;
    bool hasReturnValue = it->second.second;
     // FIXME: Check this... add test?
    if (!hasReturnValue && !target->inst->use_empty()) {
      executor.terminateStateOnExecError(state, 
                                         "expected return value from void special function");
    } else {
      (this->*h)(state, target, arguments);
    }
    return true;
  } else {
    return false;
  }
}

/****/

// reads a concrete string from memory
std::string 
SpecialFunctionHandler::readStringAtAddress(ExecutionState &state, 
                                            klee::ref<Expr> addressExpr) {
  ObjectPair op;
  addressExpr = executor.toUnique(state, addressExpr);
  klee::ref<ConstantExpr> address = cast<ConstantExpr>(addressExpr);
  if (!state.addressSpace.resolveOne(address, op, GPUConfig::HOST))   // constant strings are put in the HOST memory 
    assert(0 && "XXX out of bounds / multiple resolution unhandled");
  bool res;
  bool success = executor.solver->mustBeTrue(state,
                                     EqExpr::create(address,
                                                    op.first->getBaseExpr()),
                                     res); 
  if (!success || !res) // Avoid the silly warning
    assert(success && res && "XXX interior pointer unhandled");
  const MemoryObject *mo = op.first;
  const ObjectState *os = op.second;

  char *buf = new char[mo->size];

  unsigned i;
  for (i = 0; i < mo->size - 1; i++) {
    klee::ref<Expr> cur = os->read8(i);
    cur = executor.toUnique(state, cur);
    assert(isa<ConstantExpr>(cur) && 
           "hit symbolic char while reading concrete string");
    buf[i] = cast<ConstantExpr>(cur)->getZExtValue(8);
  }
  buf[i] = 0;
  
  std::string result(buf);
  delete[] buf;
  return result;
}

/****/

void SpecialFunctionHandler::handleAbort(ExecutionState &state,
                           KInstruction *target,
                           std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==0 && "invalid number of arguments to abort");

  //XXX:DRE:TAINT
  if(state.underConstrained) {
    std::cerr << "TAINT: skipping abort fail\n";
    executor.terminateState(state);
  } else {
    executor.terminateStateOnError(state, "abort failure", "abort.err");
  }
}

void SpecialFunctionHandler::handleExit(ExecutionState &state,
                           KInstruction *target,
                           std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to exit");
  executor.terminateStateOnExit(state);
}

void SpecialFunctionHandler::handleSilentExit(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to exit");
  executor.terminateState(state);
}

void SpecialFunctionHandler::handleAliasFunction(ExecutionState &state,
						 KInstruction *target,
						 std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==2 && 
         "invalid number of arguments to klee_alias_function");
  std::string old_fn = readStringAtAddress(state, arguments[0]);
  std::string new_fn = readStringAtAddress(state, arguments[1]);
  //std::cerr << "Replacing " << old_fn << "() with " << new_fn << "()\n";
  if (old_fn == new_fn)
    state.removeFnAlias(old_fn);
  else state.addFnAlias(old_fn, new_fn);
}

void SpecialFunctionHandler::handleAssert(ExecutionState &state,
                                          KInstruction *target,
                                          std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==3 && "invalid number of arguments to _assert");  
  
  //XXX:DRE:TAINT
  if(state.underConstrained) {
    std::cerr << "TAINT: skipping assertion:" 
               << readStringAtAddress(state, arguments[0]) << "\n";
    executor.terminateState(state);
  } else
    executor.terminateStateOnError(state, 
                                   "ASSERTION FAIL: " + readStringAtAddress(state, arguments[0]),
                                   "assert.err");
}

void SpecialFunctionHandler::handleAssertFail(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==4 && "invalid number of arguments to __assert_fail");
  if (Emacs){
    if(MDNode *N = target->inst->getMetadata("dbg")){
      DILocation Loc(N); 
      std::cout << "emacs:assert:" << state.tinfo.get_cur_bid() << ":" <<
	state.tinfo.get_cur_tid() << ":" << Loc.getDirectory().str() << 
	"/" << Loc.getFilename().str() << ":" <<  Loc.getLineNumber() <<"::::"<< std::endl;
    }
  }
  
  //XXX:DRE:TAINT
  if(state.underConstrained) {
    std::cerr << "TAINT: skipping assertion:" 
               << readStringAtAddress(state, arguments[0]) << "\n";
    executor.terminateState(state);
  } else
    executor.terminateStateOnError(state, 
                                   "ASSERTION FAIL: " + readStringAtAddress(state, arguments[0]),
                                   "assert.err");
}

void SpecialFunctionHandler::handleReportError(ExecutionState &state,
                                               KInstruction *target,
                                               std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==4 && "invalid number of arguments to klee_report_error");
  
  // arguments[0], arguments[1] are file, line
  
  //XXX:DRE:TAINT
  if(state.underConstrained) {
    std::cerr << "TAINT: skipping klee_report_error:"
               << readStringAtAddress(state, arguments[2]) << ":"
               << readStringAtAddress(state, arguments[3]) << "\n";
    executor.terminateState(state);
  } else
    executor.terminateStateOnError(state, 
                                   readStringAtAddress(state, arguments[2]),
                                   readStringAtAddress(state, arguments[3]).c_str());
}

void SpecialFunctionHandler::handleMerge(ExecutionState &state,
                           KInstruction *target,
                           std::vector<klee::ref<Expr> > &arguments) {
  // nop
}

void SpecialFunctionHandler::handleNew(ExecutionState &state,
                         KInstruction *target,
                         std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to new");
  executor.executeAlloc(state, arguments[0], false, target);
}

void SpecialFunctionHandler::handleDelete(ExecutionState &state,
                            KInstruction *target,
                            std::vector<klee::ref<Expr> > &arguments) {
  // FIXME: Should check proper pairing with allocation type (malloc/free,
  // new/delete, new[]/delete[]).

  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to delete");
  if (arguments[0]->ctype == GPUConfig::UNKNOWN)
    executor.updateCType(state, 0, arguments[0], state.tinfo.is_GPU_mode);

  executor.executeFree(state, arguments[0]);
}

void SpecialFunctionHandler::handleNewArray(ExecutionState &state,
                              KInstruction *target,
                              std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to new[]");
  executor.executeAlloc(state, arguments[0], false, target);
}

void SpecialFunctionHandler::handleDeleteArray(ExecutionState &state,
                                 KInstruction *target,
                                 std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to delete[]");
  if (arguments[0]->ctype == GPUConfig::UNKNOWN)
    executor.updateCType(state, 0, arguments[0], state.tinfo.is_GPU_mode);

  executor.executeFree(state, arguments[0]);
}

void SpecialFunctionHandler::handleMalloc(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to malloc");
  executor.executeAlloc(state, arguments[0], false, target);
}

void SpecialFunctionHandler::handleAssume(ExecutionState &state,
                            KInstruction *target,
                            std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to klee_assume");
  
  klee::ref<Expr> e = arguments[0];
  
  if (e->getWidth() != Expr::Bool)
    e = NeExpr::create(e, ConstantExpr::create(0, e->getWidth()));
  
  bool res;
  bool success = executor.solver->mustBeFalse(state, e, res);
  if (!success)
    assert(success && "FIXME: Unhandled solver failure");
  if (res) {
    executor.terminateStateOnError(state, 
                                   "invalid klee_assume call (provably false)",
                                   "user.err");
  } else {
    executor.addConstraint(state, e);
  }
}

void SpecialFunctionHandler::handleIsSymbolic(ExecutionState &state,
                                KInstruction *target,
                                std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to klee_is_symbolic");

  executor.bindLocal(target, state, 
                     ConstantExpr::create(!isa<ConstantExpr>(arguments[0]),
                                          Expr::Int32));
}

void SpecialFunctionHandler::handlePreferCex(ExecutionState &state,
                                             KInstruction *target,
                                             std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_prefex_cex");

  klee::ref<Expr> cond = arguments[1];
  if (cond->getWidth() != Expr::Bool)
    cond = NeExpr::create(cond, ConstantExpr::alloc(0, cond->getWidth()));

  Executor::ExactResolutionList rl;
  executor.resolveExact(state, arguments[0], rl, "prefex_cex", GPUConfig::LOCAL);   // ????
  
  assert(rl.size() == 1 &&
         "prefer_cex target must resolve to precisely one object");

  rl[0].first.first->cexPreferences.push_back(cond);
}

void SpecialFunctionHandler::handlePrintExpr(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_print_expr");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  std::cerr << msg_str << ":" << arguments[1] << "\n";
}

void SpecialFunctionHandler::handleSetForking(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_set_forking");
  klee::ref<Expr> value = executor.toUnique(state, arguments[0]);
  
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(value)) {
    state.forkDisabled = CE->isZero();
  } else {
    executor.terminateStateOnError(state, 
                                   "klee_set_forking requires a constant arg",
                                   "user.err");
  }
}

void SpecialFunctionHandler::handleStackTrace(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<klee::ref<Expr> > &arguments) {
  state.dumpStack(std::cout);
}

void SpecialFunctionHandler::handleWarning(ExecutionState &state,
                                           KInstruction *target,
                                           std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to klee_warning");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  klee_warning("%s: %s", state.stacks.front().back().kf->function->getName().data(), 
               msg_str.c_str());
}

void SpecialFunctionHandler::handleWarningOnce(ExecutionState &state,
                                               KInstruction *target,
                                               std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_warning_once");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  klee_warning_once(0, "%s: %s", state.stacks.front().back().kf->function->getName().data(),
                    msg_str.c_str());
}

void SpecialFunctionHandler::handlePrintRange(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_print_range");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  std::cerr << msg_str << ":" << arguments[1];
  if (!isa<ConstantExpr>(arguments[1])) {
    // FIXME: Pull into a unique value method?
    klee::ref<ConstantExpr> value;
    bool success = executor.solver->getValue(state, arguments[1], value);
    if (!success)
      assert(success && "FIXME: Unhandled solver failure");
    bool res;
    success = executor.solver->mustBeTrue(state, 
                                          EqExpr::create(arguments[1], value), 
                                          res);
    if (!success)
      assert(success && "FIXME: Unhandled solver failure");
    if (res) {
      std::cerr << " == " << value;
    } else { 
      std::cerr << " ~= " << value;
      std::pair< klee::ref<Expr>, klee::ref<Expr> > res =
        executor.solver->getRange(state, arguments[1]);
      std::cerr << " (in [" << res.first << ", " << res.second <<"])";
    }
  }
  std::cerr << "\n";
}

void SpecialFunctionHandler::handleGetObjSize(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_get_obj_size");
  Executor::ExactResolutionList rl;
  if (arguments[0]->ctype == GPUConfig::UNKNOWN)
    executor.updateCType(state, 0, arguments[0], state.tinfo.is_GPU_mode);

  unsigned b_t_index = arguments[0]->ctype == GPUConfig::LOCAL ? 
                           state.tinfo.get_cur_tid() : state.tinfo.get_cur_bid();
  executor.resolveExact(state, arguments[0], rl, "klee_get_obj_size", arguments[0]->ctype, b_t_index);
  for (Executor::ExactResolutionList::iterator it = rl.begin(), 
         ie = rl.end(); it != ie; ++it) {
    executor.bindLocal(target, *it->second, 
                       ConstantExpr::create(it->first.first->size, Expr::Int32));
  }
}

void SpecialFunctionHandler::handleGetErrno(ExecutionState &state,
                                            KInstruction *target,
                                            std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==0 &&
         "invalid number of arguments to klee_get_obj_size");
  executor.bindLocal(target, state,
                     ConstantExpr::create(errno, Expr::Int32));
}

void SpecialFunctionHandler::handleCalloc(ExecutionState &state,
                            KInstruction *target,
                            std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==2 &&
         "invalid number of arguments to calloc");

  klee::ref<Expr> size = MulExpr::create(arguments[0],
                                   arguments[1]);
  executor.executeAlloc(state, size, false, target, true);
}

void SpecialFunctionHandler::handleRealloc(ExecutionState &state,
                            KInstruction *target,
                            std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==2 &&
         "invalid number of arguments to realloc");
  klee::ref<Expr> address = arguments[0];
  klee::ref<Expr> size = arguments[1];

  Executor::StatePair zeroSize = executor.fork(state, 
                                               Expr::createIsZero(size), 
                                               true);
  
  if (zeroSize.first) { // size == 0
    executor.executeFree(*zeroSize.first, address, target);   
  }
  if (zeroSize.second) { // size != 0
    Executor::StatePair zeroPointer = executor.fork(*zeroSize.second, 
                                                    Expr::createIsZero(address), 
                                                    true);
    
    if (zeroPointer.first) { // address == 0
      executor.executeAlloc(*zeroPointer.first, size, false, target);
    } 
    if (zeroPointer.second) { // address != 0
      Executor::ExactResolutionList rl;
      executor.resolveExact(*zeroPointer.second, address, rl, "realloc",
			    GPUConfig::HOST);
      
      for (Executor::ExactResolutionList::iterator it = rl.begin(), 
             ie = rl.end(); it != ie; ++it) {
        executor.executeAlloc(*it->second, size, false, target, false, 
                              it->first.second);
      }
    }
  }
}

void SpecialFunctionHandler::handleFree(ExecutionState &state,
                                        KInstruction *target,
                                        std::vector<klee::ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 &&
         "invalid number of arguments to free");
  if (arguments[0]->ctype == GPUConfig::UNKNOWN)
    executor.updateCType(state, 0, arguments[0], state.tinfo.is_GPU_mode);

  executor.executeFree(state, arguments[0]);
}

void SpecialFunctionHandler::handleCheckMemoryAccess(ExecutionState &state,
                                                     KInstruction *target,
                                                     std::vector<klee::ref<Expr> > 
                                                       &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_check_memory_access");

  klee::ref<Expr> address = executor.toUnique(state, arguments[0]);
  klee::ref<Expr> size = executor.toUnique(state, arguments[1]);
  if (!isa<ConstantExpr>(address) || !isa<ConstantExpr>(size)) {
    executor.terminateStateOnError(state, 
                                   "check_memory_access requires constant args",
                                   "user.err");
  } else {
    ObjectPair op;

    if (!state.addressSpace.resolveOne(cast<ConstantExpr>(address), op, GPUConfig::HOST)) {
      executor.terminateStateOnError(state,
                                     "check_memory_access: memory error",
                                     "ptr.err",
                                     executor.getAddressInfo(state, address));
    } else {
      klee::ref<Expr> chk = 
        op.first->getBoundsCheckPointer(address, 
                                        cast<ConstantExpr>(size)->getZExtValue());
      if (!chk->isTrue()) {
        executor.terminateStateOnError(state,
                                       "check_memory_access: memory error",
                                       "ptr.err",
                                       executor.getAddressInfo(state, address));
      }
    }
  }
}

void SpecialFunctionHandler::handleGetValue(ExecutionState &state,
                                            KInstruction *target,
                                            std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_get_value");

  executor.executeGetValue(state, arguments[0], target);
}

void SpecialFunctionHandler::handleDefineFixedObject(ExecutionState &state,
                                                     KInstruction *target,
                                                     std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_define_fixed_object");
  assert(isa<ConstantExpr>(arguments[0]) &&
         "expect constant address argument to klee_define_fixed_object");
  assert(isa<ConstantExpr>(arguments[1]) &&
         "expect constant size argument to klee_define_fixed_object");
  
  uint64_t address = cast<ConstantExpr>(arguments[0])->getZExtValue();
  uint64_t size = cast<ConstantExpr>(arguments[1])->getZExtValue();
  MemoryObject *mo = executor.memory->allocateFixed(address, size, executor.getGPUMode(), 
                                                    state.getPrevPC()->inst);
  executor.bindObjectInState(state, mo, false);
  mo->isUserSpecified = true; // XXX hack;
}

void SpecialFunctionHandler::handleMakeSymbolic(ExecutionState &state,
                                                KInstruction *target,
                                                std::vector<klee::ref<Expr> > &arguments) {

  // need to execute this function only once
  if (state.tinfo.get_cur_tid() != 0)
    return;

  std::string name;

  // FIXME: For backwards compatibility, we should eventually enforce the
  // correct arguments.
  if (arguments.size() == 2) {
    name = "unnamed";
  } else {
    // FIXME: Should be a user.err, not an assert.
    assert(arguments.size()==3 &&
           "invalid number of arguments to klee_make_symbolic");  
    name = readStringAtAddress(state, arguments[2]);
  }

  Executor::ExactResolutionList rl;
  // seems using the value v to infer the ctype
  // is not very precise, I changed the code here...

  if (arguments[0]->ctype == GPUConfig::UNKNOWN)
    executor.updateCType(state, 0, arguments[0], state.tinfo.is_GPU_mode);
  //std::cout << "arguments[0] ctype: " 
  //          << CUDAUtil::getCTypeStr(arguments[0]->ctype)
  //          << std::endl;
  unsigned b_t_index = arguments[0]->ctype == GPUConfig::LOCAL ? state.tinfo.get_cur_tid() : state.tinfo.get_cur_bid();
  executor.resolveExact(state, arguments[0], rl, "make_symbolic",
	        	arguments[0]->ctype, b_t_index);
  assert(rl.size() > 0 && "Fail to resolve the variable in make_symbolic");
  
  for (Executor::ExactResolutionList::iterator it = rl.begin(), 
         ie = rl.end(); it != ie; ++it) {
    const MemoryObject *mo = it->first.first;
    mo->setName(name);
    
    const ObjectState *old = it->first.second;
    ExecutionState *s = it->second;
    
    if (old->readOnly) {
      executor.terminateStateOnError(*s,
                                     "cannot make readonly object symbolic", 
                                     "user.err");
      return;
    }
    
    // FIXME: Type coercion should be done consistently somewhere.
    bool res;
    bool success =
      executor.solver->mustBeTrue(*s,
         EqExpr::create(ZExtExpr::create(arguments[1],
					 Context::get().getPointerWidth()),
			mo->getSizeExpr()),
                                  res);
    if (!success) 
      assert(success && "FIXME: Unhandled solver failure");
    
    if (res) {
      executor.executeMakeSymbolic(*s, mo, name);
    } else {
      executor.terminateStateOnError(*s, 
                                     "wrong size given to klee_make_symbolic[_name]", 
                                     "user.err");
    }
  }
}

void SpecialFunctionHandler::handleMarkGlobal(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<klee::ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_mark_global");  

  Executor::ExactResolutionList rl;
  executor.resolveExact(state, arguments[0], rl, "mark_global", GPUConfig::HOST);
  
  for (Executor::ExactResolutionList::iterator it = rl.begin(), 
         ie = rl.end(); it != ie; ++it) {
    const MemoryObject *mo = it->first.first;
    assert(!mo->isLocal);
    mo->isGlobal = true;
  }
}

void SpecialFunctionHandler::handleSetCUDAConfiguration(ExecutionState &state, 
                                                        KInstruction *target, 
                                                        std::vector< klee::ref<Expr> > &arguments) {
  // Note: cudaConfigureCall is based on CUDA 4.2, this handler might  
  // be evolved with respect to CUDA 5.0 
  if (state.tinfo.get_cur_tid() != 0) return;  
  // only thread 0 does the modification job
  
  executor.initializeMissedBuiltInVariables(state);
 
  // handle the block size specification
  for (unsigned i = 0; i < 4; i++) {
    klee::ref<Expr> gdim = arguments[i];
    assert(isa<ConstantExpr>(arguments[i]) && "arguments in __set_CUDAConfig is not constant");

    if (i == 0) {
      uint64_t v = dyn_cast<ConstantExpr>(arguments[0])->getZExtValue();
      // the first dimension of grid
      unsigned v1 = (unsigned)v;
      GPUConfig::GridSize[0] = v1;
      if (UseSymbolicConfig)
        GPUConfig::SymGridSize[0] = v1;
      // the second dimension of grid
      unsigned v2 = (unsigned) (v >> 32);  
      GPUConfig::GridSize[1] = v2;
      if (UseSymbolicConfig)
        GPUConfig::SymGridSize[1] = v2;
    } else if (i == 1) {
      // the third dimension of grid
      unsigned v = dyn_cast<ConstantExpr>(arguments[1])->getZExtValue();
      GPUConfig::GridSize[2] = v;
      if (UseSymbolicConfig)
        GPUConfig::SymGridSize[2] = v;
    } else if (i == 2) {
      uint64_t v = dyn_cast<ConstantExpr>(arguments[2])->getZExtValue();
      // the first dimension of block
      unsigned v1 = (unsigned)v;
      GPUConfig::BlockSize[0] = v1;
      if (UseSymbolicConfig)
        GPUConfig::SymBlockSize[0] = v1;
      
      // the second dimension of block
      unsigned v2 = (unsigned) (v >> 32);  
      GPUConfig::BlockSize[1] = v2;
      if (UseSymbolicConfig)
        GPUConfig::SymBlockSize[1] = v2;
    } else {
      // the third dimension of block
      unsigned v = dyn_cast<ConstantExpr>(arguments[3])->getZExtValue();
      GPUConfig::BlockSize[2] = v;
      if (UseSymbolicConfig)
        GPUConfig::SymBlockSize[2] = v;
    }
  }
  GPUConfig::num_blocks = GPUConfig::GridSize[0] * GPUConfig::GridSize[1] * GPUConfig::GridSize[2]; 
  GPUConfig::block_size = GPUConfig::BlockSize[0] * GPUConfig::BlockSize[1] * GPUConfig::BlockSize[2];
  GPUConfig::num_threads = GPUConfig::block_size * GPUConfig::num_blocks;

  GKLEE_INFO2 << "The configuration, Grid: <" << GPUConfig::GridSize[0] 
              << ", " << GPUConfig::GridSize[1] << ", " << GPUConfig::GridSize[2]
              << ">" << ", Block: <" << GPUConfig::BlockSize[0] << ", "
              << GPUConfig::BlockSize[1] << ", " << GPUConfig::BlockSize[2] 
              << ">" << std::endl; 

  // Update the # of blocks
  if (UseSymbolicConfig) {
    GPUConfig::sym_num_blocks = GPUConfig::SymGridSize[0] * GPUConfig::SymGridSize[1] * GPUConfig::SymGridSize[2];
    if (GPUConfig::sym_num_blocks == 1) {
      GPUConfig::SymGridSize[0] = 2;
      GPUConfig::sym_num_blocks = 2;
    }
    GPUConfig::sym_block_size = GPUConfig::SymBlockSize[0] * GPUConfig::SymBlockSize[1] * GPUConfig::SymBlockSize[2];
    GPUConfig::sym_num_threads = GPUConfig::sym_block_size * GPUConfig::sym_num_blocks;
    state.reconfigGPUSymbolic();
  } else {
    state.reconfigGPU();
  }

  // Handle the other arguments ...
  if (arguments.size() > 4)
    state.maxKernelSharedSize = dyn_cast<ConstantExpr>(arguments[4])->getZExtValue();   

  state.tinfo.kernel_call = true;
  // clear address sets
  state.addressSpace.clearAccessSet();
  state.addressSpace.clearInstAccessSet(true);
}

void SpecialFunctionHandler::handleSetDevice(ExecutionState &state, 
                                             KInstruction *target,
                                             std::vector<klee::ref<Expr> > &arguments) {
  state.deviceSet = 1;
}

void SpecialFunctionHandler::handleClearDevice(ExecutionState &state, 
                                               KInstruction *target,
                                               std::vector<klee::ref<Expr> > &arguments) {
  state.deviceSet = 0;
}

void SpecialFunctionHandler::handleSetHost(ExecutionState &state, 
                                           KInstruction *target,
                                           std::vector<klee::ref<Expr> > &arguments) {
  state.deviceSet = 2;
}

void SpecialFunctionHandler::handleClearHost(ExecutionState &state, 
                                             KInstruction *target,
                                             std::vector<klee::ref<Expr> > &arguments) {
  state.deviceSet = 0;
}
