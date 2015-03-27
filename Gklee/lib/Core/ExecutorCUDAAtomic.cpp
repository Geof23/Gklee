//===-- ExecutorCUDAAtomic.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Executor.h"
#include "CUDA.h"
#include "CUDAIntrinsics.h"
#include "TimingSolver.h"

#include "llvm/Support/CommandLine.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/ExecutionState.h"
#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
#include "llvm/Analysis/DebugInfo.h"
#else
#include "llvm/DebugInfo.h"
#endif
#include <math.h>

using namespace llvm;
using namespace klee;

static inline const llvm::fltSemantics * fpWidthToSemantics(unsigned width) {
  switch(width) {
  case Expr::Int16:
    return &llvm::APFloat::IEEEhalf;
  case Expr::Int32:
    return &llvm::APFloat::IEEEsingle;
  case Expr::Int64:
    return &llvm::APFloat::IEEEdouble;
  case Expr::Fl80:
    return &llvm::APFloat::x87DoubleExtended;
  default:
    return 0;
  }
}

void Executor::executeAtomicAdd(ExecutionState &state, 
                                KInstruction *target, std::string fName, 
                                std::vector< klee::ref<Expr> > &arguments, 
                                unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     

  if (fName.find("fAtomicAdd") != std::string::npos) {
    klee::ref<ConstantExpr> va = toConstant(state, atomicRes, "Unsupported fAtomicAdd");
    klee::ref<ConstantExpr> vb = toConstant(state, arguments[1], "Unsupported fAtomicAdd");
     
    if (!fpWidthToSemantics(va->getWidth()) || 
        !fpWidthToSemantics(vb->getWidth()))
      return terminateStateOnExecError(state, "Unsupported fAtomicAdd operation");

    llvm::APFloat Res(va->getAPValue());
    Res.add(APFloat(vb->getAPValue()), APFloat::rmNearestTiesToEven);
    klee::ref<Expr> tmp = ConstantExpr::alloc(Res);

    // Store back to original place 
    executeMemoryOperation(state, true, base, ConstantExpr::alloc(Res), 
                           target, seqNum, true);
    bindLocal(target, state, atomicRes);
  } else {
    // Add the value
    klee::ref<Expr> sumExpr = AddExpr::create(atomicRes, arguments[1]); 
    // Store back to original place 
    executeMemoryOperation(state, true, base, sumExpr, 
                           target, seqNum, true);
    bindLocal(target, state, atomicRes);
  }

  return;
}

void Executor::executeAtomicExch(ExecutionState &state, KInstruction *target, 
                                 std::vector< klee::ref<Expr> > &arguments, 
                                 unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     

  // Store back to original place 
  executeMemoryOperation(state, true, base, arguments[1], 
                         target, seqNum, true);

  bindLocal(target, state, atomicRes);
  return;
}

void Executor::compareValue(ExecutionState &state, 
                            KInstruction *target, std::string fName, 
                            std::vector< klee::ref<Expr> > &arguments,
                            unsigned seqNum, klee::ref<Expr> base, bool isMin) {
  klee::ref<Expr> compCond;
  if (isMin) {
    if (fName.find("uAtomicMin") != std::string::npos)
      compCond = UleExpr::create(atomicRes, arguments[1]);
    else 
      compCond = SleExpr::create(atomicRes, arguments[1]);
  } else {
    if (fName.find("uAtomicMax") != std::string::npos)
      compCond = UgeExpr::create(atomicRes, arguments[1]);
    else 
      compCond = SgeExpr::create(atomicRes, arguments[1]);
  }

  state.tinfo.is_Atomic_op = 2;
  Executor::StatePair branches = fork(state, compCond, true); 
  if (branches.first) {
    bindLocal(target, *branches.first, atomicRes);
  }
  if (branches.second) {
    executeMemoryOperation(*branches.second, true, base, arguments[1], 
                           target, seqNum, true);
    bindLocal(target, *branches.second, atomicRes);
  } 
  state.tinfo.is_Atomic_op = 0;
}
 
void Executor::executeAtomicMin(ExecutionState &state, 
                                KInstruction *target, std::string fName, 
                                std::vector< klee::ref<Expr> > &arguments, 
                                unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     
  // compare and store the value
  compareValue(state, target, fName, arguments, seqNum, base, true);
}

void Executor::executeAtomicMax(ExecutionState &state, 
                                KInstruction *target, std::string fName, 
                                std::vector< klee::ref<Expr> > &arguments, 
                                unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     
  // compare and store the value
  compareValue(state, target, fName, arguments, seqNum, base, false);
}

void Executor::executeAtomicInc(ExecutionState &state, KInstruction *target, 
                                std::vector< klee::ref<Expr> > &arguments, 
                                unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     

  // old >= val 
  klee::ref<Expr> compCond = UgeExpr::create(atomicRes, arguments[1]);
  klee::ref<Expr> zeroExpr = klee::ConstantExpr::create(0, arguments[1]->getWidth());
  klee::ref<Expr> oneExpr = klee::ConstantExpr::create(1, arguments[1]->getWidth());

  Executor::StatePair branches = fork(state, compCond, true); 
  if (branches.first) {
    executeMemoryOperation(*branches.first, true, base, zeroExpr, 
                           target, seqNum, true);
    bindLocal(target, *branches.first, atomicRes);
  }
  if (branches.second) {
    executeMemoryOperation(*branches.second, true, base,
                           AddExpr::create(atomicRes, oneExpr), 
                           target, seqNum, true);
    bindLocal(target, *branches.second, atomicRes);
  } 
}

void Executor::executeAtomicDec(ExecutionState &state, 
                                KInstruction *target, 
                                std::vector< klee::ref<Expr> > &arguments, 
                                unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     

  // (old == 0) | (old > val) 
  klee::ref<Expr> zeroExpr = klee::ConstantExpr::create(0, arguments[1]->getWidth());
  klee::ref<Expr> oneExpr = klee::ConstantExpr::create(1, arguments[1]->getWidth());
  klee::ref<Expr> compCond = OrExpr::create(EqExpr::create(atomicRes, zeroExpr),
                                      UgtExpr::create(atomicRes, arguments[1]));

  Executor::StatePair branches = fork(state, compCond, true); 
  if (branches.first) {
    executeMemoryOperation(*branches.first, true, base, arguments[1], 
                           target, seqNum, true);
    bindLocal(target, *branches.first, atomicRes);
  } 
  if (branches.second) {
    executeMemoryOperation(*branches.second, true, base, 
                           SubExpr::create(atomicRes, oneExpr), 
                           target, seqNum, true);
    bindLocal(target, *branches.second, atomicRes);
  } 
}

void Executor::executeAtomicCAS(ExecutionState &state, KInstruction *target, 
                                std::vector< klee::ref<Expr> > &arguments, 
                                unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     
   
  // old == compare 
  klee::ref<Expr> compCond = EqExpr::create(atomicRes, arguments[1]);

  state.tinfo.is_Atomic_op = 1;
  Executor::StatePair branches = fork(state, compCond, true); 
  if (branches.first) {
    executeMemoryOperation(*branches.first, true, base, arguments[2], 
                           target, seqNum, true);
    bindLocal(target, *branches.first, atomicRes);
  }
  if (branches.second) {
    //executeMemoryOperation(*branches.second, true, base, atomicRes, 
    //                       target, seqNum, true, true);
    bindLocal(target, *branches.second, atomicRes);
  }
  state.tinfo.is_Atomic_op = 0;
}

void Executor::executeAtomicBitWise(ExecutionState &state, 
                                    KInstruction *target, std::string fName,
                                    std::vector< klee::ref<Expr> > &arguments, 
                                    unsigned seqNum) {
  // Load the value from addr
  klee::ref<Expr> base = arguments[0];
  CallInst *ci = static_cast<CallInst*>(target->inst);     

  updateCType(state, ci->getArgOperand(0), base, state.tinfo.is_GPU_mode);
  executeMemoryOperation(state, false, base, 0, 
                         target, seqNum, true);     

  klee::ref<Expr> Res;
  if (fName.find("And") != std::string::npos)
    Res = AndExpr::create(atomicRes, arguments[1]);
  else if (fName.find("Or") != std::string::npos)
    Res = OrExpr::create(atomicRes, arguments[1]);
  else 
    Res = XorExpr::create(atomicRes, arguments[1]);

  executeMemoryOperation(state, true, base, atomicRes, 
                         target, seqNum, true);
  bindLocal(target, state, atomicRes);
}

bool Executor::executeCUDAAtomic(ExecutionState &state,
                                 KInstruction *target, std::string fName,
                                 std::vector< klee::ref<Expr> > &arguments, 
                                 unsigned seqNum) {
  bool intrinsicFound = true;

  if (fName.find("AtomicAdd") != std::string::npos) {
    executeAtomicAdd(state, target, fName, arguments, seqNum);
  } else if (fName.find("AtomicExch") != std::string::npos) {
    executeAtomicExch(state, target, arguments, seqNum);
  } else if (fName.find("AtomicMin") != std::string::npos) {
    executeAtomicMin(state, target, fName, arguments, seqNum);
  } else if (fName.find("AtomicMax") != std::string::npos) {
    executeAtomicMax(state, target, fName, arguments, seqNum);
  } else if (fName.find("AtomicInc") != std::string::npos) {
    executeAtomicInc(state, target, arguments, seqNum);
  } else if (fName.find("AtomicDec") != std::string::npos) {
    executeAtomicDec(state, target, arguments, seqNum);
  } else if (fName.find("AtomicCAS") != std::string::npos) {
    executeAtomicCAS(state, target, arguments, seqNum);
  } else if (fName.find("AtomicAnd") != std::string::npos
              || fName.find("AtomicOr") != std::string::npos
                || fName.find("AtomicXor") != std::string::npos) {
    executeAtomicBitWise(state, target, fName, arguments, seqNum);
  } else intrinsicFound = false;

  return intrinsicFound;
}
