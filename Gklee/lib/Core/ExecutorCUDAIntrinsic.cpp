//===-- ExecutorCUDAIntrinsic.cpp -----------------------------------------===//
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
using namespace Gklee;

#define PI 3.1415926535897932384626433832795

namespace runtime {
  cl::opt<bool>
  SimdSchedule("simd-schedule",
          cl::desc("SIMD aware canonical scheduling"),
          cl::init(false));

  cl::opt<bool>
  CheckBarrierRedundant("check-barrier-redundant",
	  cl::desc("Check whether this barrier is redundant or not"),
	  cl::init(false));

  cl::opt<bool>
  CheckBC("check-BC",
	  cl::desc("Check bank conflicts"),
	  cl::init(true));

  cl::opt<bool>
  CheckMC("check-MC",
	  cl::desc("Check whether the global memory accesses can be coalesced"),
	  cl::init(true));

  cl::opt<bool>
  CheckWD("check-WD",
	  cl::desc("Check whether there exists warp divergence"),
	  cl::init(true));

  cl::opt<bool>
  CheckVolatile("check-volatile",
	  cl::desc("Check whether volatile keywork is missing"),
	  cl::init(true));

  cl::opt<unsigned>
  DevCap("device-capability",
           cl::desc("Set device capability, (0): 1.0 and 1.1; (1), 1.2 and 1.3; (2), 2.x"),
           cl::init(2));

  cl::opt<bool>
  IgnoreConcurBug("ignore-concur-bug",
		  cl::desc("Continue execution even a concurrency bug is encountered"),
		  cl::init(false));

  extern cl::opt<bool> Emacs;
}

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

void Executor::encounterBarrier(ExecutionState &state,
	   		        KInstruction *target,
				bool is_end_GPU_barrier, 
                                bool &allThreadsBarrier) {
  Gklee::Logging::enterFunc< std::string >( std::string( "current thread: " ) + 
					    std::to_string( state.tinfo.get_cur_tid()),
					    __PRETTY_FUNCTION__ );
  if (GPUConfig::verbose > 0 || UseSymbolicConfig)
    std::cout << "Thread " << state.tinfo.get_cur_tid()
	      << " reaches a barrier: moving to the next thread.\n";

  // increase the barrier count
  unsigned tid = state.tinfo.get_cur_tid();
  Logging::outItem( std::to_string( state.tinfo.get_cur_tid() ), 
		    "curTID" );
  std::vector<BarrierInfo> &barrierVec = state.tinfo.numBars[tid].first;
  barrierVec.push_back(BarrierInfo(target->inst, target->info->file, target->info->line));
  state.tinfo.numBars[tid].second = is_end_GPU_barrier;
  state.encounterSyncthreadsBarrier(tid);
  if (state.fence != "") state.fence = ""; 

  if (!UseSymbolicConfig)
    allThreadsBarrier = SimdSchedule ? state.allThreadsEncounterBarrier() : state.tinfo.at_last_tid();
  else 
    allThreadsBarrier = state.allSymbolicThreadsEncounterBarrier();
  Gklee::Logging::outItem< std::string >( std::to_string( allThreadsBarrier ), "all threads at barrier" );
  if (allThreadsBarrier) {
    unsigned BINum = state.tinfo.numBars[tid].first.size();

    // if all threads in the last warp encounter __syncthreads(),
    // then start checking races, bc, wd, mc... 
    if (!UseSymbolicConfig) {
      state.addressSpace.ensureThreadsDivergenceRegion(state.cTidSets);
      state.addressSpace.constructGlobalMemAccessSets(*this, state, state.cTidSets, BINum);

      if (state.tinfo.hasMismatchBarrier(state.cTidSets)) {
        std::cout << "Found a deadlock: #barriers at the threads:\n";
        for (unsigned i = 0; i < GPUConfig::num_threads; i++)
          std::cout << "t" << i << ":" << state.tinfo.numBars[i].first.size() << " ";
        std::cout << std::endl;

        terminateStateOnExecErrorPublic(state, "execution halts on a barrier mismatch");
        if (Emacs) {
          if (MDNode *N = target->inst->getMetadata("dbg")) {
            DILocation Loc(N); 
	    std::cout << "emacs:dlbm:" << state.tinfo.get_cur_bid() << ":" 
                      << state.tinfo.get_cur_tid() << ":" 
                      << Loc.getDirectory().str() << "/" 
                      << Loc.getFilename().str() << ":" 
                      << Loc.getLineNumber() <<"::::"<< std::endl;
	  } else {
	    std::cout << "emacs:dlbm, location information unavailable (inst->getMetadata returned null)" << std::endl;
	  }
	}
      }
    } else { //symbolic config
      // Check if the memory access is thread parametric ...
      if (state.addressSpace.hasMismatchBarrierInParametricFlow(*this, state)) {
        std::cout << "Found a deadlock: #barriers at the flows:\n";
        for (unsigned i = 0; i < state.cTidSets.size(); i++) {
          if (i != 1) {
            if (state.cTidSets[i].slotUsed)
              std::cout << "t" << i << ":" << state.tinfo.numBars[i].first.size() << " ";
            else 
              break;
          }
        }
        std::cout << std::endl;

        terminateStateOnExecErrorPublic(state, "execution halts on a barrier mismatch in parametric flow");
      }
    }

    if (CheckBC) {
      if (!UseSymbolicConfig)
        state.addressSpace.hasBankConflict(*this, state, state.cTidSets, DevCap);
      else {
        bool hasBC = state.addressSpace.hasSymBankConflict(*this, state, DevCap);
        if (hasBC) symBC = true;
      }
    }

    if (CheckMC) {
      if (!UseSymbolicConfig)
        state.addressSpace.hasMemoryCoalescing(*this, state, state.cTidSets, DevCap);
      else { 
        bool hasMC = state.addressSpace.hasSymMemoryCoalescing(*this, state, DevCap);
        if (!hasMC) symMC = false;
      }
    }

    if (CheckWD) {
      if (!UseSymbolicConfig)
        state.addressSpace.hasWarpDivergence(state.cTidSets);
      else {
        bool hasWD = state.addressSpace.hasSymWarpDivergence(state);
        if (hasWD) symWD = true;
      }
    }

    if (CheckVolatile) {
      if (!UseSymbolicConfig)
        state.addressSpace.hasVolatileMissing(*this, state, state.cTidSets);
      else { 
        bool hasVMiss = state.addressSpace.hasSymVolatileMissing(*this, state);
        if (hasVMiss) symVMiss = true;
      }
    }

    if (!UseSymbolicConfig) {
      // check races on shared memory 
      klee::ref<Expr> shareRaceCond = klee::ConstantExpr::create(1, Expr::Bool);
      if (state.addressSpace.hasRaceInShare(*this, state, state.cTidSets, shareRaceCond)) {
        terminateStateOnExecError(state, "execution halts on encounering a (shared) race");
      }
    } else {
      bool hasRace = state.addressSpace.hasSymRaceInShare(*this, state);
      if (hasRace) {
        symRace = true;
        terminateStateOnExecError(state, "execution halts on encounering a (shared) race");
      }
    }

    if (!UseSymbolicConfig) {
      // check races on the device and CPU memory
      klee::ref<Expr> globalRaceCond = klee::ConstantExpr::create(1, Expr::Bool);
      if (state.addressSpace.hasRaceInGlobal(*this, state, state.cTidSets, globalRaceCond, BINum, is_end_GPU_barrier)) {
        terminateStateOnExecError(state, "execution halts on encounering a (global) race");
      }
    } else {
      bool hasRace = state.addressSpace.hasSymRaceInGlobal(*this, state, is_end_GPU_barrier);
      if (hasRace) {
        symRace = true;
        terminateStateOnExecError(state, "execution halts on encounering a (global) race");
      }
    }

    if (!is_end_GPU_barrier) {
      bc_cov_monitor.atBarrier(state.getKernelNum(), BINum);
    }
      
    if (UseSymbolicConfig) {
      unsigned num = 0;
      for (unsigned i = 0; i < state.cTidSets.size(); i++) {
        if (state.cTidSets[i].slotUsed)
          num++;
      }
      GKLEE_INFO << "Within the current Barrier Interval, " << num-1
                 << " flows are used to represent all threads !" 
                 << std::endl;
    }
    state.addressSpace.clearAccessSet();
    state.addressSpace.clearInstAccessSet(true);
  }

  if (!UseSymbolicConfig) {
    if (!SimdSchedule) state.tinfo.incTid();
  }
  Gklee::Logging::exitFunc();
}

void Executor::handleBarrier(ExecutionState &state,
	   		     KInstruction *target) {
  Gklee::Logging::enterFunc( *target->inst, __PRETTY_FUNCTION__ );
  Logging::fgInfo( "encounterBarrier", *target->inst );
  bool allThreadsBarrier = false;
  encounterBarrier(state, target, false, allThreadsBarrier);
  Gklee::Logging::exitFunc();
}

void Executor::handleMemfence(ExecutionState &state, 
                              KInstruction *target) {
  std::string fName = target->inst->getName();
  if (fName.find("__threadfence_block") != std::string::npos)
    state.fence = "__threadfence_block";
  else if (fName.find("__threadfence_system") != std::string::npos)
    state.fence = "__threadfence_system";
  else 
    state.fence = "__threadfence";
}

void Executor::handleEndGPU(ExecutionState &state,
	  	            KInstruction *target) {
  bool allThreadsBarrier = false;
  encounterBarrier(state, target, true, allThreadsBarrier);

  if (allThreadsBarrier) {
    GKLEE_INFO2 << "Finish executing a GPU kernel \n";

    state.tinfo.allEndKernel = true;
    // report the time
    if (states.size() == 1) { // the last state
      // GPU execution time
      GKLEE_INFO2 << "GPU Execution time: " << state.tinfo.getGPUTime() << "s\n";
      // coverage per thread
      bc_cov_monitor.computePerThreadCoverage();
    }
  }
}

// __mul64hi, __umul64hi, __mul24hi, __umul24hi, __mulhi, __umulhi 
static void executeMulHiIntrinsic(Executor &executor, ExecutionState &state, 
                                  KInstruction *target, std::string fName, 
                                  std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<Expr> result;
  if (fName.find("64") != std::string::npos) { // __mul64hi, __umul64hi
    assert(arguments[0]->getWidth() == Expr::Int64 && "arguments 0's width is wrong");  
    assert(arguments[1]->getWidth() == Expr::Int64 && "arguments 1's width is wrong");  
    klee::ref<Expr> tmpA, tmpB;
    if (fName.find("umul64hi") != std::string::npos) {
      tmpA = ZExtExpr::create(arguments[0], 128); 
      tmpB = ZExtExpr::create(arguments[1], 128); 
    } else {
      tmpA = SExtExpr::create(arguments[0], 128); 
      tmpB = SExtExpr::create(arguments[1], 128); 
    } 
    result = MulExpr::create(tmpA, tmpB);
    result = LShrExpr::create(result, klee::ConstantExpr::create(64, result->getWidth()));
    result = ExtractExpr::create(result, 0, 64);
  } else if (fName.find("24") != std::string::npos) { // __mul24hi, __umul24hi
    assert(arguments[0]->getWidth() == Expr::Int32 && "arguments 0's width is wrong");  
    assert(arguments[1]->getWidth() == Expr::Int32 && "arguments 1's width is wrong");  
    result = MulExpr::create(arguments[0], arguments[1]);
    result = ExtractExpr::create(result, 0, 32);
  } else { // __mulhi, __umulhi
    assert(arguments[0]->getWidth() == Expr::Int32 && "arguments 0's width is wrong");  
    assert(arguments[1]->getWidth() == Expr::Int32 && "arguments 1's width is wrong");  
    klee::ref<Expr> tmpA, tmpB;
    if (fName.find("umulhi") != std::string::npos) {
      tmpA = ZExtExpr::create(arguments[0], 64); 
      tmpB = ZExtExpr::create(arguments[1], 64); 
    } else {
      tmpA = SExtExpr::create(arguments[0], 64); 
      tmpB = SExtExpr::create(arguments[1], 64); 
    } 
    result = MulExpr::create(tmpA, tmpB);
    result = LShrExpr::create(result, klee::ConstantExpr::create(32, result->getWidth()));
    result = ExtractExpr::create(result, 0, 32);
  }

  executor.bindLocal(target, state, result);
  return;
}

// __saturatef
static void executeSaturateIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "__saturatef floating point");
  if (!fpWidthToSemantics(va->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported saturate operation");
  
  llvm::APFloat value(va->getAPValue());
  llvm::APFloat zero = APFloat::getZero(value.getSemantics());
  llvm::APFloat one(value.getSemantics());
  if (va->getWidth() == Expr::Int32) {
    float f = 1.0;
    one = APFloat(f); 
  } else {
    double d = 1.0;
    one = APFloat(d); 
  }
  llvm::APFloat Res(value.getSemantics());

  if (value.isNaN()) {
    Res = zero;
  } else {
    if (value.compare(zero) == APFloat::cmpLessThan) {
      Res = zero;
    } else if (value.compare(one) == APFloat::cmpGreaterThan) {
      Res = one;
    } else {
      Res = value;
    }
  }

  executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  return;
}

// __usad, __sad
static void executeSadIntrinsic(Executor &executor, ExecutionState &state, 
                                KInstruction *target, std::string fName, 
                                std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<Expr> Res;
  klee::ref<Expr> geCond; 
  if (fName.find("__usad") != std::string::npos)
    geCond = UgeExpr::create(arguments[0], arguments[1]); 
  else
    geCond = SgeExpr::create(arguments[0], arguments[1]); 
    
  Executor::StatePair branches = executor.forkAsPublic(state, geCond, true); 
  if (branches.first) {
    Res = AddExpr::create(SubExpr::create(arguments[0], arguments[1]), arguments[2]);
    executor.bindLocal(target, *branches.first, Res);
  }
  if (branches.second) {
    Res = AddExpr::create(SubExpr::create(arguments[1], arguments[0]), arguments[2]);
    executor.bindLocal(target, *branches.second, Res);
  } 
  return;
}

static APFloat::roundingMode determineRoundingMode(std::string fName) {
  if (fName.find("_rn") != std::string::npos) {
    return APFloat::rmNearestTiesToEven;
  } else if (fName.find("_rz") != std::string::npos) {
    return APFloat::rmTowardZero;
  } else if (fName.find("_ru") != std::string::npos) {
    return APFloat::rmTowardPositive;
  } else if (fName.find("_rd") != std::string::npos) {
    return APFloat::rmTowardNegative;
  } else {
    return APFloat::rmNearestTiesToEven;
  }
}

// fdividef, __fdividef, fdivide
static void executeFDivideIntrinsic(Executor &executor, ExecutionState &state, 
                                    KInstruction *target, std::string fName, 
                                    std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "__fdivide floating point");
  klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], "__fdivide floating point");
  
  if (!fpWidthToSemantics(va->getWidth()) ||
      !fpWidthToSemantics(vb->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported fdivide operation");
 
  if (fName.compare("fdividef") == 0) { // fdividef
    llvm::APFloat Res(va->getAPValue()); 
    Res.divide(APFloat(vb->getAPValue()), determineRoundingMode(fName));
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else { // __fdividef, fdivide
    llvm::APFloat VA(va->getAPValue());
    llvm::APFloat VB(vb->getAPValue());
   
    if (VB.isInfinity()) {
      if (VA.isInfinity()) {
        llvm::APFloat Res = APFloat::getNaN(VA.getSemantics()); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      } else {
        llvm::APFloat Res = APFloat::getZero(VA.getSemantics()); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      }
    } else {
      llvm::APFloat Res(va->getAPValue());
      Res.divide(APFloat(vb->getAPValue()), determineRoundingMode(fName));
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } 
  } 

  return;
}

static bool compareSubNormal(APFloat &value, bool isNeg) {
  return value.compare(APFloat::getSmallestNormalized(value.getSemantics(), 
                                                      isNeg)) == APFloat::cmpEqual; 
}

static void handleSpecialFloatValue(Executor &executor, ExecutionState &state, 
                                    KInstruction *target, std::string fName, 
                                    std::vector< klee::ref<Expr> > &arguments, 
                                    bool &special) {
  std::cout << "handleSpecialFloatValue" << std::endl;
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                         "__{sin,cos,rcp,sqrt}_ floating point");
  if (!fpWidthToSemantics(va->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

  llvm::APFloat value(va->getAPValue());
  llvm::APFloat Res(value.getSemantics());

  if (value.isInfinity() && value.isNegative()) {
    if (fName.find("sin") != std::string::npos
         || fName.find("cos") != std::string::npos
           || fName.find("sqrt") != std::string::npos
             || fName.find("log") != std::string::npos) 
      Res = APFloat::getNaN(value.getSemantics());
    else if (fName.find("rcp") != std::string::npos)
      Res = APFloat::getZero(value.getSemantics(), true); 
    else if (fName.find("exp") != std::string::npos)
      Res = APFloat::getZero(value.getSemantics());

    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (compareSubNormal(value, true)) {
    if (fName.find("sin") != std::string::npos 
         || fName.find("sqrt") != std::string::npos) {
      if (fName.find("rsqrt") != std::string::npos) 
        Res = APFloat::getNaN(value.getSemantics());
      else 
        Res = APFloat::getZero(value.getSemantics(), true);
    } else if (fName.find("cos") != std::string::npos
              || fName.find("exp") != std::string::npos) {
      if (va->getWidth() == Expr::Int32) {
        float f = 1.0;
        Res = APFloat(f);
      } else {
        double d = 1.0;
        Res = APFloat(d);
      }
    } else if (fName.find("rcp") != std::string::npos
                || fName.find("log") != std::string::npos) 
      Res = APFloat::getInf(value.getSemantics(), true);

    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (value.isZero() && value.isNegative()) {
    if (fName.find("sin") != std::string::npos
         || fName.find("sqrt") != std::string::npos) {
      if (fName.find("rsqrt") != std::string::npos)
        Res = APFloat::getInf(value.getSemantics(), true);
      else
        Res = APFloat::getZero(value.getSemantics(), true);
    } else if (fName.find("cos") != std::string::npos 
              || fName.find("exp") != std::string::npos) {
      if (va->getWidth() == Expr::Int32) {
        float f = 1.0;
        Res = APFloat(f);
      } else {
        double d = 1.0;
        Res = APFloat(d);
      }
    } else if (fName.find("rcp") != std::string::npos
                || fName.find("log") != std::string::npos) 
      Res = APFloat::getInf(value.getSemantics(), true);

    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (value.isZero() && !value.isNegative()) {
    if (fName.find("sin") != std::string::npos
         || fName.find("sqrt") != std::string::npos) {
      if (fName.find("rsqrt") != std::string::npos)
        Res = APFloat::getInf(value.getSemantics());
      else
        Res = APFloat::getZero(value.getSemantics());
    } else if (fName.find("cos") != std::string::npos
              || fName.find("exp") != std::string::npos) {
      if (va->getWidth() == Expr::Int32) {
        float f = 1.0;
        Res = APFloat(f);
      } else {
        double d = 1.0;
        Res = APFloat(d);
      }
    } 
    else if (fName.find("rcp") != std::string::npos) 
      Res = APFloat::getInf(value.getSemantics());
    else if (fName.find("log") != std::string::npos)
      Res = APFloat::getInf(value.getSemantics(), true); 

    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (compareSubNormal(value, false)) {
    if (fName.find("sin") != std::string::npos
         || fName.find("sqrt") != std::string::npos) {
      if (fName.find("rsqrt") != std::string::npos)
        Res = APFloat::getInf(value.getSemantics());
      else
        Res = APFloat::getZero(value.getSemantics());
    } else if (fName.find("cos") != std::string::npos
              || fName.find("exp") != std::string::npos) {
      if (va->getWidth() == Expr::Int32) {
        float f = 1.0;
        Res = APFloat(f);
      } else {
        double d = 1.0;
        Res = APFloat(d);
      }
    } 
    else if (fName.find("rcp") != std::string::npos) 
      Res = APFloat::getInf(value.getSemantics());
    else if (fName.find("log") != std::string::npos)
      Res = APFloat::getInf(value.getSemantics(), true); 

    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (value.isInfinity() && !value.isNegative()) {
    if (fName.find("sin") != std::string::npos
         || fName.find("cos") != std::string::npos)
      Res = APFloat::getNaN(value.getSemantics());
    else if (fName.find("rcp") != std::string::npos)
      Res = APFloat::getZero(value.getSemantics());
    else if (fName.find("sqrt") != std::string::npos
              || fName.find("exp") != std::string::npos
                || fName.find("log") != std::string::npos) { 
      if (fName.find("rsqrt") != std::string::npos)
        Res = APFloat::getZero(value.getSemantics());
      else 
        Res = APFloat::getInf(value.getSemantics());
    }

    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (value.isNaN()) {
    Res = APFloat::getNaN(value.getSemantics());
    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  } else if (value.isNormal() && value.isNegative() 
              && fName.find("sqrt") != std::string::npos) {
    Res = APFloat::getNaN(value.getSemantics());
    special = true;
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  }
  
  return;
}

// __sinf, sinf, sin, sinpif, sinpi
// __cosf, cosf, cos, cospif, cospi
// __tanf, tanf, tan
static void executeParTriangleOpIntrinsic(Executor &executor, ExecutionState &state, 
                                          KInstruction *target, std::string fName, 
                                          std::vector< klee::ref<Expr> > &arguments) {
  if (fName.find("sin") != std::string::npos) { // __sinf, sinf, sin
    bool special = false;
    handleSpecialFloatValue(executor, state, target, fName, arguments, special);

    if (!special) {
      klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "sin floating point");
      llvm::APFloat value(va->getAPValue());
      if (va->getWidth() == Expr::Int32) {
        float f = value.convertToFloat();
        float result = 0.0;
        if (fName.find("sinpi") != std::string::npos)
          result = sin(f * PI);
        else 
          result = sin(f);
        llvm::APFloat Res(result); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      } else {
        double d = value.convertToDouble();
        double result = 0.0;
        if (fName.find("sinpi") != std::string::npos)
          result = sin(d * PI);
        else 
          result = sin(d);
        llvm::APFloat Res(result); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      }
    }
  } else if (fName.find("cos") != std::string::npos) { // __cosf, cosf, cos
    bool special = false;
    handleSpecialFloatValue(executor, state, target, fName, arguments, special);

    if (!special) {
      klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "cos floating point");
      llvm::APFloat value(va->getAPValue());
      if (va->getWidth() == Expr::Int32) {
        float f = value.convertToFloat();
        float result = 0.0;
        if (fName.find("cospi") != std::string::npos)
          result = cos(f * PI);
        else 
          result = cos(f);
        llvm::APFloat Res(result); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      } else {
        double d = value.convertToDouble();
        double result = 0.0;
        if (fName.find("cospi") != std::string::npos)
          result = cos(d * PI);
        else 
          result = cos(d);
        llvm::APFloat Res(result);
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      }
    }
  } else { // __tanf, tanf, tan
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "tan floating point");
    llvm::APFloat value(va->getAPValue());
    if (va->getWidth() == Expr::Int32) {
      float f = value.convertToFloat();
      float result = tan(f);
      llvm::APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double d = value.convertToDouble();
      double result = tan(d);
      llvm::APFloat Res(result); 
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    }
  } 

  return;
}

// __fadd_, __dadd_
static void executeFloatAddIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> >& arguments) {
  klee::ref<klee::ConstantExpr> left = executor.toConstantPublic(state, arguments[0],
                                                     "__{f,d}add_ floating point");
  klee::ref<klee::ConstantExpr> right = executor.toConstantPublic(state, arguments[1],
                                                      "__{f,d}add_ floating point");
  if (!fpWidthToSemantics(left->getWidth()) ||
      !fpWidthToSemantics(right->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FAdd operation");

  llvm::APFloat Res(left->getAPValue());
  Res.add(APFloat(right->getAPValue()), determineRoundingMode(fName));
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  return;
}

// __fmul_, __dmul_
static void executeFloatMulIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<klee::ConstantExpr> left = executor.toConstantPublic(state, arguments[0],
                                               "__{f,d}mul_ floating point");
  klee::ref<klee::ConstantExpr> right = executor.toConstantPublic(state, arguments[1],
                                                "__{f,d}mul_ floating point");
  if (!fpWidthToSemantics(left->getWidth()) ||
      !fpWidthToSemantics(right->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FMul operation");

  llvm::APFloat Res(left->getAPValue());
  Res.multiply(APFloat(right->getAPValue()), determineRoundingMode(fName));
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  return;
}

// __fmaf_, __fma_, fmaf, fma
static void executeFloatFmaIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0],
                                                         "__fma_ floating point");
  klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1],
                                                         "__fma_ floating point");
  klee::ref<klee::ConstantExpr> vc = executor.toConstantPublic(state, arguments[2],
                                                         "__fma_ floating point");
  if (!fpWidthToSemantics(va->getWidth()) ||
      !fpWidthToSemantics(vb->getWidth()) ||
      !fpWidthToSemantics(vc->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FMma operation");

  llvm::APFloat Res(va->getAPValue());
  Res.fusedMultiplyAdd(APFloat(vb->getAPValue()), APFloat(vc->getAPValue()), 
                       determineRoundingMode(fName));
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  return;
}

// __frcp_, __drcp_ 
static void executeFloatRcpIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  bool special = false;
  handleSpecialFloatValue(executor, state, target, fName, arguments, special);
  
  if (!special) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "__fma_ floating point");
    if (!fpWidthToSemantics(va->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported FRcp operation");

    llvm::APFloat value(va->getAPValue());
    llvm::APFloat Res(value.getSemantics());

    if (fName.find("frcp") != std::string::npos) {
      float one = 1.0;
      Res = APFloat(one);
      Res.divide(value, determineRoundingMode(fName));
    } else { // __drcp
      double one = 1.0;
      Res = APFloat(one);
      Res.divide(value, determineRoundingMode(fName));
    }
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  }
  return;
}

// __fsqrt_, __dsqrt_, rsqrtf, rsqrt, sqrtf
static void executeFloatSqrtIntrinsic(Executor &executor, ExecutionState &state, 
                                      KInstruction *target, std::string fName, 
                                      std::vector< klee::ref<Expr> > &arguments) {
  bool special = false;
  handleSpecialFloatValue(executor, state, target, fName, arguments, special);
  
  if (!special) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "__sqrt_ floating point");
    llvm::APFloat value(va->getAPValue());
    llvm::APFloat Res(value.getSemantics());

    if (fName.find("rsqrt") != std::string::npos) {
      if (va->getWidth() == Expr::Int32) { // rsqrtf
        float f = value.convertToFloat();
        float result = 1.0/sqrt(f);
        Res = APFloat(result);
      } else { // rsqrt
        double d = value.convertToDouble();
        double result = 1.0/sqrt(d);
        Res = APFloat(result);
      }
    } else { // __fsqrt__
      if (fName.find("fsqrt") != std::string::npos) {
        float f = value.convertToFloat();
        float result = sqrt(f);
        Res = APFloat(result);
      } else {
        if (fName.find("dsqrt") != std::string::npos) { // __dsqrt__
          double d = value.convertToDouble();
          double result = sqrt(d);
          Res = APFloat(result);
        } else { //sqrtf
          float f = value.convertToFloat();
          float result = sqrt(f);
          Res = APFloat(result);
        }
      }
    }

    executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  }
  return;
}

// __fdiv_, __ddiv_
static void executeFloatDivIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                         "__div_ floating point");
  klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], 
                                                         "__div_ floating point");
  if (!fpWidthToSemantics(va->getWidth()) || 
      !fpWidthToSemantics(vb->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FDiv operation");

  llvm::APFloat Res(va->getAPValue());
  Res.divide(APFloat(vb->getAPValue()), determineRoundingMode(fName));
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
  return;
}

// exp, __expf, __exp10f, exp2, exp2f, exp10, expm1, expm1f
// __log2f, __log10f, __logf, log2f, log10, log, log1p, log1pf
// powf
static void executeParExponentialOpIntrinsic(Executor &executor, ExecutionState &state, 
                                             KInstruction *target, std::string fName, 
                                             std::vector< klee::ref<Expr> > &arguments) {
  if (fName.find("exp") != std::string::npos) { 
    // __expf, __exp10f, 
    // expf, exp2f, exp10f, expm1f
    // exp, exp2, exp10, expm1
    bool special = false;
    handleSpecialFloatValue(executor, state, target, fName, arguments, special);
    
    if (!special) {
      klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "expf floating point");
      llvm::APFloat value(va->getAPValue());
      llvm::APFloat Res(value.getSemantics());

      if (va->getWidth() == Expr::Int32) {
        float f = value.convertToFloat();
        if (fName.find("expf") != std::string::npos) {
          float result = exp(f);
          Res = APFloat(result); 
        } else if (fName.find("exp10f") != std::string::npos) { 
          float result = pow(10.0, f);
          Res = APFloat(result);
        } else if (fName.find("exp2f") != std::string::npos) { 
          float result = pow(2.0, f);
          Res = APFloat(result);
        } else {
          assert(fName.find("expm1f") != std::string::npos && "Another exp related function encountered");
          float result = exp(f) - 1;
          Res = APFloat(result);
        }

        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      } else {
        double d = value.convertToDouble();
        if (fName.find("exp10") != std::string::npos) {
          double result = pow(10.0, d);
          Res = APFloat(result);
        } else if (fName.find("exp2") != std::string::npos) {
          double result = pow(2.0, d);
          Res = APFloat(result);
        } else if (fName.find("exp") == 0) {
          double result = exp(d);
          Res = APFloat(result);
        } else {
          assert(fName.find("expm1") != std::string::npos && "Another exp related function encountered");
          double result = exp(d) - 1;
          Res = APFloat(result);
        }

        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      }
    }
  } else if (fName.find("log") != std::string::npos) { 
    // __log2f, __log10f, __logf, 
    // log2f, log10f, logf, log1pf
    // log2, log10, log, log1p
    bool special = false;
    if (fName.compare("log1p") != 0 && fName.compare("log1pf") != 0)
      handleSpecialFloatValue(executor, state, target, fName, arguments, special);
    
    if (!special) {
      klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "log floating point");
      llvm::APFloat value(va->getAPValue());
      if (va->getWidth() == Expr::Int32) {
        float f = value.convertToFloat();
        float result = 0.0;
        if (fName.find("log2f") != std::string::npos)
          result = log(f)/log(2.0);
        else if (fName.find("log10f") != std::string::npos) 
          result = log(f)/log(10.0);
        else if (fName.find("logf") != std::string::npos) 
          result = log(f);
        else {
          assert(fName.find("log1pf") != std::string::npos && "Another exp related function encountered");
          result = log(1.0+f);
        }

        llvm::APFloat Res(result); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      } else {
        double d = value.convertToDouble();
        double result = 0.0;
        if (fName.find("log10") != std::string::npos)
          result = log(d)/log(10.0);
        else if (fName.find("log2") != std::string::npos)
          result = log(d)/log(2.0); 
        else if (fName.compare("log") == 0) 
          result = log(d);
        else  {
          assert(fName.find("log1p") != std::string::npos && "Another exp related function encountered");
          result = log(1.0+d);
        }

        llvm::APFloat Res(result); 
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      }
    } 
  } else { // powf
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "powf floating point");
    klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], "powf floating point");

    if (!fpWidthToSemantics(va->getWidth()) ||
        !fpWidthToSemantics(vb->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    llvm::APFloat VB(vb->getAPValue());

    if (va->getWidth() == Expr::Int32) {
      float fa = VA.convertToFloat();
      float fb = VB.convertToFloat();

      float result = pow(fa, fb);
      llvm::APFloat Res(result);  
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double da = VA.convertToDouble();
      double db = VB.convertToDouble();

      double result = pow(da, db);
      llvm::APFloat Res(result);  
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } 
  }

  return;
}

static void fpComparisonOp(Executor &executor, ExecutionState &state, 
                           KInstruction *target, std::string fName, 
                           std::vector< klee::ref<Expr> > &arguments, bool isMin) {
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], "floating point");
  klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], "floating point");

  if (!fpWidthToSemantics(va->getWidth()) ||
      !fpWidthToSemantics(vb->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FCmp operation");

  llvm::APFloat VA(va->getAPValue()); 
  llvm::APFloat VB(vb->getAPValue()); 

  APFloat::cmpResult CmpRes = VA.compare(VB);

  if (CmpRes == APFloat::cmpLessThan)
    executor.bindLocal(target, state, isMin ? klee::ConstantExpr::alloc(VA) : klee::ConstantExpr::alloc(VB)); 
  else if (CmpRes == APFloat::cmpGreaterThan)
    executor.bindLocal(target, state, isMin ? klee::ConstantExpr::alloc(VB) : klee::ConstantExpr::alloc(VA)); 
  else 
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(VA)); 
}

static void integerComparisonOp(Executor &executor, ExecutionState &state, 
                                KInstruction *target, std::string fName, 
                                std::vector< klee::ref<Expr> > &arguments, bool isMin) {
  klee::ref<Expr> cond;
  if (isMin) {
    if (fName.find("umin") != std::string::npos 
         || fName.find("ullmin") != std::string::npos)
      cond = UleExpr::create(arguments[0], arguments[1]);
    else 
      cond = SleExpr::create(arguments[0], arguments[1]);
  } else { 
    if (fName.find("umax") != std::string::npos 
         || fName.find("ullmax") != std::string::npos)
      cond = UgeExpr::create(arguments[0], arguments[1]);
    else 
      cond = SgeExpr::create(arguments[0], arguments[1]);
  } 
  
  Executor::StatePair branches = executor.forkAsPublic(state, cond, true); 
  if (branches.first) {
    executor.bindLocal(target, *branches.first, arguments[0]);
  }
  if (branches.second) {
    executor.bindLocal(target, *branches.second, arguments[1]);
  } 
} 

// min, umin, llmin, ullmin, fminf, fmin
// max, umax, llmax, ullmax, fmaxf, fmax
static void executeParComparisonOpIntrinsic(Executor &executor, ExecutionState &state, 
                                            KInstruction *target, std::string fName, 
                                            std::vector< klee::ref<Expr> > &arguments) {
  if (fName.find("min") != std::string::npos) {
    if (fName.find("fmin") != std::string::npos) {
      fpComparisonOp(executor, state, target, fName, arguments, true); 
    } else {
      integerComparisonOp(executor, state, target, fName, arguments, true);
    } 
  } else {
    if (fName.find("fmax") != std::string::npos) {
      fpComparisonOp(executor, state, target, fName, arguments, false); 
    } else {
      integerComparisonOp(executor, state, target, fName, arguments, false);
    } 
  }
}

static uint64_t reverseBits(uint64_t a, unsigned numBits) {
  uint64_t cmp = 0; 
  uint64_t final = 0;
  for (unsigned i = 0; i < numBits; i++) {
    cmp = (1 << i);
    if (i < numBits/2) {
      uint64_t tmp = (a & cmp);
      tmp <<= (numBits-2*i-1);
      final |= tmp;
    } else { 
      uint64_t tmp = (a & cmp);
      tmp >>= (2*i-numBits+1);
      final |= tmp; 
    }
  }
  return final;
}

// clz, ffs, popc, brev, byte_perm 
static void executeParBitWiseOpIntrinsic(Executor &executor, ExecutionState &state, 
                                         KInstruction *target, std::string fName, 
                                         std::vector< klee::ref<Expr> > &arguments) {
  if (fName.find("clz") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "__clz integer");

    llvm::APInt value = va->getAPValue();
    unsigned clz = value.countLeadingZeros();
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(APInt(32, clz, true))); 
  } else if (fName.find("ffs") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "__ffs integer");
    llvm::APInt value = va->getAPValue();
    unsigned ffs = value.countTrailingZeros();
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(APInt(32, ffs, true))); 
  } else if (fName.find("popc") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "__popc integer");
    llvm::APInt value = va->getAPValue();
    unsigned popc = value.countPopulation();
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(APInt(32, popc, true))); 
  } else if (fName.find("brev") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "__brev integer");
    llvm::APInt value = va->getAPValue();
    const uint64_t *tmp = value.getRawData();
    if (va->getWidth() == Expr::Int32) {
      uint64_t a = reverseBits(*tmp, 32);   
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(APInt(32, a)));
    } else {
      uint64_t a = reverseBits(*tmp, 64);   
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(APInt(64, a)));
    }
  } else {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "__perm_byte integer");
    klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], 
                                                           "__perm_byte integer");
    klee::ref<klee::ConstantExpr> vc = executor.toConstantPublic(state, arguments[2], 
                                                           "__perm_byte integer");
    llvm::APInt VA = va->getAPValue();
    llvm::APInt VB = vb->getAPValue();
    llvm::APInt VC = vc->getAPValue();  

    unsigned final = 0;
    unsigned mValue = 0;
    const uint64_t *cvalue = VC.getRawData();
    uint64_t mark = *cvalue; 

    for (unsigned i = 0; i < 4; i++) {
      mark >>= 4*i;
      uint64_t b = mark & 7U;
      if (b <= 3) {
        llvm::APInt tmpInt = APInt::getBitsSet(32, b*8, (b+1)*8-1);
        llvm::APInt tmpVA(VA);
        tmpVA &= tmpInt; 
        const uint64_t *v = tmpVA.getRawData();
        mValue = *v;
      } else { 
        llvm::APInt tmpInt = APInt::getBitsSet(32, (b%4)*8, (b%4+1)*8-1);
        llvm::APInt tmpVB(VB);
        tmpVB &= tmpInt; 
        const uint64_t *v = tmpVB.getRawData();
        mValue = *v;
      } 

      unsigned remainder = b%4;
      if (i < remainder) {
        mValue >>= 8*(remainder-i); 
      } else {
        mValue <<= 8*(i-remainder); 
      }

      final |= mValue;
    }
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(final, Expr::Int32));
  }
}

// rhadd, urhadd, hadd, uhadd
static void executeParAddOpIntrinsic(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<Expr> result;
  if (fName.find("rhadd") != std::string::npos) { // __rhadd, __urhadd
    klee::ref<Expr> orExpr = OrExpr::create(arguments[0], arguments[1]); 
    klee::ref<Expr> xorExpr = XorExpr::create(arguments[0], arguments[1]); 
    klee::ref<Expr> oneExpr = klee::ConstantExpr::create(1, Expr::Int32);
    if (fName.find("urhadd") != std::string::npos)
      result = SubExpr::create(orExpr, LShrExpr::create(xorExpr, oneExpr)); 
    else 
      result = SubExpr::create(orExpr, AShrExpr::create(xorExpr, oneExpr)); 
  } else { // __hadd, __uhadd 
    klee::ref<Expr> andExpr = AndExpr::create(arguments[0], arguments[1]); 
    klee::ref<Expr> xorExpr = XorExpr::create(arguments[0], arguments[1]); 
    klee::ref<Expr> oneExpr = klee::ConstantExpr::create(1, Expr::Int32);
    if (fName.find("uhadd") != std::string::npos)
      result = AddExpr::create(andExpr, LShrExpr::create(xorExpr, oneExpr)); 
    else 
      result = AddExpr::create(andExpr, AShrExpr::create(xorExpr, oneExpr)); 
  }

  executor.bindLocal(target, state, result);
}

// abs, labs, llabs, fabs, fabsf 
static void executeAbsOpIntrinsic(Executor &executor, ExecutionState &state, 
                                  KInstruction *target, std::string fName, 
                                  std::vector< klee::ref<Expr> > &arguments) {
  if (fName.find("fabs") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "__fabs integer");
    if (!fpWidthToSemantics(va->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    if (VA.isNegative()) {
      if (va->getWidth() == Expr::Int32) {
        float result = fabs(VA.convertToFloat());
        APFloat Res(result);  
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      } else {
        double result = fabs(VA.convertToDouble());
        APFloat Res(result);  
        executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
      }
    } else
      executor.bindLocal(target, state, arguments[0]);
  } else {
    klee::ref<Expr> zeroExpr = klee::ConstantExpr::create(0, arguments[0]->getWidth());   
    klee::ref<Expr> geCond = SgeExpr::create(arguments[0], zeroExpr);
    
    Executor::StatePair branches = executor.forkAsPublic(state, geCond, true); 
    if (branches.first) {
      executor.bindLocal(target, *branches.first, arguments[0]);
    }
    if (branches.second) {
      executor.bindLocal(target, *branches.second, SubExpr::create(zeroExpr, arguments[0]));
    } 
  }
}

static void executeParFPConversionOpIntrinsic(Executor &executor, ExecutionState &state, 
                                              KInstruction *target, std::string fName, 
                                              std::vector< klee::ref<Expr> > &arguments) {
  if (fName.find("saturate") != std::string::npos) {
    executeSaturateIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("floor") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "floor fp operation");  
    if (!fpWidthToSemantics(va->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    if (va->getWidth() == Expr::Int32) {
      float result = floor(VA.convertToFloat()); 
      APFloat Res(result);  
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double result = floor(VA.convertToDouble()); 
      APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    }
  } else if (fName.find("ceil") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "ceil fp operation");  
    if (!fpWidthToSemantics(va->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    if (va->getWidth() == Expr::Int32) {
      float result = ceil(VA.convertToFloat()); 
      APFloat Res(result);  
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double result = ceil(VA.convertToDouble()); 
      APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    }
  } else if (fName.find("round") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "round fp operation");  
    if (!fpWidthToSemantics(va->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    if (va->getWidth() == Expr::Int32) {
      float result = round(VA.convertToFloat()); 
      APFloat Res(result);  
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double result = round(VA.convertToDouble()); 
      APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    }
  } else if (fName.find("trunc") != std::string::npos) { // trunc
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "trunc fp operation");  
    if (!fpWidthToSemantics(va->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    if (va->getWidth() == Expr::Int32) {
      float result = trunc(VA.convertToFloat()); 
      APFloat Res(result);  
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double result = trunc(VA.convertToDouble()); 
      APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    }
  } else { // fmod
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "trunc fp operation");
    klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], 
                                                           "trunc fp operation");
    if (!fpWidthToSemantics(va->getWidth()) 
         || !fpWidthToSemantics(vb->getWidth()))
      return executor.terminateStateOnExecErrorPublic(state, "Unsupported operation");

    llvm::APFloat VA(va->getAPValue());
    llvm::APFloat VB(vb->getAPValue());

    if (va->getWidth() == Expr::Int32) {
      float result = fmod(VA.convertToFloat(), VB.convertToFloat());
      APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    } else {
      double result = fmod(VA.convertToDouble(), VB.convertToDouble());
      APFloat Res(result);
      executor.bindLocal(target, state, klee::ConstantExpr::alloc(Res.bitcastToAPInt()));
    }
  }
}

#define NELEMS(array) (sizeof(array)/sizeof(array[0]))

static bool particularMul(std::string fName) {
  return (fName.find("mulhi") != std::string::npos 
           || fName.find("mul64hi") != std::string::npos
             || fName.find("mul24") != std::string::npos);
}

static bool particularTriangleOp(std::string fName) {
  return (fName.compare("__sinf") == 0
           || fName.compare("__cosf") == 0 
             || fName.compare("__tanf") == 0
               || fName.compare("sinf") == 0
                 || fName.compare("cosf") == 0
                   || fName.compare("tanf") == 0
                     || fName.compare("sin") == 0
                       || fName.compare("cos") == 0
                         || fName.compare("tan") == 0
                           || fName.find("sinpi") != std::string::npos
                             || fName.find("cospi") != std::string::npos);
}

static bool particularExponentialOp(std::string fName) {
  if (fName.find("frexp") != std::string::npos
       || fName.find("ldexp") != std::string::npos
         || fName.find("logb") != std::string::npos)
    return false;
  return (fName.find("exp") != std::string::npos
           || fName.find("log") != std::string::npos
             || fName.find("pow") != std::string::npos);
}

static bool particularComparisonOp(std::string fName) {
  return (fName.find("min") != std::string::npos
           || fName.find("max") != std::string::npos);
}

static bool particularBitWiseOp(std::string fName) {
  return (fName.find("clz") != std::string::npos
           || fName.find("ffs") != std::string::npos
             || fName.find("popc") != std::string::npos
               || fName.find("brev") != std::string::npos
                 || fName.find("byte_perm") != std::string::npos);
}

static bool particularFloatConversion(std::string fName) {
  return (fName.find("saturate") != std::string::npos
           || fName.find("round") != std::string::npos
             || fName.find("trunc") != std::string::npos
               || fName.find("floor") != std::string::npos
                 || fName.find("ceil") != std::string::npos
                   || fName.find("fmod") != std::string::npos);
}

static bool executeCUDAArithmetic(Executor &executor, 
                                  ExecutionState &state,
                                  KInstruction *target, 
                                  Function *f,
                                  std::vector< klee::ref<Expr> > &arguments) {
  std::string fName = f->getName().str();
  bool intrinsicFound = true;

  if (particularMul(fName)) {
    executeMulHiIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("sad") != std::string::npos) { 
    executeSadIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("fdivide") != std::string::npos) {
    executeFDivideIntrinsic(executor, state, target, fName, arguments);
  } else if (particularTriangleOp(fName)) {
    executeParTriangleOpIntrinsic(executor, state, target, fName, arguments);
  } else if (particularExponentialOp(fName)) {
    executeParExponentialOpIntrinsic(executor, state, target, fName, arguments);
  } else if (particularComparisonOp(fName)) {
    executeParComparisonOpIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("__fadd_") != std::string::npos
              || fName.find("__dadd_") != std::string::npos) {
    executeFloatAddIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("__fmul_") != std::string::npos
              || fName.find("__dmul_") != std::string::npos) {
    executeFloatMulIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("fma") != std::string::npos) {
    executeFloatFmaIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("rcp") != std::string::npos) {
    executeFloatRcpIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("sqrt") != std::string::npos) {
    executeFloatSqrtIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("__fdiv_") != std::string::npos 
              || fName.find("__ddiv_") != std::string::npos) {
    executeFloatDivIntrinsic(executor, state, target, fName, arguments);
  } else if (particularBitWiseOp(fName)) {
    executeParBitWiseOpIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("hadd") != std::string::npos) {
    executeParAddOpIntrinsic(executor, state, target, fName, arguments);
  } else if (fName.find("abs") != std::string::npos) {
    executeAbsOpIntrinsic(executor, state, target, fName, arguments);
  } else if (particularFloatConversion(fName)) {
    executeParFPConversionOpIntrinsic(executor, state, target, fName, arguments);
  } else { 
    intrinsicFound = false;
  }

  return intrinsicFound;
} 

static void executeCUDAFPToSI(Executor &executor, ExecutionState &state, 
                              KInstruction *target, std::string fName, 
                              std::vector< klee::ref<Expr> > &arguments) {
  CallSite cs(target->inst);
  Value *si = cs.getCalledValue();
  Expr::Width resultType = executor.getWidthForLLVMType(si->getType());
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0],
                                                         "floating point");
  if (!fpWidthToSemantics(va->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FPToSI operation");

  llvm::APFloat Arg(va->getAPValue());
  uint64_t value = 0;
  bool isExact = true;
  Arg.convertToInteger(&value, resultType, true,
                       determineRoundingMode(fName), &isExact);
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(value, resultType));
  return;
}

static void executeCUDAFPToUI(Executor &executor, ExecutionState &state, 
                              KInstruction *target, std::string fName, 
                              std::vector< klee::ref<Expr> > &arguments) {
  CallSite cs(target->inst);
  Value *ui = cs.getCalledValue();
  Expr::Width resultType = executor.getWidthForLLVMType(ui->getType());
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0],
                                                         "floating point");
  if (!fpWidthToSemantics(va->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported FPToUI operation");

  llvm::APFloat Arg(va->getAPValue());
  uint64_t value = 0;
  bool isExact = true;
  Arg.convertToInteger(&value, resultType, false,
                       determineRoundingMode(fName), &isExact);
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(value, resultType));
  return;
}

static void executeCUDASIToFP(Executor &executor, ExecutionState &state, 
                              KInstruction *target, std::string fName, 
                              std::vector< klee::ref<Expr> > &arguments) {
  CallSite cs(target->inst);
  Value *fp = cs.getCalledValue();
  Expr::Width resultType = executor.getWidthForLLVMType(fp->getType());
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0],
                                                         "floating point");
  const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
  if (!semantics)
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported SIToFP operation");
  llvm::APFloat f(*semantics, 0);
  f.convertFromAPInt(va->getAPValue(), true, determineRoundingMode(fName));
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(f));
  return;
}

static void executeCUDAUIToFP(Executor &executor, ExecutionState &state, 
                              KInstruction *target, std::string fName, 
                              std::vector< klee::ref<Expr> > &arguments) {
  CallSite cs(target->inst);
  Value *fp = cs.getCalledValue();
  Expr::Width resultType = executor.getWidthForLLVMType(fp->getType());
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0],
                                                         "floating point");
  const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
  if (!semantics)
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported UIToFP operation");
  llvm::APFloat f(*semantics, 0);
  f.convertFromAPInt(va->getAPValue(), false, determineRoundingMode(fName));
  executor.bindLocal(target, state, klee::ConstantExpr::alloc(f));
  return;
}

static void executeCUDAFPToHiOrLoInt(Executor &executor, ExecutionState &state, 
                                     KInstruction *target, std::string fName, 
                                     std::vector< klee::ref<Expr> > &arguments) {
  klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0],
                                                         "floating point");
  if (!fpWidthToSemantics(va->getWidth()))
    return executor.terminateStateOnExecErrorPublic(state, "Unsupported double2hiint operation");

  llvm::APInt tmp = va->getAPValue();
  if (fName.find("hiint") != std::string::npos) {
    tmp = tmp.getHiBits(32);
    tmp = tmp.ashr(32);
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(tmp.trunc(32)));  
  } else {
    tmp = tmp.getLoBits(32);
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(tmp.trunc(32)));  
  }

  return;
}

static bool executeCUDAConversion(Executor &executor, 
                                  ExecutionState &state,
                                  KInstruction *target, 
                                  Function *f,
                                  std::vector< klee::ref<Expr> > &arguments) {
  bool intrinsicFound = true;
  std::string fName = f->getName().str();

  if (fName.find("float2int") != std::string::npos
       || fName.find("float2ll") != std::string::npos
         || fName.find("double2int") != std::string::npos 
           || fName.find("double2ll") != std::string::npos) {
    executeCUDAFPToSI(executor, state, target, fName, arguments);
  } else if (fName.find("float2half") != std::string::npos 
              || fName.find("float2uint") != std::string::npos
                || fName.find("float2ull") != std::string::npos
                  || fName.find("double2uint") != std::string::npos
                    || fName.find("double2ull") != std::string::npos) {
    executeCUDAFPToUI(executor, state, target, fName, arguments);
  } else if (fName.find("int2float") != std::string::npos
              || fName.find("ll2float") != std::string::npos
                || fName.find("int2double") != std::string::npos
                  || fName.find("ll2double") != std::string::npos) {
    executeCUDASIToFP(executor, state, target, fName, arguments);
  } else if (fName.find("half2float") != std::string::npos 
              || fName.find("uint2float") != std::string::npos
                || fName.find("ull2float") != std::string::npos
                  || fName.find("uint2double") != std::string::npos
                    || fName.find("ull2double") != std::string::npos) {
    executeCUDAUIToFP(executor, state, target, fName, arguments);
  } else if (fName.find("double2hiint") != std::string::npos
              || fName.find("double2loint") != std::string::npos) {
    executeCUDAFPToHiOrLoInt(executor, state, target, fName, arguments);
  } else if (fName.find("hiloint2double") != std::string::npos) { // hiloint2double
    llvm::APInt tmp(64, 0);
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "hiloint2double op"); 
    klee::ref<klee::ConstantExpr> vb = executor.toConstantPublic(state, arguments[1], 
                                                           "hiloint2double op"); 
    llvm::APInt VA = va->getAPValue();
    llvm::APInt VB = vb->getAPValue();

    tmp |= VA;
    tmp <<= 32;
    tmp |= VB;

    llvm::APFloat fp(tmp);
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(fp.bitcastToAPInt()));
  } else if (fName.find("float_as_int") != std::string::npos
              || fName.find("double_as_longlong") != std::string::npos) {
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "bitcast to int or longlong");
    if (!fpWidthToSemantics(va->getWidth()))
      executor.terminateStateOnExecErrorPublic(state, "Unsupported bitcast operation");
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(va->getAPValue()));
  } else if (fName.find("int_as_float") != std::string::npos 
              || fName.find("longlong_as_double") != std::string::npos) {
    CallSite cs(target->inst);
    Value *fp = cs.getCalledValue();
    Expr::Width resultType = executor.getWidthForLLVMType(fp->getType());
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics)
      executor.terminateStateOnExecErrorPublic(state, "Unsupported bitcast operation");
    klee::ref<klee::ConstantExpr> va = executor.toConstantPublic(state, arguments[0], 
                                                           "bitcast to fp");
    llvm::APFloat FP(va->getAPValue());
    executor.bindLocal(target, state, klee::ConstantExpr::alloc(FP.bitcastToAPInt()));
  } 
  else intrinsicFound = false;

  return intrinsicFound;
}

void Executor::executeCUDAIntrinsics(ExecutionState &state, KInstruction *target, 
                                     Function *f, std::vector< klee::ref<Expr> > &arguments, 
                                     unsigned seqNum) {
  Gklee::Logging::enterFunc( arguments, __PRETTY_FUNCTION__ );
  Gklee::Logging::outItem< std::string >( f->getName(), "func name" );
  std::string fName = f->getName().str();

  // Some functions in host code are also able to reuse those functions
  if (state.tinfo.is_GPU_mode) {
    if (executeCUDAArithmetic(*this, state, target, f, arguments)){
      Gklee::Logging::exitFunc();
      return;
    }

    if (executeCUDAConversion(*this, state, target, f, arguments)){
      Gklee::Logging::exitFunc();
      return;
    }

    if (executeCUDAAtomic(state, target, fName, arguments, seqNum)) {
      Gklee::Logging::exitFunc();
      return;
    }

    for (unsigned i = 0; i < NELEMS(CUDAMemfence); i++) {
      if (fName.find(CUDAMemfence[i]) != std::string::npos) {
        // No need to write function body for thread_fence intrinsics
        handleMemfence(state, target);
	Gklee::Logging::exitFunc();
        return; 
      }
    }
   
    for (unsigned i = 0; i < NELEMS(CUDASync); i++) {
      if (fName.find(CUDASync[i]) != std::string::npos) {
	 //TODO flow experiment
        handleBarrier(state, target);
	 //TODO flow experiment
	Gklee::Logging::exitFunc();
        return; 
      }
    }
  }

  //temp fix for cleanup problem (see branch) -- this should be unnecessary, as it's handled in kleeUclibc, I 
  // think we're neglecting to use that
  if( fName != "__cxa_atexit" ){ //&& 
    //fName.find( "GLOBAL" ) == std::string::npos ){
    callExternalFunction(state, target, f, arguments);
  }
  //  callExternalFunction(state, target, f, arguments);
  Gklee::Logging::exitFunc();
}
