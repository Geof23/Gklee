#include "Executor.h"
#include "klee/Expr.h"
#include "klee/util/ExprUtil.h"
#include "AddressSpace.h"
#include "TimingSolver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_os_ostream.h"

#include "CUDA.h"
#include <string>

using namespace llvm;
using namespace klee;

namespace runtime {
  cl::opt<bool>
  UnboundConfig("unbound-config", 
                cl::desc("use unbounded configuration instead"), 
                cl::init(false)); 
  extern cl::opt<bool> UseSymbolicConfig;
  extern cl::opt<bool> SimdSchedule;
}

using namespace runtime;

static bool isCurrentConfigFulfilled(Executor &executor, ExecutionState &state, 
                                     ref<Expr> &configExpr, ref<Expr> &typeExpr) {
  bool result = false;
  state.paraConstraints = state.constraints;
  if (!UnboundConfig)
    ExecutorUtil::constructSymConfigEncodedConstraint(state);
  else 
    ExecutorUtil::constructSymBlockDimPrecondition(state);
  
  ExecutorUtil::addConfigConstraint(state, configExpr);
  executor.solver->mayBeTrue(state, typeExpr, result);
  return result;
}

void AddressSpaceUtil::updateBuiltInRelatedConstraint(ExecutionState &state, ConstraintManager &constr, 
                                                      ref<Expr> &expr) {
  std::map< ref<Expr>, ref<Expr> > equalities; 
  // update bid 
  MemoryObject *bo = state.tinfo.block_id_mo;
  std::vector<AddressSpace> &sharedMemories = state.addressSpace.sharedMemories;

  ObjectState *bos0 = sharedMemories[0].findNonConstantObject(bo);
  ref<Expr> bidx0 = bos0->read(0, Expr::Int32);     
  ref<Expr> bidy0 = bos0->read(4, Expr::Int32);     
  ref<Expr> bidz0 = bos0->read(8, Expr::Int32);     

  ObjectState *bos1 = sharedMemories[1].findNonConstantObject(bo);
  ref<Expr> bidx1 = bos1->read(0, Expr::Int32);     
  ref<Expr> bidy1 = bos1->read(4, Expr::Int32);     
  ref<Expr> bidz1 = bos1->read(8, Expr::Int32);     

  equalities.insert(std::make_pair(bidx0, bidx1));
  equalities.insert(std::make_pair(bidy0, bidy1));
  equalities.insert(std::make_pair(bidz0, bidz1));

  // update tid 
  MemoryObject *mo = state.tinfo.thread_id_mo;
  std::vector<AddressSpace> &localMemories = state.addressSpace.localMemories;
  ObjectState *tos0 = localMemories[0].findNonConstantObject(mo);
  ref<Expr> tidx0 = tos0->read(0, Expr::Int32);
  ref<Expr> tidy0 = tos0->read(4, Expr::Int32);
  ref<Expr> tidz0 = tos0->read(8, Expr::Int32);

  ObjectState *tos1 = localMemories[1].findNonConstantObject(mo);
  ref<Expr> tidx1 = tos1->read(0, Expr::Int32);
  ref<Expr> tidy1 = tos1->read(4, Expr::Int32);
  ref<Expr> tidz1 = tos1->read(8, Expr::Int32);

  equalities.insert(std::make_pair(tidx0, tidx1));
  equalities.insert(std::make_pair(tidy0, tidy1));
  equalities.insert(std::make_pair(tidz0, tidz1));

  if (expr.get() != NULL) {
    ref<Expr> tmp = constr.updateExprThroughReplacement(expr, equalities); 
    expr = constr.simplifyExpr(tmp);  
  }
}

void AddressSpaceUtil::updateMemoryAccess(ExecutionState &state, ConstraintManager &constr, 
                                          MemoryAccess &access) {
  AddressSpaceUtil::updateBuiltInRelatedConstraint(state, constr, access.offset);
  AddressSpaceUtil::updateBuiltInRelatedConstraint(state, constr, access.accessCondExpr);
  if (access.val.get() != NULL)
    AddressSpaceUtil::updateBuiltInRelatedConstraint(state, constr, access.val);
}

static int getSegmentSize(Expr::Width width, unsigned capability) {
  unsigned wordsize = width / 8;

  // For capability 1.0 and 1.1 ...
  if (capability == 0) {
    if (wordsize == 4)
      return 64;
    else if (wordsize == 8)
      return 128;
    else if (wordsize == 16)
      return 256;
    else 
      return -1;
  } else if(capability == 1) {
    // For capability 1.2 and 1.3 ...
    if (wordsize == 1)
      return 32;
    else if (wordsize == 2)
      return 64;
    else if (wordsize == 4 || wordsize == 8 || wordsize == 16)
      return 128;
    else 
      return -1;
  } else {
    if (wordsize == 4)
      return 1;
    else if (wordsize == 8)
      return 2;
    else if (wordsize == 16)
      return 4;
    else 
      return -1;
  }
}

// Ensure that instructions belong to different BBs ... 
static bool belongToDifferentBB(MemoryAccess &access1, MemoryAccess &access2) {
  llvm::Instruction *inst1 = access1.instr;
  llvm::Instruction *inst2 = access2.instr;

  std::string func1Name = inst1->getParent()->getParent()->getName().str();
  std::string func2Name = inst2->getParent()->getParent()->getName().str();
  std::string bb1Name = inst1->getParent()->getName().str();
  std::string bb2Name = inst2->getParent()->getName().str();

  if (func1Name.compare(func2Name) == 0) {
    if (bb1Name.compare(bb2Name) == 0)
      return false;
    else 
      return true;
  } else return true;
}

static inline bool existThreadInSet(std::set<unsigned> &set, unsigned tid) {
  std::set<unsigned>::iterator found = set.find(tid);
  return found != set.end();
}

ref<Expr> AddressSpaceUtil::constructSameBlockExpr(ExecutionState &state, Expr::Width width) {
  MemoryObject *mo = state.tinfo.block_id_mo;
  ObjectState *bos1 = state.addressSpace.sharedMemories[0].findNonConstantObject(mo);
  // bid x ...
  ref<Expr> bidx1 = bos1->read(0, width);
  // bid y ...
  ref<Expr> bidy1 = bos1->read(4, width);
  // bid z ...
  ref<Expr> bidz1 = bos1->read(8, width);

  ObjectState *bos2 = state.addressSpace.sharedMemories[1].findNonConstantObject(mo);
  // bid x ...
  ref<Expr> bidx2 = bos2->read(0, width);
  // bid y ...
  ref<Expr> bidy2 = bos2->read(4, width);
  // bid z ...
  ref<Expr> bidz2 = bos2->read(8, width);

  ref<Expr> result = AndExpr::create(AndExpr::create(EqExpr::create(bidx1, bidx2), 
                                                     EqExpr::create(bidy1, bidy2)), 
                                     EqExpr::create(bidz1, bidz2));   
  return result;
}

ref<Expr> AddressSpaceUtil::constructSameThreadExpr(ExecutionState &state, Expr::Width width) {
  MemoryObject *mo = state.tinfo.thread_id_mo;
  ObjectState *tos1 = state.addressSpace.localMemories[0].findNonConstantObject(mo);
  // tid x ...
  ref<Expr> tidx1 = tos1->read(0, width);
  // tid y ...
  ref<Expr> tidy1 = tos1->read(4, width);
  // tid z ...
  ref<Expr> tidz1 = tos1->read(8, width);

  ObjectState *tos2 = state.addressSpace.localMemories[1].findNonConstantObject(mo);
  // tid x ...
  ref<Expr> tidx2 = tos2->read(0, width);
  // tid y ...
  ref<Expr> tidy2 = tos2->read(4, width);
  // tid z ...
  ref<Expr> tidz2 = tos2->read(8, width);

  ref<Expr> result = AndExpr::create(AndExpr::create(EqExpr::create(tidx1, tidx2), 
                                                     EqExpr::create(tidy1, tidy2)), 
                                     EqExpr::create(tidz1, tidz2));   
  return result;
}

ref<Expr> AddressSpaceUtil::constructRealThreadNumConstraint(ExecutionState &state, unsigned tid,
                                                             Expr::Width width) {
  MemoryObject *mo = state.tinfo.thread_id_mo;
  ObjectState *os = state.addressSpace.localMemories[tid].findNonConstantObject(mo);
  // tid x ...
  ref<Expr> tidx = os->read(0, width);
  // tid y ...
  ref<Expr> tidy = os->read(4, width);
  // tid z ...
  ref<Expr> tidz = os->read(8, width);

  ref<Expr> bs0, bs1;
  if (!UnboundConfig) {
    bs0 = klee::ConstantExpr::create(GPUConfig::SymBlockSize[0], Expr::Int32); 
    bs1 = klee::ConstantExpr::create(GPUConfig::SymBlockSize[1], Expr::Int32); 
  } else {
    if (state.tinfo.sym_bdim_mo == NULL) 
      GKLEE_INFO << "sym_bdim_mo is NULL" << std::endl; 

    ObjectState *bdimos = state.addressSpace.cpuMemory.findNonConstantObject(state.tinfo.sym_bdim_mo);
    bs0 = bdimos->read(0, Expr::Int32);
    bs1 = bdimos->read(4, Expr::Int32); 
  }

  ref<Expr> part1 = AddExpr::create(MulExpr::create(bs1, tidz), tidy);
  ref<Expr> part2 = MulExpr::create(bs0, part1);
  ref<Expr> part3 = AddExpr::create(part2, tidx); 
  return part3;
}

// construct the RW or WW conflict constraint ...
ref<Expr> symCheckConflictExprs(ref<Expr> &addr1, Expr::Width width1, 
                                ref<Expr> &addr2, Expr::Width width2) {
  unsigned boffset1 = (width1 - 1) >> 3; 
  ref<Expr> hbound1 =  boffset1 == 0 ? addr1 : 
    AddExpr::create(addr1, klee::ConstantExpr::create(boffset1, addr1->getWidth()));

  unsigned boffset2 = (width2 - 1) >> 3; 
  ref<Expr> hbound2 =  boffset2 == 0 ? addr2 : 
    AddExpr::create(addr2, klee::ConstantExpr::create(boffset2, addr2->getWidth()));

  ref<Expr> expr1 = AndExpr::create(UleExpr::create(addr1, addr2),
				    UleExpr::create(addr2, hbound1));
  ref<Expr> expr2 = AndExpr::create(UleExpr::create(addr2, addr1),
				    UleExpr::create(addr1, hbound2));
  ref<Expr> expr = OrExpr::create(expr1, expr2);
 
  return expr;
}

// construct the condition: 
// (B_1 == B_2) && (t_1 != t_2) && (t_1/32 == t_2/32)
ref<Expr> AddressSpaceUtil::threadSameWarpConstraint(ExecutionState &state, unsigned warpsize) {
  ref<Expr> sameTidExpr = AddressSpaceUtil::constructSameThreadExpr(state, Expr::Int32);

  ref<Expr> tConstr1 = AddressSpaceUtil::constructRealThreadNumConstraint(state, 0, Expr::Int32);
  ref<Expr> tConstr2 = AddressSpaceUtil::constructRealThreadNumConstraint(state, 1, Expr::Int32);
  ref<Expr> warpSizeExpr = klee::ConstantExpr::create(warpsize, Expr::Int32);
  ref<Expr> tidExpr1 = UDivExpr::create(tConstr1, warpSizeExpr); 
  ref<Expr> tidExpr2 = UDivExpr::create(tConstr2, warpSizeExpr);

  ref<Expr> sameWarpExpr = AndExpr::create(Expr::createIsZero(sameTidExpr), 
                                           EqExpr::create(tidExpr1, tidExpr2)); 

  if (GPUConfig::num_blocks > 1) {
    ref<Expr> sameBidExpr = AddressSpaceUtil::constructSameBlockExpr(state, Expr::Int32);
    sameWarpExpr = AndExpr::create(sameBidExpr, sameWarpExpr);
  } 

  return sameWarpExpr;
}

// construct the condition :
// (B_1 == B_2) && (t_1/32 != t_2/32)
ref<Expr> AddressSpaceUtil::threadSameBlockDiffWarpConstraint(ExecutionState &state, unsigned warpsize) {
  ref<Expr> tConstr1 = AddressSpaceUtil::constructRealThreadNumConstraint(state, 0, Expr::Int32);
  ref<Expr> tConstr2 = AddressSpaceUtil::constructRealThreadNumConstraint(state, 1, Expr::Int32);
  ref<Expr> warpSizeExpr = klee::ConstantExpr::create(warpsize, Expr::Int32);
  ref<Expr> tidExpr1 = UDivExpr::create(tConstr1, warpSizeExpr); 
  ref<Expr> tidExpr2 = UDivExpr::create(tConstr2, warpSizeExpr);

  ref<Expr> diffWarpExpr = NeExpr::create(tidExpr1, tidExpr2);

  if (GPUConfig::num_blocks > 1) {
    ref<Expr> sameBlockExpr = AddressSpaceUtil::constructSameBlockExpr(state, Expr::Int32);
    diffWarpExpr = AndExpr::create(sameBlockExpr, diffWarpExpr);
  }

  return diffWarpExpr;
}

// construct the condition :
// (B_1 != B_2)
ref<Expr> AddressSpaceUtil::threadDiffBlockConstraint(ExecutionState &state) {
  ref<Expr> sameBlockExpr = AddressSpaceUtil::constructSameBlockExpr(state, Expr::Int32);
  return Expr::createIsZero(sameBlockExpr);
}

static void dumpSymBankConflict(Executor &executor, ExecutionState &state, 
                                ref<Expr> &bcCond, 
                                MemoryAccess &access1, MemoryAccess &access2) {
  bool benign = false;
  std::vector<SymBlockID_t> symBlockIDs;
  std::vector<SymThreadID_t> symThreadIDs;
  SymBlockDim_t symBlockDim(0, 0, 0);
 
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  
  std::vector< ref<Expr> > offsetVec, cOffsetVec;
  offsetVec.push_back(access1.offset); 
  offsetVec.push_back(access2.offset); 

  bool success = executor.getSymbolicConfigSolution(state, bcCond, offsetVec, 
                                                    cOffsetVec, access1.val, 
                                                    access2.val, benign, 
                                                    symBlockIDs, symThreadIDs, 
                                                    symBlockDim); 

  if (success) {
    GKLEE_INFO << "++++++++++" << std::endl;
    GKLEE_INFO << "Access1 in Bank Conflict: " << std::endl;
    access1.dump(executor, state, bcCond);
    GKLEE_INFO << "Access2 in Bank Conflict: " << std::endl;
    access2.dump(executor, state, bcCond);
    GKLEE_INFO << "Thread 1 : { <" << symBlockIDs[0].x << ", " 
               << symBlockIDs[0].y << ", " << symBlockIDs[0].z << ">" 
               << ", <" << symThreadIDs[0].x << ", " << symThreadIDs[0].y 
               << ", " << symThreadIDs[0].z
               << "> }" << " and Thread 2 : { <" << symBlockIDs[1].x << ", " 
               << symBlockIDs[1].y << ", " << symBlockIDs[1].z << ">" 
               << ", <" << symThreadIDs[1].x << ", " << symThreadIDs[1].y 
               << ", " << symThreadIDs[1].z
               << "> } incur the bank conflict!" << std::endl; 
    GKLEE_INFO << "Thread 1's concretized offset: " << std::endl;
    cOffsetVec[0]->dump();
    GKLEE_INFO << "Thread 2's concretized offset: " << std::endl;
    cOffsetVec[1]->dump();
    GKLEE_INFO << "++++++++++" << std::endl;
  }
}

static bool symCheckWriteBankConflictExprsCap1x(Executor &executor, ExecutionState &state, 
                                                MemoryAccess &access1, MemoryAccess &access2,
                                                unsigned BankNum, ThreadInfo &tinfo) {
  ref<Expr> addr1 = access1.offset;
  ref<Expr> addr2 = access2.offset;

  ref<Expr> bankSize = klee::ConstantExpr::create(BankNum * 4, addr1->getWidth());
  ref<Expr> wordSize = klee::ConstantExpr::create(4, addr1->getWidth());
  ref<Expr> a1 = UDivExpr::create(URemExpr::create(addr1, bankSize), wordSize);
  ref<Expr> a2 = UDivExpr::create(URemExpr::create(addr2, bankSize), wordSize);
  ref<Expr> expr = EqExpr::create(a1, a2);

  // Determine whether the offset expression is related to 
  // Symbolic values ...
  ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
  ref<Expr> tRelationExpr = AddressSpaceUtil::threadSameWarpConstraint(state, BankNum); 
  bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, tRelationExpr); 

  if (configFulfilled) {
    bool result = false;
    ExecutorUtil::addConfigConstraint(state, tRelationExpr);
    bool success = executor.solver->mustBeFalse(state, expr, result);
    if (success) {
      if (!result) {
        dumpSymBankConflict(executor, state, expr, access1, access2);
      }
      ExecutorUtil::copyBackConstraint(state);
      return !result;
    }
  }
  ExecutorUtil::copyBackConstraint(state);
  return false;
}

static bool symCheckReadBankConflictExprsCap1x(Executor &executor, ExecutionState &state, 
                                               MemoryAccess &access1, MemoryAccess access2,
                                               unsigned BankNum, ThreadInfo &tinfo) {
  ref<Expr> addr1 = access1.offset; 
  ref<Expr> addr2 = access2.offset; 

  ref<Expr> origEq = EqExpr::create(addr1, addr2);
  // Determine whether the offset expression is related to 
  // Symbolic values ...
  ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
  ref<Expr> tRelationExpr = AddressSpaceUtil::threadSameWarpConstraint(state, BankNum);
  bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, tRelationExpr); 

  if (configFulfilled) {
    bool result = false;
    ExecutorUtil::addConfigConstraint(state, tRelationExpr);
    bool success = executor.solver->mustBeTrue(state, origEq, result);
    if (success) {
      if (result) {
        ExecutorUtil::copyBackConstraint(state);
        return false; // broadcast...
      }
    }

    ref<Expr> tmpExpr = NeExpr::create(addr1, addr2);
    ref<Expr> bankSize = klee::ConstantExpr::create(BankNum * 4, addr1->getWidth());
    ref<Expr> wordSize = klee::ConstantExpr::create(4, addr1->getWidth());
    ref<Expr> a1 = UDivExpr::create(URemExpr::create(addr1, bankSize), wordSize);
    ref<Expr> a2 = UDivExpr::create(URemExpr::create(addr2, bankSize), wordSize);
    ref<Expr> expr = EqExpr::create(a1, a2);
    ref<Expr> andExpr = AndExpr::create(tmpExpr, expr);
  
    success = executor.solver->mustBeFalse(state, andExpr, result);
    if (success) {
      if (!result) {
        // try the first one...
        dumpSymBankConflict(executor, state, expr, access1, access2);
      }
      ExecutorUtil::copyBackConstraint(state);
      return !result;
    }
  }

  ExecutorUtil::copyBackConstraint(state);
  return false;
}

static bool checkSymBankConflictCap1x(Executor &executor, ExecutionState &state, 
                                      MemoryAccessVec &rwSet, bool isWrite) {
  bool hasBC = false;
  
  for (unsigned i = 0; i < rwSet.size(); i++) {
    MemoryAccess tmpAccess(rwSet[i]);
    ConstraintManager constr;     
    AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  
       
    bool result = isWrite ? 
        symCheckWriteBankConflictExprsCap1x(executor, state, 
                                            rwSet[i], tmpAccess,
                                            GPUConfig::warpsize/2, state.tinfo) : 
        symCheckReadBankConflictExprsCap1x(executor, state,
                                           rwSet[i], tmpAccess,
                                           GPUConfig::warpsize/2, state.tinfo);
    if (result) {
      hasBC = true;
      break;
    }
  }

  return hasBC;
}

static bool symCheckBankConflictExprsCap2x(Executor &executor, ExecutionState &state, 
                                           MemoryAccess &access1, MemoryAccess &access2, 
                                           unsigned BankNum, ThreadInfo &tinfo) {
  ref<Expr> addr1 = access1.offset; 
  ref<Expr> addr2 = access2.offset; 

  // Eliminate the same word ...
  ref<Expr> wordSize = klee::ConstantExpr::create(4, addr1->getWidth());
  ref<Expr> a1 = UDivExpr::create(addr1, wordSize);
  ref<Expr> a2 = UDivExpr::create(addr2, wordSize);
  ref<Expr> expr = EqExpr::create(a1, a2);

  // Determine whether the offset expression is related to 
  // Symbolic values ...
  ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
  ref<Expr> tRelationExpr = AddressSpaceUtil::threadSameWarpConstraint(state, BankNum); 
  bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, tRelationExpr);

  if (configFulfilled) {
    bool result = false;
    ExecutorUtil::addConfigConstraint(state, tRelationExpr);
    bool success = executor.solver->mustBeTrue(state, expr, result);

    if (success)
      if (result) {
        ExecutorUtil::copyBackConstraint(state);
        return false; // broadcast ...
      }

    ref<Expr> tmpExpr = NeExpr::create(a1, a2);
    ref<Expr> bankSize = klee::ConstantExpr::create(BankNum * 4, addr1->getWidth());
    ref<Expr> b1 = UDivExpr::create(URemExpr::create(addr1, bankSize), wordSize);
    ref<Expr> b2 = UDivExpr::create(URemExpr::create(addr2, bankSize), wordSize);
    ref<Expr> andExpr = AndExpr::create(tmpExpr, EqExpr::create(b1, b2));

    success = executor.solver->mayBeTrue(state, andExpr, result);
    if (success) {
      if (result) {
        // try the first one...
        dumpSymBankConflict(executor, state, andExpr, access1, access2);
      }
      ExecutorUtil::copyBackConstraint(state);
      return result;
    }
  }

  ExecutorUtil::copyBackConstraint(state);
  return false;
}

static bool checkSymBankConflictCap2x(Executor &executor, ExecutionState &state, 
                                      MemoryAccessVec &rwSet) {
  bool hasBC = false;
  
  for (unsigned i = 0; i < rwSet.size(); i++) {
    MemoryAccess tmpAccess(rwSet[i]);
    ConstraintManager constr;     
    AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

    bool result = symCheckBankConflictExprsCap2x(executor, state, rwSet[i], tmpAccess, 
                                                 GPUConfig::warpsize, state.tinfo);
    if (result) {
      hasBC = true;
      break;
    }
  }

  return hasBC;
} 

bool AddressSpace::hasSymBankConflict(Executor &executor, ExecutionState &state,
                                      unsigned DevCap) {
  bool hasReadBC = false;
  bool hasWriteBC = false;

  if (DevCap == 0 || DevCap == 1) {
    // 1.x
    // ReadSet first...
    if (readSet.empty()) {
      GKLEE_INFO << "The read set is empty in bank conflict checking for capability 1.x"
                 << std::endl;
    } else {
      hasReadBC = checkSymBankConflictCap1x(executor, state, readSet, false);
    }
    // WriteSet ...
    if (writeSet.empty()) {
      GKLEE_INFO << "The write set is empty in bank conflict checking for capability 1.x"
                 << std::endl; 
    } else {
      hasWriteBC = checkSymBankConflictCap1x(executor, state, writeSet, true); 
    }
  } else {
    // 2.x
    // ReadSet first...
    if (readSet.empty()) {
      GKLEE_INFO << "The read set is empty in bank conflict checking for capability 2.x"
                 << std::endl; 
    } else {
      hasReadBC = checkSymBankConflictCap2x(executor, state, readSet);
    }
    // WriteSet ...
    if (writeSet.empty()) {
      GKLEE_INFO << "The write set is empty in bank conflict checking for capability 2.x"
                 << std::endl; 
    } else {
      hasWriteBC = checkSymBankConflictCap2x(executor, state, writeSet);
    }
  }

  return (hasReadBC || hasWriteBC);
}

static void dumpSymMemoryNonCoalescing(Executor &executor, ExecutionState &state, 
                                       ref<Expr> &expr, 
                                       MemoryAccess &access1, MemoryAccess &access2,
                                       unsigned capability) {
  bool benign = false;
  std::vector<SymBlockID_t> symBlockIDs;
  std::vector<SymThreadID_t> symThreadIDs;
  SymBlockDim_t symBlockDim(0, 0, 0);
 
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  
  std::vector< ref<Expr> > offsetVec, cOffsetVec;
  offsetVec.push_back(access1.offset); 
  offsetVec.push_back(access2.offset); 

  bool success = executor.getSymbolicConfigSolution(state, expr, offsetVec, 
                                                    cOffsetVec, access1.val, 
                                                    access2.val, benign, 
                                                    symBlockIDs, symThreadIDs, 
                                                    symBlockDim); 

  std::string cap;
  if (capability == 0)
    cap = "capability 1.0 and 1.1";
  else if (capability == 1)
    cap = "capability 1.2 and 1.3";
  else 
    cap = "capabiity 2.x";

  if (success) {
    GKLEE_INFO << "++++++++++ Non Memory Coalescing at " << cap 
               << "++++++++++" << std::endl;
    GKLEE_INFO << "Access1 in Non Memory Coalescing: " << std::endl;
    access1.dump(executor, state, expr);
    GKLEE_INFO << "Access2 in Non Memory Coalescing: " << std::endl;
    access2.dump(executor, state, expr);

    GKLEE_INFO << "Thread 1 : { <" << symBlockIDs[0].x << ", " 
               << symBlockIDs[0].y << ", " << symBlockIDs[0].z << ">" 
               << ", <" << symThreadIDs[0].x << ", " << symThreadIDs[0].y
               << ", " << symThreadIDs[0].z
               << "> }" << " and Thread 2 : { <" << symBlockIDs[1].x << ", " 
               << symBlockIDs[1].y << ", " << symBlockIDs[1].z << ">" 
               << ", <" << symThreadIDs[1].x << ", " << symThreadIDs[1].y 
               << ", " << symThreadIDs[1].z
               << "> } incur the non-coalesced memory access!" << std::endl; 
    GKLEE_INFO << "Thread 1's concretized offset: " << std::endl;
    cOffsetVec[0]->dump();
    GKLEE_INFO << "Thread 2's concretized offset: " << std::endl;
    cOffsetVec[1]->dump();
    GKLEE_INFO << "++++++++++" << std::endl;
  }
}

static bool symCheckMemoryAccessOutOfSegBound(Executor &executor, ExecutionState &state, 
                                              MemoryAccess &access1, MemoryAccess access2,
                                              unsigned segSize, unsigned threadSize, 
                                              unsigned capability) {
  bool outOfBound = false;
  ref<Expr> addr1 = access1.offset; 
  ref<Expr> addr2 = access2.offset; 

  ref<Expr> segExpr = klee::ConstantExpr::create(segSize, addr1->getWidth());
  ref<Expr> expr = EqExpr::create(UDivExpr::create(addr1, segExpr), 
                                  UDivExpr::create(addr2, segExpr));
  
  ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
  ref<Expr> tThreadExpr = AddressSpaceUtil::threadSameWarpConstraint(state, threadSize);
  bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, tThreadExpr); 

  if (configFulfilled) {
    bool result = false;
    ExecutorUtil::addConfigConstraint(state, tThreadExpr);
    bool success = executor.solver->mustBeTrue(state, expr, result);
    if (success) {
      if (!result) {
        ref<Expr> tmpExpr = Expr::createIsZero(expr);
        dumpSymMemoryNonCoalescing(executor, state, tmpExpr, 
                                   access1, access2, capability);
        outOfBound = true;
      }
    }
  }
  ExecutorUtil::copyBackConstraint(state);
  return outOfBound;
}

static bool symCheckMemoryCoalescingCap0(Executor &executor, ExecutionState &state, 
                                         MemoryAccessVec &rwSet) {
  bool hasMC = true;

  for (unsigned i = 0; i < rwSet.size(); i++) {
    MemoryAccess tmpAccess(rwSet[i]);
    ConstraintManager constr;     
    AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

    unsigned wordsize = rwSet[i].width / 8;
    int segSize = getSegmentSize(rwSet[i].width, 0);
    if (segSize == -1) {
      GKLEE_INFO << "CAPACITY 1.0 or 1.1: the word size can not be recognized!" 
                 << std::endl;
      break;
    }

    unsigned threadNum = GPUConfig::warpsize / 2;
    if (segSize == 256) {
      threadNum /= 2;
      segSize /= 2;
    }

    // Memory access exceeds the segment bound
    bool outOfBound = symCheckMemoryAccessOutOfSegBound(executor, state, 
                                                        rwSet[i], tmpAccess, 
                                                        segSize, threadNum, 0);
    if (outOfBound) {
      hasMC = false;
      break;
    }

    // violation of the sequential rule ... 
    ref<Expr> segSizeExpr = klee::ConstantExpr::create(segSize, rwSet[i].offset->getWidth());
    ref<Expr> remExpr = URemExpr::create(rwSet[i].offset, segSizeExpr);
    ref<Expr> idxExpr = UDivExpr::create(remExpr, klee::ConstantExpr::create(wordsize, rwSet[i].offset->getWidth()));
    ref<Expr> tidExpr = AddressSpaceUtil::constructRealThreadNumConstraint(state, rwSet[i].tid, rwSet[i].offset->getWidth());  
    ref<Expr> remTidExpr = URemExpr::create(tidExpr, klee::ConstantExpr::create(threadNum, rwSet[i].offset->getWidth()));
    ref<Expr> cond = EqExpr::create(remTidExpr, idxExpr);

    bool result = false;
    bool success = executor.solver->mustBeTrue(state, remTidExpr, result);
    if (success) {
      if (!result) { 
        hasMC = false;
        ref<Expr> tmpExpr = Expr::createIsZero(remTidExpr);
        dumpSymMemoryNonCoalescing(executor, state, tmpExpr, rwSet[i], tmpAccess, 0);
        break;
      }
    }
  }

  return hasMC;
}

bool AddressSpace::hasSymMemoryCoalescingCap0(Executor &executor, ExecutionState &state) {
  bool hasReadMC = true;
  bool hasWriteMC = true;
 
  // readSet...
  if (readSet.empty())
    GKLEE_INFO << "The read set for memory coalescing cap 1 is empty" << std::endl;
  else 
    hasReadMC = symCheckMemoryCoalescingCap0(executor, state, readSet); 
  // writeSet...
  if (writeSet.empty())
    GKLEE_INFO << "The write set for memory coalescing cap 1 is empty" << std::endl;
  else
    hasWriteMC = symCheckMemoryCoalescingCap0(executor, state, writeSet);

  return hasReadMC && hasWriteMC; 
}

static bool symCheckMemoryCoalescingCap1(Executor &executor, ExecutionState &state, 
                                         MemoryAccessVec &rwSet) {
  bool hasMC = true;

  for (unsigned i = 0; i < rwSet.size(); i++) {
    MemoryAccess tmpAccess(rwSet[i]);
    ConstraintManager constr;     
    AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

    int segSize = getSegmentSize(rwSet[i].width, 1);

    if (segSize == -1) {
      GKLEE_INFO << "CAPACITY 1.2 or 1.3: the word size can not be recognized!" 
                 << std::endl;
      break;
    }
    
    bool outOfBound = symCheckMemoryAccessOutOfSegBound(executor, state, 
                                                        rwSet[i], tmpAccess, 
                                                        segSize, GPUConfig::warpsize/2, 1);
    if (outOfBound) {
      hasMC = false;
      break;
    }

  }
  return hasMC;
}

bool AddressSpace::hasSymMemoryCoalescingCap1(Executor &executor, ExecutionState &state) {
  bool hasReadMC = true;
  bool hasWriteMC = true;
 
  // readSet...
  if (readSet.empty())
    GKLEE_INFO << "The read set for memory coalescing cap 1 is empty" << std::endl;
  else 
    hasReadMC = symCheckMemoryCoalescingCap1(executor, state, readSet); 
  // writeSet...
  if (writeSet.empty())
    GKLEE_INFO << "The write set for memory coalescing cap 1 is empty" << std::endl;
  else
    hasWriteMC = symCheckMemoryCoalescingCap1(executor, state, writeSet);

  return hasReadMC && hasWriteMC; 
}

static bool symCheckMemoryCoalescingCap2(Executor &executor, ExecutionState &state,
                                         MemoryAccessVec &rwSet) {
  bool hasMC = true;

  for (unsigned i = 0; i < rwSet.size(); i++) {
    MemoryAccess tmpAccess(rwSet[i]);
    ConstraintManager constr;     
    AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

    unsigned wordsize = rwSet[i].width / 8;
    unsigned reqThread = 128 / wordsize;
   
    bool outOfBound = symCheckMemoryAccessOutOfSegBound(executor, state, 
                                                        rwSet[i], tmpAccess,
                                                        128, reqThread, 2);   
    if (outOfBound) {
      hasMC = false;
      break;
    }
  }
  return hasMC;
}

bool AddressSpace::hasSymMemoryCoalescingCap2(Executor &executor, ExecutionState &state) {
  bool hasReadMC = true;
  bool hasWriteMC = true;
 
  // readSet...
  if (readSet.empty())
    GKLEE_INFO << "The read set for memory coalescing cap 2 is empty" << std::endl;
  else 
    hasReadMC = symCheckMemoryCoalescingCap2(executor, state, readSet); 
  // writeSet...
  if (writeSet.empty())
    GKLEE_INFO << "The write set for memory coalescing cap 2 is empty" << std::endl;
  else
    hasWriteMC = symCheckMemoryCoalescingCap2(executor, state, writeSet);

  return hasReadMC && hasWriteMC;
}

static bool accessSameMemoryRegion(Executor &executor, ExecutionState &state, 
                                   ref<Expr> &addr1, ref<Expr> &addr2) {
  ref<Expr> cond = EqExpr::create(addr1, addr2);
  bool result = false;
  bool success = executor.solver->mustBeTrue(state, cond, result);
  if (success) return result;
  return false; 
}

static void dumpSymVolatileMissing(Executor &executor, ExecutionState &state, 
                                   ref<Expr> &expr, 
                                   MemoryAccess &access1, MemoryAccess &access2,
                                   unsigned mark) {
  bool benign = false;
  std::vector<SymBlockID_t> symBlockIDs;
  std::vector<SymThreadID_t> symThreadIDs;
  SymBlockDim_t symBlockDim(0, 0, 0);
 
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  
  std::vector< ref<Expr> > offsetVec, cOffsetVec;
  offsetVec.push_back(access1.offset); 
  offsetVec.push_back(access2.offset); 

  bool success = executor.getSymbolicConfigSolution(state, expr, offsetVec, 
                                                    cOffsetVec, access1.val, 
                                                    access2.val, benign, 
                                                    symBlockIDs, symThreadIDs, 
                                                    symBlockDim); 
  
  if (success) {
    GKLEE_INFO << "++++++++++" << std::endl;
    GKLEE_INFO << "Access1: " << std::endl;
    access1.dump(executor, state, expr);
    GKLEE_INFO << "Access2: " << std::endl;
    access2.dump(executor, state, expr);
    if (mark == 0) {
      GKLEE_INFO << "Thread 1 : { <" << symBlockIDs[0].x << ", " 
                 << symBlockIDs[0].y << ", " << symBlockIDs[0].z << ">" 
                 << ", <" << symThreadIDs[0].x << ", " << symThreadIDs[0].y 
                 << ", " << symThreadIDs[0].z
                 << "> }" << " and Thread 2 : { <" << symBlockIDs[1].x << ", " 
                 << symBlockIDs[1].y << ", " << symBlockIDs[1].z << ">" 
                 << ", <" << symThreadIDs[1].x << ", " << symThreadIDs[1].y 
                 << ", " << symThreadIDs[1].z
                 << "> } has the read-write memory sharing, " 
	         << "so 'volatile' qualifier required!" << std::endl; 
    } else {
      GKLEE_INFO << "Thread 1 : { <" << symBlockIDs[0].x << ", " 
                 << symBlockIDs[0].y << ", " << symBlockIDs[0].z << ">" 
                 << ", <" << symThreadIDs[0].x << ", " << symThreadIDs[0].y 
                 << ", " << symThreadIDs[0].z
                 << "> }" << " and Thread 2 : { <" << symBlockIDs[1].x << ", " 
                 << symBlockIDs[1].y << ", " << symBlockIDs[1].z << ">" 
                 << ", <" << symThreadIDs[1].x << ", " << symThreadIDs[1].y 
                 << ", " << symThreadIDs[1].z
                 << "> } has the write-write memory sharing, " 
                 << "so 'volatile' qualifier required!" << std::endl; 
    }
    GKLEE_INFO << "Thread 1's concretized offset: " << std::endl;
    cOffsetVec[0]->dump();
    GKLEE_INFO << "Thread 2's concretized offset: " << std::endl;
    cOffsetVec[1]->dump();
    GKLEE_INFO << "++++++++++" << std::endl;
  }
}

static bool symCheckVolatileMissing(Executor &executor, ExecutionState &state, 
                                    MemoryAccess &access1, MemoryAccess &access2, 
                                    unsigned mark) {
  bool vmissing = false;

  ref<Expr> baseAddr1 = access1.mo->getBaseExpr();
  ref<Expr> baseAddr2 = access2.mo->getBaseExpr();

  if (accessSameMemoryRegion(executor, state, baseAddr1, baseAddr2)) {
    ref<Expr> conflictExpr = symCheckConflictExprs(access1.offset, access2.width, 
                                                   access2.offset, access2.width);
    ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
    ref<Expr> sameWarpExpr = AddressSpaceUtil::threadSameWarpConstraint(state, GPUConfig::warpsize);  

    bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, sameWarpExpr); 
    if (configFulfilled) {
      bool result = false;
      ExecutorUtil::addConfigConstraint(state, sameWarpExpr);
      bool success = executor.solver->mustBeFalse(state, conflictExpr, result);
      if (success) {
        if (!result) {
          vmissing = true;
          dumpSymVolatileMissing(executor, state, conflictExpr, access1, access2, mark);
        }
      }
    }
  }
  ExecutorUtil::copyBackConstraint(state);
  return vmissing;
}

bool AddressSpace::hasSymVolatileMissing(Executor &executor, ExecutionState &state) {
  bool volatilemiss = false;

  // Read-Write Sharing 
  for (MemoryAccessVec::iterator ii = readSet.begin(); ii != readSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = writeSet.begin(); jj != writeSet.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      ConstraintManager constr;     
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

      if (symCheckVolatileMissing(executor, state, *ii, tmpAccess, 0)) {
        volatilemiss = true;
        break;
      }
    }
    if (volatilemiss) break;
  }

  if (!volatilemiss) {
    // Write-Write sharing
    for (MemoryAccessVec::iterator ii = writeSet.begin(); ii != writeSet.end(); ii++) {
      MemoryAccessVec::iterator jj = ii;
      jj++;
      for (;jj != writeSet.end(); jj++) {
        MemoryAccess tmpAccess(*jj);
        ConstraintManager constr;     
        AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  
        
        if (symCheckVolatileMissing(executor, state, *ii, tmpAccess, 1)) {
          volatilemiss = true;
          break;
        }
      }
      if (volatilemiss) break;
    }
  }

  return volatilemiss;
}

static bool fenceRelation(const MemoryAccess &access1, 
                          const MemoryAccess &access2, 
                          bool withinBlock) {
  if (access1.fence == access2.fence) {
    return true; 
  } else {
    if (access1.fence == "" || access2.fence == "") {
      if (withinBlock) {
        return false;
      } else {
        bool judge1 = access1.fence == "" 
                       && access2.fence != "__threadfence_block";
        bool judge2 = access2.fence == "" 
                       && access1.fence != "__threadfence_block";
        if (judge1 || judge2)
          return false;
        else 
          return true; 
      }
    } else 
      return true; 
  }
}

static bool dumpSymRace(Executor &executor, ExecutionState &state, 
                        ref<Expr> &conflictExpr, bool benign, 
                        MemoryAccess &access1, MemoryAccess &access2, 
                        unsigned BI1, unsigned BI2) {
  std::vector<SymBlockID_t> symBlockIDs;
  std::vector<SymThreadID_t> symThreadIDs;
  SymBlockDim_t symBlockDim(0, 0, 0);
 
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symBlockIDs.push_back(SymBlockID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
  symThreadIDs.push_back(SymThreadID_t(0, 0, 0));
   
  std::vector< ref<Expr> > offsetVec, cOffsetVec;
  offsetVec.push_back(access1.offset); 
  offsetVec.push_back(access2.offset); 

  bool success = executor.getSymbolicConfigSolution(state, conflictExpr, 
                                                    offsetVec, cOffsetVec, 
                                                    access1.val, access2.val, 
                                                    benign, symBlockIDs, 
                                                    symThreadIDs, symBlockDim); 
  
  if (success) {
    GKLEE_INFO << "++++++++++" << std::endl;
    GKLEE_INFO << "Access1: " << std::endl;
    access1.dump(executor, state, conflictExpr);
    GKLEE_INFO << "Access2: " << std::endl;
    access2.dump(executor, state, conflictExpr);
    bool existRead = !(access1.is_write && access2.is_write);

    std::string tmp;
    if (existRead)
      tmp = "incur the (Actual) read-write race";
    else { 
      if (benign)
        tmp = "incur the (Benign) write-write race";
      else
        tmp = "incur the (Actual) write-write race";
    }

    GKLEE_INFO << "Thread 1 : { <" << symBlockIDs[0].x << ", " 
               << symBlockIDs[0].y << ", " << symBlockIDs[0].z << ">" 
               << ", <" << symThreadIDs[0].x << ", " << symThreadIDs[0].y 
               << ", " << symThreadIDs[0].z
               << "> }" << " and Thread 2 : { <" << symBlockIDs[1].x << ", " 
               << symBlockIDs[1].y << ", " << symBlockIDs[1].z << ">" 
               << ", <" << symThreadIDs[1].x
               << ", " << symThreadIDs[1].y << ", " << symThreadIDs[1].z
               << "> } " << tmp << std::endl;
    if (UnboundConfig)
      GKLEE_INFO << " with BlockDim {" << symBlockDim.x << ", " << symBlockDim.y
                << ", " << symBlockDim.z << "}" << std::endl;
    GKLEE_INFO << "Thread 1's concretized offset: " << std::endl;
    cOffsetVec[0]->dump();
    GKLEE_INFO << "Thread 2's concretized offset: " << std::endl;
    cOffsetVec[1]->dump();
    GKLEE_INFO << "Thread 1 resides in BI: " << BI1 
               << ", Thread 2 resides in BI: " << BI2 << std::endl;
    GKLEE_INFO << "++++++++++" << std::endl;
  }
  return benign;
}

static bool isBothAtomic(const MemoryAccess &access1, 
                         const MemoryAccess &access2) {
  return access1.isAtomic 
          && access2.isAtomic;
}

// type: 0 Same Warp, Same Block; 1, Diff Warp, Same Block; 2, Diff Block
// WW: true WW race checking; false RW race checking
static bool checkSymTwoAccessRace(Executor &executor, ExecutionState &state, 
                                  MemoryAccess &access1, MemoryAccess &access2, 
                                  unsigned type, bool WW, 
                                  unsigned BI1, unsigned BI2) {
  bool hasRace = false;

  ref<Expr> baseAddr1 = access1.mo->getBaseExpr(); 
  ref<Expr> baseAddr2 = access2.mo->getBaseExpr(); 

  if (!isBothAtomic(access1, access2)
       && accessSameMemoryRegion(executor, state, baseAddr1, baseAddr2)) {
    ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
    ref<Expr> typeExpr;
    if (type == 0) 
      typeExpr = AddressSpaceUtil::threadSameWarpConstraint(state, GPUConfig::warpsize);
    else if (type == 1)
      typeExpr = AddressSpaceUtil::threadSameBlockDiffWarpConstraint(state, GPUConfig::warpsize);
    else
      typeExpr = AddressSpaceUtil::threadDiffBlockConstraint(state);

    bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, typeExpr); 

    if (configFulfilled) {
      bool result = false;
      ExecutorUtil::addConfigConstraint(state, typeExpr);

      //state.dumpStateConstraint();
      //GKLEE_INFO << "access1 offset: " << std::endl;
      //access1.offset->dump();
      //GKLEE_INFO << "access2 offset: " << std::endl;
      //access2.offset->dump();

      ref<Expr> conflictExpr = symCheckConflictExprs(access1.offset, access1.width, 
                                                     access2.offset, access2.width);
      bool success = executor.solver->mustBeFalse(state, conflictExpr, result); 
      if (success) {
        if (!result) {
          bool benign = dumpSymRace(executor, state, conflictExpr, 
                                    true, access1, access2, BI1, BI2); 
          hasRace = !benign;
        }
      }
    }
  }
  
  ExecutorUtil::copyBackConstraint(state);
  return hasRace;
}

static bool checkParaTreeNodeTwoSides(ParaTreeNode *node, 
                                      unsigned tid1, unsigned tid2, 
                                      unsigned &which1, unsigned &which2) {
  bool diffSide = false;
  std::vector< std::set<unsigned> > divergeSet = node->divergeThreadSet;
  std::set<unsigned>::iterator found1 = divergeSet[0].find(tid1);
  std::set<unsigned>::iterator found2 = divergeSet[0].find(tid2);

  std::set<unsigned>::iterator setEnd = divergeSet[0].end();

  if (found1 != setEnd && found2 != setEnd) {
    which1 = 0;
    which2 = 0; 
  } else if (found1 != setEnd && found2 == setEnd) {
    std::set<unsigned>::iterator aFound2 = divergeSet[1].find(tid2); 
    assert(aFound2 != divergeSet[1].end() && "tid2 not found from divergence thread set 1");
    diffSide = true;
    which1 = 0;
    which2 = 1;
  } else if (found1 == setEnd && found2 != setEnd) {
    std::set<unsigned>::iterator aFound1 = divergeSet[1].find(tid1);
    assert(aFound1 != divergeSet[1].end() && "tid1 not found from divergence thread set 1");
    diffSide = true;
    which1 = 1;
    which2 = 0;
  } else {
    std::set<unsigned>::iterator aFound1 = divergeSet[1].find(tid1);
    std::set<unsigned>::iterator aFound2 = divergeSet[1].find(tid2);
    assert(aFound1 != divergeSet[1].end() && "tid1 not found from divergence thread set 1");
    assert(aFound2 != divergeSet[1].end() && "tid2 not found from divergence thread set 2");
    which1 = 1;
    which2 = 1;
  }

  return diffSide;
}

static inline bool instSeqNumInBound(unsigned tid, unsigned instSeqNum,
                                     ParaConfig &config) {
  if (config.sym_tid == tid) {
    return (config.start <= instSeqNum) && (instSeqNum <= config.end); 
  } else { 
    return true;
  }
}

static bool verifyTwoDifferentFlows(ParaTreeNode *node, MemoryAccess &access1,
                                    MemoryAccess &access2) {
  unsigned which1 = 0;
  unsigned which2 = 0;
  unsigned tid1 = access1.tid;
  unsigned tid2 = access2.tid;

  ParaTreeNode *tmp = node;
  while (tmp != NULL) {
    if (tmp->symBrType == TDC) {
      bool diffFlow = checkParaTreeNodeTwoSides(tmp, tid1, tid2, which1, which2);
      if (diffFlow) {
        std::vector<ParaConfig> &configVec = tmp->successorConfigVec;

        if (instSeqNumInBound(tid1, access1.instSeqNum, configVec[which1])
             && instSeqNumInBound(tid2, access2.instSeqNum, configVec[which2]))
          return true;
        else 
          return false;
      } else {
        std::vector<ParaTreeNode*> &treeNodeVec = tmp->successorTreeNodes;
        tmp = treeNodeVec[which1];
      }
    } else {
      std::vector<ParaTreeNode*> &tmpTreeVec = tmp->successorTreeNodes;
      tmp = tmpTreeVec[0]; 
    }
  }

  return false;
}

static bool checkTwoFlowBelongToSameParaTree(ExecutionState &state, 
                                             unsigned tid1, unsigned tid2, 
                                             unsigned &whichTree) {
  bool sameTree = false;
  std::vector<ParaTree> &paraTreeVec = state.getCurrentParaTreeVec(); 

  for (unsigned i = 0; i < paraTreeVec.size(); i++) {
    ParaTreeNode *root = paraTreeVec[i].getRootNode();

    if (root == NULL) continue;

    std::vector< std::set<unsigned> > &repSet = root->repThreadSet;

    bool found1 = false; 
    for (unsigned j = 0; j < repSet.size(); j++) {
      if (repSet[j].find(tid1) != repSet[j].end()) {
        found1 = true;
        break;
      }
    }  

    bool found2 = false; 
    for (unsigned j = 0; j < repSet.size(); j++) {
      if (repSet[j].find(tid2) != repSet[j].end()) {
        found2 = true;
        break;
      }
    }  

    if (found1 && found2) {
      sameTree = true;
      whichTree = i;
      break;
    } else if (found1 || found2) {
      break;
    }
  }

  return sameTree;
}
  
static bool checkSymPortingRace(Executor &executor, ExecutionState &state, 
                                MemoryAccess &access1, MemoryAccess &access2, 
                                bool WW, unsigned BI1, unsigned BI2) {
  bool hasRace = false;

  if (access1.tid != access2.tid) {
    // Those two accesses belong to different parametric flows ...
    unsigned which = 0;
    bool sameTree = checkTwoFlowBelongToSameParaTree(state, access1.tid, 
                                                     access2.tid, which); 
    if (sameTree) {
      std::vector<ParaTree> & paraTreeVec = state.getCurrentParaTreeVec();
      
      ParaTreeNode *node = paraTreeVec[which].getRootNode();
      bool diffFlows = verifyTwoDifferentFlows(node, access1, access2); 
      if (diffFlows) {
        MemoryAccess tmpAccess(access2);
        ConstraintManager constr;
        AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);
        hasRace = checkSymTwoAccessRace(executor, state, 
                                        access1, tmpAccess, 
                                        0, WW, BI1, BI2);
      }
    } 
  } 

  return hasRace;
}

static bool checkSymTwoAccessRacePureCS(Executor &executor, ExecutionState &state, 
                                        MemoryAccess &access1, MemoryAccess &access2, 
                                        unsigned type, bool WW, 
                                        unsigned BI1, unsigned BI2) {
  bool hasRace = false;

  ref<Expr> baseAddr1 = access1.mo->getBaseExpr(); 
  ref<Expr> baseAddr2 = access2.mo->getBaseExpr(); 
  if (!isBothAtomic(access1, access2)
       && accessSameMemoryRegion(executor, state, baseAddr1, baseAddr2)) {
    ref<Expr> configExpr = AndExpr::create(access1.accessCondExpr, access2.accessCondExpr);
    ref<Expr> typeExpr;
    ref<Expr> sameBlockExpr = AddressSpaceUtil::constructSameBlockExpr(state, Expr::Int32);
    ref<Expr> sameTidExpr = AddressSpaceUtil::constructSameThreadExpr(state, Expr::Int32);
    bool fence = false;
    if (type == 0) { // Same block ...
      typeExpr = AndExpr::create(sameBlockExpr, Expr::createIsZero(sameTidExpr));
      fence = fenceRelation(access1, access2, true);
    } else { // Different block ...
      typeExpr = Expr::createIsZero(sameBlockExpr);
      fence = fenceRelation(access1, access2, false);
    }
    bool configFulfilled = isCurrentConfigFulfilled(executor, state, configExpr, typeExpr); 

    if (configFulfilled && fence) {
      bool result = false;
      ExecutorUtil::addConfigConstraint(state, typeExpr);

      //state.dumpStateConstraint();
      //GKLEE_INFO << "access1 offset: " << std::endl;
      //access1.offset->dump();
      //GKLEE_INFO << "access2 offset: " << std::endl;
      //access2.offset->dump();

      ref<Expr> conflictExpr = symCheckConflictExprs(access1.offset, access1.width, 
                                                     access2.offset, access2.width);
      bool success = executor.solver->mustBeFalse(state, conflictExpr, result); 
      if (success) {
        if (!result) {
          bool benign = dumpSymRace(executor, state, conflictExpr, 
                                    true, access1, access2, BI1, BI2); 
          hasRace = !benign;
        }
      }
    }
  }
  
  ExecutorUtil::copyBackConstraint(state);
  return hasRace;
}

static bool checkRWRacePureCS(Executor &executor, ExecutionState &state, 
                              MemoryAccessVec &readSet, MemoryAccessVec &writeSet) {
  bool hasRace = false;
  unsigned BINum = state.BINum;

  GKLEE_INFO << "++++++++++ Read-Write race checking (Pure Canonical Schedule) ++++++++++" << std::endl;
  for (MemoryAccessVec::iterator ii = readSet.begin(); ii != readSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = writeSet.begin(); jj != writeSet.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      ConstraintManager constr;     
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  
      hasRace = checkSymTwoAccessRacePureCS(executor, state, *ii, tmpAccess, 0, false, BINum, BINum);
      if (hasRace) break;
    }
    if (hasRace) break;
  }
  
  return hasRace;
} 

static bool checkWWRacePureCS(Executor &executor, ExecutionState &state, 
                              MemoryAccessVec &writeSet) {
  bool hasRace = false;
  unsigned BINum = state.BINum;

  GKLEE_INFO << "++++++++++ Write-Write race checking (Pure Canonical Schedule) ++++++++++" << std::endl;
  for (MemoryAccessVec::iterator ii = writeSet.begin(); ii != writeSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = ii; jj != writeSet.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      ConstraintManager constr;     
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  
      hasRace = checkSymTwoAccessRacePureCS(executor, state, *ii, tmpAccess, 0, true, BINum, BINum);
      if (hasRace) break;
    }
    if (hasRace) break;
  }

  return hasRace;
}

bool AddressSpace::hasSymRaceInSharePureCS(Executor &executor, ExecutionState &state) {
  bool hasRace = false;
  bool result = false;

  // read, write race ...
  result = checkRWRacePureCS(executor, state, readSet, writeSet);
  if (result) hasRace = true;

  // write write race ...
  result = checkWWRacePureCS(executor, state, writeSet);
  if (result) hasRace = true;
  
  return hasRace;
}

bool AddressSpace::hasSymRaceInGlobalWithinBlockPureCS(Executor &executor, 
                                                       ExecutionState &state) {
  bool hasRace = false; 
  bool result = false;

  // read, write race ...
  result = checkRWRacePureCS(executor, state, readSet, writeSet);
  if (result) hasRace = true;

  // write write race ...
  result = checkWWRacePureCS(executor, state, writeSet);
  if (result) hasRace = true;
 
  symGlobalReadSets.push_back(readSet);
  symGlobalWriteSets.push_back(writeSet);  

  return hasRace;
}

static bool hasSymRaceWithinBlock(Executor &executor, ExecutionState &state, 
                                  ConstraintManager &constr, MemoryAccessVec &readSet, 
                                  MemoryAccessVec &writeSet) {
  bool hasRace = false; 
  unsigned BINum = state.BINum;
    
  // within the warp ...
  // write-write ...
  GKLEE_INFO << "++++++++++ Write-Write race checking within the warp ++++++++++" << std::endl;
  for (unsigned i = 0; i < writeSet.size(); i++) {
    MemoryAccess tmpAccess(writeSet[i]);
    AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

    if (checkSymTwoAccessRace(executor, state, writeSet[i], tmpAccess, 
                              0, true, BINum, BINum)) {
      hasRace = true;
      break;
    }
  }

  for (MemoryAccessVec::iterator ii = writeSet.begin(); 
       ii != writeSet.end(); ii++) {
    MemoryAccessVec::iterator jj = ii;
    jj++;
    for (; jj != writeSet.end(); jj++) {
      if (AddressSpaceUtil::isTwoInstIdentical(ii->instr, jj->instr) 
           && ii->instSeqNum == jj->instSeqNum) {
        MemoryAccess tmpAccess(*jj);
        AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);
        if (checkSymTwoAccessRace(executor, state, *ii, tmpAccess, 
                                  0, true, BINum, BINum)) {
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) break;
  }

  // write-write, porting race ...
  GKLEE_INFO << "++++++++++ Write-Write Porting race checking ++++++++++" << std::endl;
  for (MemoryAccessVec::iterator ii = writeSet.begin(); 
       ii != writeSet.end(); ii++) {
    MemoryAccessVec::iterator jj = ii;
    jj++;
    for (; jj != writeSet.end(); jj++) {
      if (belongToDifferentBB(*ii, *jj)) {
        if (checkSymPortingRace(executor, state, *ii, *jj, 
                                true, BINum, BINum)) {
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) break;
  }

  // read-write, porting race ...
  GKLEE_INFO << "++++++++++ Read-Write Porting race checking ++++++++++" << std::endl;
  for (MemoryAccessVec::iterator ii = writeSet.begin(); 
       ii != writeSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = readSet.begin();
         jj != readSet.end(); jj++) {
      if (belongToDifferentBB(*ii, *jj)) {
        if (checkSymPortingRace(executor, state, *ii, *jj, 
                                false, BINum, BINum)) {
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) break;
  }

  // across warps within the same block...
  // read-write ...
  GKLEE_INFO << "++++++++++ Read-Write race checking across warps ++++++++++" << std::endl;
  for (MemoryAccessVec::iterator ii = readSet.begin(); ii != readSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = writeSet.begin(); jj != writeSet.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

      // inter-warp ...
      if (checkSymTwoAccessRace(executor, state, *ii, tmpAccess, 
                                1, false, BINum, BINum)) {
        hasRace = true;
        break;
      }
    }
    if (hasRace) break;
  }

  // write write race ...
  GKLEE_INFO << "++++++++++ Write-Write race checking across warps ++++++++++" << std::endl;
  for (MemoryAccessVec::iterator ii = writeSet.begin(); ii != writeSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = ii; jj != writeSet.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  
      // inter-warp ...
      if (checkSymTwoAccessRace(executor, state, *ii, tmpAccess, 
                                1, true, BINum, BINum)) {
        hasRace = true;
        break;
      }
    }
    if (hasRace) break;
  }

  return hasRace;
} 

bool AddressSpace::hasSymRaceInShare(Executor &executor, ExecutionState &state) {
  bool hasRace = false;
  ConstraintManager constr;     
  hasRace = hasSymRaceWithinBlock(executor, state, constr, readSet, writeSet);  
  return hasRace;
}

bool AddressSpace::hasSymRaceInGlobalWithinBlock(Executor &executor, ExecutionState &state) {
  bool hasRace = false;
  ConstraintManager constr;     
  hasRace = hasSymRaceWithinBlock(executor, state, constr, readSet, writeSet);  
  symGlobalReadSets.push_back(readSet);  
  symGlobalWriteSets.push_back(writeSet);  
  return hasRace;
}

static bool checkRWAcrossBlock(Executor &executor, ExecutionState &state, 
                               MemoryAccessVec &readSet, MemoryAccessVec &writeSet, 
                               unsigned BI1, unsigned BI2, bool PureCS) {
  bool hasRace = false;
  ConstraintManager constr;
  for (MemoryAccessVec::iterator ii = readSet.begin(); ii != readSet.end(); ii++) {
    for (MemoryAccessVec::iterator jj = writeSet.begin(); jj != writeSet.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);  

      if (!PureCS)
        hasRace = checkSymTwoAccessRace(executor, state, *ii, 
                                        tmpAccess, 2, false, BI1, BI2);
      else
        hasRace = checkSymTwoAccessRacePureCS(executor, state, *ii, 
                                              tmpAccess, 1, false, BI1, BI2);
      if (hasRace) break;
    }
    if (hasRace) break;
  }

  return hasRace;
}  

static bool checkWWAcrossBlock(Executor &executor, ExecutionState &state, 
                               MemoryAccessVec &writeSet1, MemoryAccessVec &writeSet2, 
                               unsigned BI1, unsigned BI2, bool PureCS) {
  bool hasRace = false;
  ConstraintManager constr;
  for (MemoryAccessVec::iterator ii = writeSet1.begin(); ii != writeSet1.end(); ii++) {
    for (MemoryAccessVec::iterator jj = writeSet2.begin(); jj != writeSet2.end(); jj++) {
      MemoryAccess tmpAccess(*jj);
      AddressSpaceUtil::updateMemoryAccess(state, constr, tmpAccess);

      if (!PureCS) {
        // Across blocks
        hasRace = checkSymTwoAccessRace(executor, state, *ii, tmpAccess, 2, true, BI1, BI2);
      } else {
        // Across blocks
        hasRace = checkSymTwoAccessRacePureCS(executor, state, *ii, tmpAccess, 1, true, BI1, BI2);
      }
 
      if (hasRace) break;
    }
    if (hasRace) break;
  }

  return hasRace;
}  

bool AddressSpace::hasSymRaceInGlobalAcrossBlocks(Executor &executor, ExecutionState &state, bool PureCS) {
  bool hasRace = false;

  // read-write 
  for (unsigned i = 0; i < symGlobalReadSets.size(); i++) {
    if (!symGlobalReadSets[i].empty()) {
      for (unsigned j = 0; j < symGlobalWriteSets.size(); j++) {
        if (!symGlobalWriteSets[j].empty()) {
          hasRace = checkRWAcrossBlock(executor, state, symGlobalReadSets[i], 
                                       symGlobalWriteSets[j], i+1, j+1, PureCS); 
          if (hasRace) break;
        } 
      }  
      if (hasRace) break;
    }
  }

  // write-write
  for (unsigned i = 0; i < symGlobalWriteSets.size(); i++) {
    if (!symGlobalWriteSets[i].empty()) {
      for (unsigned j = i; j < symGlobalWriteSets.size(); j++) {
        if (!symGlobalWriteSets[j].empty()) {
          hasRace = checkWWAcrossBlock(executor, state, symGlobalWriteSets[i], 
                                       symGlobalWriteSets[j], i+1, j+1, PureCS); 
          if (hasRace) break;
        }
      } 
      if (hasRace) break;
    }
  }

  return hasRace;
}

bool HierAddressSpace::hasSymMemoryCoalescing(Executor &executor, ExecutionState &state, 
                                              unsigned DevCap) {
   
  std::string str;
  if (DevCap == 0)
    str = "1.0 or 1.1";
  else if (DevCap == 1)
    str = "1.2 or 1.3";
  else
    str = "2.x";

  GKLEE_INFO2 << "********** (Symbolic Configuration) Start checking coalesced device memory access at capability: " 
              << str << " **********\n";

  bool hasMC = true;

  if (DevCap == 0) {
    hasMC = deviceMemory.hasSymMemoryCoalescingCap0(executor, state);
  } else if (DevCap == 1) {
    hasMC = deviceMemory.hasSymMemoryCoalescingCap1(executor, state); 
  } else {
    hasMC = deviceMemory.hasSymMemoryCoalescingCap2(executor, state);
  }

  if (hasMC) {
    GKLEE_INFO << "********** All memory accesses are coalesced at capability: " 
               << str << " **********" << std::endl;
  } else {
    GKLEE_INFO << "********** Non coalesced memory accesses found at capability: " 
               << str << " **********" << std::endl;
  }

  return hasMC;
}

bool HierAddressSpace::hasSymBankConflict(Executor &executor, ExecutionState &state, 
                                          unsigned DevCap) {
  bool hasBC = false;

  std::vector<AddressSpace>::iterator ii = sharedMemories.begin();
  GKLEE_INFO2 << "********** (Symbolic Configuration) Start checking bank conflicts at SharedMemory " 
              << " **********\n";

  bool BC = ii->hasSymBankConflict(executor, state, DevCap);
  if (BC) hasBC = true;

  if (!BC) {
    GKLEE_INFO << "********** No Bank Conflict found at this SharedMemory **********" << std::endl;
  } else {
    GKLEE_INFO << "********** Bank Conflict found at this SharedMemory **********" << std::endl;
  } 

  return hasBC;
}

bool HierAddressSpace::hasSymWarpDivergence(ExecutionState &state) {
  GKLEE_INFO2 << "********** (Symbolic Configuration) Start checking warp divergence " 
              << " **********\n"; 
  bool hasWD = (state.tinfo.symExecuteSet.size() == 1)? false : true;
  if (hasWD) {
    GKLEE_INFO << "********** Warp Divergence found for this kernel **********" 
               << std::endl;
  } else {
    GKLEE_INFO << "********** Warp Divergence not found for this kernel **********" 
               << std::endl;
  }
  return hasWD;
}

bool HierAddressSpace::hasSymVolatileMissing(Executor &executor, 
                                             ExecutionState &state) {
  bool vmiss = false;
  std::vector<AddressSpace>::iterator ii = sharedMemories.begin();
  GKLEE_INFO2 << "********** (Symbolic Config) Start checking missed volatile at SharedMemory " 
              << " ********** \n";
  if (GPUConfig::verbose > 0) {
    ii->dump(true);
  }
    
  bool miss = ii->hasSymVolatileMissing(executor, state);
  if (miss) vmiss = true;

  if (!miss) {
    GKLEE_INFO << "********** No 'Volatile' Qualifier Missed at this SharedMemory **********" 
               << std::endl;
  } else {
    GKLEE_INFO << "********** 'Volatile' Qualifier Missed at this SharedMemory **********" 
               << std::endl;
  }
 
  return vmiss;
}

bool HierAddressSpace::hasSymRaceInShare(Executor &executor, ExecutionState &state) {
  if (GPUConfig::check_level == 0)  // skip checking
    return false;

  bool race = false;
  std::vector<AddressSpace>::iterator ii = sharedMemories.begin();
  GKLEE_INFO2 << "********** (Symbolic Config) Start checking races at SharedMemory " 
              << " **********\n";
  if (GPUConfig::verbose > 0) 
    ii->dump(true);
    
  bool hasRace = false;
  if (!SimdSchedule) {
    hasRace = ii->hasSymRaceInSharePureCS(executor, state);
  } else {
    hasRace = ii->hasSymRaceInShare(executor, state);
  }

  if (hasRace) {
    race = true;
    GKLEE_INFO << "********** races found at SharedMemory ***********" << std::endl;
  } else {
    GKLEE_INFO << "********** no races found at SharedMemory ***********" << std::endl;
  }
  
  return race;
}

bool HierAddressSpace::hasSymRaceInGlobal(Executor &executor, ExecutionState &state, bool is_end_GPU_barrier) {
  if (GPUConfig::check_level == 0)
    return false;

  GKLEE_INFO2 << "********** (Symbolic Config) Start checking races at DeviceMemory (Within Same Block) " 
              << " **********\n";

  bool race = false;
  bool hasRace = false;
  if (!SimdSchedule)
    hasRace = deviceMemory.hasSymRaceInGlobalWithinBlockPureCS(executor, state); 
  else 
    hasRace = deviceMemory.hasSymRaceInGlobalWithinBlock(executor, state);

  if (hasRace) race = true;
  
  if (is_end_GPU_barrier) {
    // skip checking
    if (GPUConfig::num_blocks == 1) {
      GKLEE_INFO2 << "********** (Symbolic Config) Checking races at DeviceMemory (Across Blocks) not allowed (number of block > 1) " 
                  << " **********\n";
    } else {
      GKLEE_INFO2 << "********** (Symbolic Config) Start checking races at DeviceMemory (Across Blocks) " 
                  << " **********\n";
      hasRace = deviceMemory.hasSymRaceInGlobalAcrossBlocks(executor, state, !SimdSchedule); 
      if (hasRace) race = true;
    }
    deviceMemory.clearSymGlobalMemoryAccessSets();
  }

  if (race) {
    GKLEE_INFO << "********* races found at DeviceMemory **********" << std::endl;
  } else {
    GKLEE_INFO << "********* no races found at DeviceMemory **********" << std::endl;
  }
  
  return race;
}

static bool determineTwoFlowInSameBlock(Executor &executor, ExecutionState &state, 
                                        ref<Expr> exp1, ref<Expr> exp2) {
  ExecutionState tmp(state); 
  ConstraintManager constr;
  ref<Expr> tExp2 = exp2;
  AddressSpaceUtil::updateBuiltInRelatedConstraint(tmp, constr, tExp2);

  ref<Expr> exp = AndExpr::create(exp1, tExp2);
  if (!isa<klee::ConstantExpr>(exp)) 
    tmp.addConstraint(exp);

  ref<Expr> diffBlockExpr = AddressSpaceUtil::threadDiffBlockConstraint(tmp);
  bool result = false;
  bool success = executor.solver->mustBeTrue(state, diffBlockExpr, result);

  if (!success)
    return false;
  else 
    return !result;
}

bool HierAddressSpace::foundMismatchBarrierInParametricFlow(ExecutionState &state, 
                                                            unsigned src, unsigned dst) {
  bool hasMismatch = false;

  if (state.tinfo.numBars[src].second != state.tinfo.numBars[dst].second) {
    GKLEE_INFO << "Flow " << src << " and Flow " << dst 
               << " encounter different barrier sequences"
               << std::endl;

    if (state.tinfo.numBars[src].second) 
      GKLEE_INFO << "Flow " << src << " hits the end of kernel, but Flow "
                 << dst << " encounters the __syncthreads() barrier!" 
                 << std::endl;  
    else 
      GKLEE_INFO << "Flow " << dst << " hits the end of kernel, but Flow "
                 << src << " encounters the __syncthreads() barrier!" 
                 << std::endl;  
    hasMismatch = true; 
  } else {
    if (state.tinfo.numBars[src].first.size() 
         != state.tinfo.numBars[dst].first.size()) {  
      // The number of barriers explored by two threads are different 
      GKLEE_INFO << "Flow " << src << " and Flow " << dst 
                 << " explore barrier sequences with different length, "
                 << "violating the property that barriers have to be textually aligned!" 
                 << std::endl;  
      hasMismatch = true;
    } else {
      std::vector<BarrierInfo> &bVec1 = state.tinfo.numBars[src].first;
      std::vector<BarrierInfo> &bVec2 = state.tinfo.numBars[dst].first;
      for (unsigned i = 0; i < bVec1.size(); i++) {
        if (bVec1[i].filePath.compare(bVec2[i].filePath) != 0 || bVec1[i].line != bVec2[i].line) {
          hasMismatch = true;
          GKLEE_INFO << "Flow " << src << " __syncthread : <" 
                     << bVec1[i].filePath << ", " << bVec1[i].line 
                     << ">" << std::endl;
          GKLEE_INFO << "Flow " << dst << " __syncthread : <" 
                     << bVec2[i].filePath << ", " << bVec2[i].line 
                     << ">" << std::endl;
          GKLEE_INFO << "Flow " << src << " and Flow " << dst 
                     << " encounter different barrier sequences, "
                     << "violating the property that barriers have to be textually aligned!" 
                     << std::endl;
          break;
        }
      }
    }
  }    
  
  return hasMismatch;
}

bool HierAddressSpace::hasMismatchBarrierInParametricFlow(Executor &executor, ExecutionState &state) {
  bool hasMismatch = false;

  for (unsigned i = 2; i < state.cTidSets.size(); i++) {
    if (!state.cTidSets[i].slotUsed)
      break;
    else {
      if (i == 2) {
        bool sameBlock = determineTwoFlowInSameBlock(executor, state, 
                                                     state.cTidSets[0].inheritExpr, 
                                                     state.cTidSets[2].inheritExpr);
        if (sameBlock) {
          hasMismatch = foundMismatchBarrierInParametricFlow(state, 0, 2); 
          if (hasMismatch) break;
        }
      } else {
        bool sameBlock = determineTwoFlowInSameBlock(executor, state, 
                                                     state.cTidSets[i-1].inheritExpr, 
                                                     state.cTidSets[i].inheritExpr);
        if (sameBlock) {
          hasMismatch = foundMismatchBarrierInParametricFlow(state, i-1, i);
          if (hasMismatch) break;
        }
      } 
    }
  } 

  return hasMismatch;
}

