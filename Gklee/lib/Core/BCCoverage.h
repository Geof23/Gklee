#ifndef KLEE_BCCOVERAGE_H
#define KLEE_BCCOVERAGE_H

#include "klee/Expr.h"
#include "llvm/Instructions.h"

// ****************************************************************************************
// byte code coverage designed for CUDA kernels
// ****************************************************************************************

namespace klee {

// coverage information 
struct CovInfo {
  typedef std::set<llvm::Instruction*> InstSet;
  typedef std::vector<InstSet> ThreadInstSet;

  InstSet coveredInsts;        // instructions in the GPU computaton
  InstSet coveredFullBrans;    // full branches in the GPU computaton

  InstSet accumInsts;          // instructions in the GPU computaton
  ThreadInstSet visitedInsts;  // visited instructions by each thread
  ThreadInstSet trueBranSet;   // untaken left branches by each thread
  ThreadInstSet falseBranSet;  // untaken right branches by each thread

  unsigned getNumVisitedInsts(unsigned i) { return visitedInsts[i].size(); };
  unsigned getNumTakenBranches(unsigned i) { return trueBranSet[i].size() + falseBranSet[i].size(); };
  void clear();
};

typedef std::vector<CovInfo> CovInfos;

// coverage information for BIs
struct BICovInfo {
  CovInfos infos;
  void clear() { infos.clear(); };

  CovInfo& getCurInfo(unsigned BI_index) { return infos[BI_index]; };
  void atBarrier(unsigned BI_index);       // start a new BI
};

// coverage information for threads
struct ThreadCovInfo {
  CovInfos infos;
  void clear() { infos.clear(); };

  CovInfo& getCurInfo(unsigned BI_index) { return infos[BI_index]; };
  void atBarrier(unsigned BI_index);       // start a new BI
};

struct ThreadCovInfoBI {
  unsigned instCovPercent;
  unsigned branCovPercent;
  ThreadCovInfoBI(unsigned _instCov, unsigned _branCov) : 
  instCovPercent(_instCov), 
  branCovPercent(_branCov) {};
}; 

typedef std::vector<ThreadCovInfoBI> TCIBVec; 

// the runtime coverage monitor
class BCCoverage {
  // byte code coverage in the per-thread level
  std::vector<BICovInfo> covInfoVec;
public:
  BCCoverage() { }
  BICovInfo& getCovInfo(unsigned kernelNum) { return covInfoVec[kernelNum-1]; };

  void initPerThreadCov();
  // process stats for a single instruction step, es is the state
  // about to be stepped
  void stepInstruction(ExecutionState &es, llvm::Instruction *inst);
  
  // called when some side of a branch has been visited.
  void coverFullBranch(ExecutionState *es);
  void coverPartialBranch(ExecutionState *es);
  void markTakenBranch(ExecutionState *es, bool is_left);
  void atBarrier(unsigned kernelNum, unsigned BI_index);
  void clear() {
    covInfoVec.clear();
  };

  // per thread coverage information
  void computePerThreadCoverage();
};

}  // end namespace klee

#endif
