
#include "Executor.h"
#include "klee/util/ExprUtil.h"
#include "klee/Statistics.h"

#include <string>
#include <iostream>
#include "CUDA.h"

// ***

using namespace klee;

// ****************************************************************************************
// path reduction
// ****************************************************************************************

void PRInfo::init(std::string& op, bool _use_dep) {
  init(op[0], op.size() > 1 ? std::atoi(op.substr(1).c_str()) : 0, _use_dep);
}

void PRInfo::init(char c, unsigned _val, bool _use_dep) {
  val = _val;
  use_dep = _use_dep;
  switch (c) {
  case 'B': mode = PER_BI; break;
  case 'T': mode = PER_THREAD; break;
  default: mode = NONE;
  }
}

std::string PRInfo::toModeStr(char c) {
  switch (c) {
  case 'B': return "Per barrier interval";
  case 'T': return "Per barrier interval & Per thread";
  default:  return "None";
  }
  return "";
}

// ***************************************************************************************

// path reduction under parametric flow 
bool PRInfo::symFullyExplore(ExecutionState &current, BICovInfo &covInfo) {
  switch (mode) {
  case PRInfo::NONE:
    return true;
  case PRInfo::PER_BI: {
    llvm::Instruction *inst = current.getPrevPC()->inst;
    unsigned tid = current.tinfo.get_cur_tid(); 
    unsigned BI_index = current.tinfo.getNumBars(tid);
    CovInfo::InstSet &is = covInfo.getCurInfo(BI_index).accumInsts;
    if (is.find(inst) == is.end()) {
      // not visited w.r.t the entire BI
      is.insert(inst);
      return true;
    } else 
      return false;
  }
  case PRInfo::PER_THREAD: {
    llvm::Instruction *inst = current.getPrevPC()->inst;
    unsigned tid = current.tinfo.get_cur_tid();
    unsigned BI_index = current.tinfo.getNumBars(tid);
    CovInfo::InstSet& isTrue = covInfo.getCurInfo(BI_index).trueBranSet[tid];
    CovInfo::InstSet& isFalse = covInfo.getCurInfo(BI_index).falseBranSet[tid];
    if (isTrue.find(inst) == isTrue.end() 
        || isFalse.find(inst) == isFalse.end()) { 
      // not visited w.r.t. this parametric flow
      return true;
    }
    else
      return false;
  }
  default: 
  ;
  }
  return true;
}

// path reduction
bool PRInfo::fullyExplore(ExecutionState &current, BICovInfo& covInfo) {
  switch (mode) {
  case PRInfo::NONE:
    return true;
  case PRInfo::PER_BI: {
    llvm::Instruction *inst = current.getPrevPC()->inst;
    unsigned tid = current.tinfo.get_cur_tid();
    unsigned BI_index = current.tinfo.getNumBars(tid);
    //std::cout << "tid: " << tid << std::endl;
    //std::cout << "PER_BI reduce before, BI_Index: " << BI_index 
    //          << ", total size: " << covInfo.infos.size() << std::endl;
    CovInfo::InstSet& is = covInfo.getCurInfo(BI_index).accumInsts;
    if (is.find(inst) == is.end()) { 
      // not visited w.r.t the entire BI
      is.insert(inst);
      return true;
    } else
      return false;
  }
  case PRInfo::PER_THREAD: {
    llvm::Instruction *inst = current.getPrevPC()->inst;
    unsigned tid = current.tinfo.get_cur_tid();
    unsigned BI_index = current.tinfo.getNumBars(tid);
    //std::cout << "BI_index:" << BI_index 
    //          << ", infos size: " << covInfo.infos.size() << std::endl;
    CovInfo::InstSet& isTrue = covInfo.getCurInfo(BI_index).trueBranSet[tid];
    CovInfo::InstSet& isFalse = covInfo.getCurInfo(BI_index).falseBranSet[tid];
    if (isTrue.find(inst) == isTrue.end() 
        || isFalse.find(inst) == isFalse.end()) { // not visited w.r.t. this thread
      return true;
    }
    else
      return false;
  }
  default:
  ;
  }
  return true;
}
