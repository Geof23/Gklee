#ifndef KLEE_CUDA_H
#define KLEE_CUDA_H

#include "ParametricTree.h"
#include "klee/Expr.h"
#include "klee/Internal/Module/KInstIterator.h"
#include "llvm/GlobalValue.h"
/* #include "llvm/Constants.h" */
/* #include "llvm/Instruction.h" */
#include "llvm/Support/CommandLine.h"
#include "Memory.h"

#include <iostream>
#include <map>
#include <vector>

namespace runtime {
  extern llvm::cl::opt<bool> UseSymbolicConfig;
}

using namespace runtime;

namespace klee {

class CUDAUtil {

  static const llvm::GlobalValue* getGlobalValue(const llvm::Value* v);
 
public:
  static GPUConfig::CTYPE getCType(const llvm::Value* v, bool is_GPU_mode);
  static GPUConfig::CTYPE getUpdatedCType(const llvm::Value* v, bool is_GPU_mode);
  static std::string getCTypeStr(GPUConfig::CTYPE tp);

};

struct BarrierInfo {
  llvm::Instruction *inst;
  std::string filePath;
  unsigned line;
  
  BarrierInfo(llvm::Instruction *_inst, std::string _filePath, unsigned _line):
  inst(_inst), filePath(_filePath), line(_line) {};
 
  BarrierInfo(const BarrierInfo &info) : inst(info.inst), 
                                         filePath(info.filePath),
                                         line(info.line) {}; 
};

struct BranchDivRegion {
  unsigned tid;
  unsigned regionStart;
  unsigned regionEnd;
 
  BranchDivRegion() {
    tid = 0;
    regionStart = regionEnd = 0;
  }

  BranchDivRegion(unsigned _tid, unsigned _regionStart, unsigned _regionEnd) :
    tid(_tid), regionStart(_regionStart), regionEnd(_regionEnd) {};

  BranchDivRegion(const BranchDivRegion &_divRegion) : tid(_divRegion.tid), 
                                                       regionStart(_divRegion.regionStart), 
                                                       regionEnd(_divRegion.regionEnd) {};
};

struct BoundedBranchDivRegion {
  unsigned side1;
  BranchDivRegion divRegion1;
  unsigned side2;
  BranchDivRegion divRegion2;

  BoundedBranchDivRegion() {
    side1 = side2 = 0;  
  }

  BoundedBranchDivRegion(unsigned _side1, BranchDivRegion _divRegion1, 
                         unsigned _side2, BranchDivRegion _divRegion2) : side1(_side1), 
                                                                         divRegion1(_divRegion1), 
                                                                         side2(_side2), 
                                                                         divRegion2(_divRegion2) {
  }

  BoundedBranchDivRegion(const BoundedBranchDivRegion &_region) : side1(_region.side1),
                                                                  divRegion1(_region.divRegion1), 
                                                                  side2(_region.side2), 
                                                                  divRegion2(_region.divRegion2) {};
};

struct BranchDivRegionVec {
  llvm::BasicBlock *whichBB; // Only useful to identify successors of switch instruction
  std::vector<BranchDivRegion> branchDivRegionVec;
    
  BranchDivRegionVec(llvm::BasicBlock *_whichBB) : whichBB(_whichBB) {};

  ~BranchDivRegionVec() {
    branchDivRegionVec.clear();
  }
};

struct BranchDivRegionSet {
  // conditional branch or switch instruction
  llvm::Instruction *brInst;
  llvm::BasicBlock *postDominator;
  bool isCondBr; // true: conditional branch, false: switch instruction 
  bool allSync;
  std::vector< std::vector<unsigned> > nonSyncSets; // threads not synchronized yet ...
  std::vector<BranchDivRegionVec> branchSets;
  bool explored;

  BranchDivRegionSet(llvm::Instruction *_brInst, llvm::BasicBlock *_postDominator, 
                     bool _isCondBr, bool _allSync, bool _explored) : 
  brInst(_brInst), postDominator(_postDominator), 
  isCondBr(_isCondBr), allSync(_allSync), explored(_explored) {};
    
  BranchDivRegionSet(const BranchDivRegionSet &tmpSet) : 
  brInst(tmpSet.brInst), postDominator(tmpSet.postDominator), 
  isCondBr(tmpSet.isCondBr), allSync(tmpSet.allSync), explored(tmpSet.explored) {
    nonSyncSets = tmpSet.nonSyncSets;
    branchSets = tmpSet.branchSets;
  }

  ~BranchDivRegionSet() {
    for (unsigned i = 0; i < branchSets.size(); i++) {
      nonSyncSets[i].clear();
    }
    nonSyncSets.clear();
  }
};

struct SymBlockID_t {
  unsigned x;
  unsigned y;
  unsigned z;
  SymBlockID_t(unsigned _x, unsigned _y, unsigned _z) 
  : x(_x), y(_y), z(_z) {};
};
   
struct SymThreadID_t {
  unsigned x;
  unsigned y;
  unsigned z;
  SymThreadID_t(unsigned _x, unsigned _y, unsigned _z) 
  : x(_x), y(_y), z(_z) {};
};

struct SymBlockDim_t {
  unsigned x;
  unsigned y;
  unsigned z;
  SymBlockDim_t(unsigned _x, unsigned _y, unsigned _z) 
  : x(_x), y(_y), z(_z) {};
};
   
// the static and dynamic information of the threads
class ThreadInfo {
  unsigned cur_tid;
  unsigned cur_bid;
  unsigned cur_wid;
  unsigned cur_warp_start_tid; 
  unsigned cur_warp_end_tid;

  unsigned sym_thread_num;
  unsigned sym_cur_bid;
  unsigned sym_cur_tid;

  clock_t start_time, end_time;

 public:
  // The concrete configuration
  // only used in parametric mode 
  typedef std::vector<KInstIterator> pcs_ty;

  pcs_ty PCs, prevPCs;

  // <number of barreris at each thread, in kernel execution>; 
  // for deadlock detection
  std::vector< std::pair<std::vector<BarrierInfo>, bool> > numBars;       

  bool kernel_call; 
  bool is_GPU_mode;
  bool is_Atomic_op;
  bool just_enter_GPU_mode;
  bool allEndKernel;
  bool escapeFromBranch;
  bool warpInBranch; // if all threads in current warp are in the branch
  std::vector<unsigned> executeSet;

  unsigned sym_warp_num;
  unsigned sym_block_num;

  // For parametric implementation ...
  unsigned sym_tdc_eval;
  bool builtInFork;
  std::vector<unsigned> symExecuteSet; 
  std::vector<unsigned> symParaTreeVec; 
  
  // links to configuration settings
  MemoryObject* thread_id_mo;
  MemoryObject* block_id_mo;
  MemoryObject* sym_bdim_mo;
  MemoryObject* sym_gdim_mo;
  ObjectState* block_size_os;
  ObjectState* grid_size_os;

 public:

  ThreadInfo();
  ThreadInfo(KInstIterator _pc);
  ThreadInfo(const ThreadInfo& info);

  inline void set_cur_tid(unsigned tid) {
    if (UseSymbolicConfig)
      sym_cur_tid = tid;
    else 
      cur_tid = tid;
  }

  inline unsigned get_cur_tid() {
    /* return cur_tid + cur_bid * GPUConfig::BlockSize[0]; */
    return (UseSymbolicConfig)? sym_cur_tid : cur_tid;
  }

  inline unsigned get_cur_bid() {
    return (UseSymbolicConfig)? sym_cur_bid : cur_bid;
  }

  inline unsigned get_num_threads() {
    return (UseSymbolicConfig) ? GPUConfig::sym_num_threads: 
                                 GPUConfig::num_threads;
  }

  inline unsigned get_cur_warp_start_tid() {
    return cur_warp_start_tid;
  }
  
  inline unsigned get_cur_warp_end_tid() {
    return cur_warp_end_tid;
  }

  void setInitPCs(KInstIterator _pc) {
    PCs.clear();
    prevPCs.clear();
    std::vector<BarrierInfo> bVec;
    for (unsigned i = 0; i < get_num_threads(); i++) {
      PCs.push_back(_pc);
      prevPCs.push_back(_pc);
      numBars.push_back(std::make_pair(bVec,false));
    }
  }

  void synchronizePCs() {
    std::vector<BarrierInfo> bVec;
    numBars[0].first = bVec;
    numBars[0].second = false;
    for (unsigned i = 1; i < PCs.size(); i++) {
      PCs[i] = PCs[0];
      prevPCs[i] = prevPCs[0];
      numBars[i].first = bVec;
      numBars[i].second = false;
    }
    for (unsigned i = PCs.size(); i < get_num_threads(); i++) {
      PCs.push_back(PCs[0]);
      prevPCs.push_back(prevPCs[0]);
      numBars.push_back(std::make_pair(bVec, false));
    }
  }

  void synchronizeBranchPCs(ParaTreeNode *current) {
    std::vector<ParaConfig> &configVec = current->successorConfigVec;
    unsigned sTid = configVec[0].sym_tid;
    for (unsigned i = 1; i < configVec.size(); i++) {
      unsigned tid = configVec[i].sym_tid;
      PCs[tid] = PCs[sTid];
      prevPCs[tid] = PCs[sTid];
    }
  }

  unsigned lastTidInCurrentWarp(std::vector<CorrespondTid> &cTidSets);

  void setInitThreadInfo(std::vector<CorrespondTid> &cTidSets) {
    cur_bid = 0;
    cur_tid = 0;
    cur_wid = 0;
    cur_warp_start_tid = 0;
    cur_warp_end_tid = lastTidInCurrentWarp(cTidSets); 
  }

  bool allThreadsInWarpEncounterBarrier(std::vector<CorrespondTid> &);
  void updateStateAfterBarriers(std::vector<CorrespondTid> &, 
                                std::vector<BranchDivRegionSet> &);
  void updateSymStateAfterBarriers(std::vector<CorrespondTid> &, ParaTree &);
  void incTid(std::vector<CorrespondTid> &, std::vector<BranchDivRegionSet> &, 
              bool &, bool &, bool &);
  void incTid();
  void dumpSymExecuteSet();
  void incParametricFlow(std::vector<CorrespondTid> &, ParaTree &, bool &);

  KInstIterator getPC() { 
    return (UseSymbolicConfig)? PCs[sym_cur_tid] : PCs[cur_tid]; 
  }

  void setPC(KInstIterator _pc) { 
    if (UseSymbolicConfig)
      PCs[sym_cur_tid] = _pc; 
    else 
      PCs[cur_tid] = _pc; 
  }

  KInstIterator getPrevPC() {
    return (UseSymbolicConfig)? prevPCs[sym_cur_tid] : prevPCs[cur_tid]; 
  }

  void setPrevPC(KInstIterator _pc) { 
    if (UseSymbolicConfig)
      prevPCs[sym_cur_tid] = _pc; 
    else 
      prevPCs[cur_tid] = _pc;
  }

  void incPC() {
    if (!UseSymbolicConfig) {
      prevPCs[cur_tid] = PCs[cur_tid];
      ++(PCs[cur_tid]);
    } else {
      prevPCs[sym_cur_tid] = PCs[sym_cur_tid];
      ++(PCs[sym_cur_tid]);
    }
  }

  unsigned getNumBars(unsigned tid) { return numBars[tid].first.size(); }

  inline bool at_last_tid() {
    return cur_tid == get_num_threads() - 1;
  }

  bool foundMismatchBarrierWithinTheBlock(std::vector<CorrespondTid> &, unsigned, 
                                          unsigned, unsigned);
  void synchronizeBarrierInfo(ParaTreeNode *); 
  bool hasMismatchBarrierInSym(std::vector<CorrespondTid> &);
  bool hasMismatchBarrier(std::vector<CorrespondTid> &);

  double getGPUTime() {
    end_time = clock();
    return ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
  }

};  // end class

} // end namespace klee

#endif
