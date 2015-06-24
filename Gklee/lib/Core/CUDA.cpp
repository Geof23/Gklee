#include "CUDA.h"
#include "AddressSpace.h"
#include "klee/logging.h"

#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "Memory.h"

#include <iostream>

using namespace llvm;
using namespace klee;

///
unsigned GPUConfig::GridSize[3] = {1,1,1};
// Initially, two symbolic blocks .. 
unsigned GPUConfig::SymGridSize[3] = {2,1,1};
// Currently, use the GTX 480 standard configuration
unsigned GPUConfig::SymMaxGridSize[3] = {65535, 65535, 65535};

unsigned GPUConfig::BlockSize[3] = {1,1,1};
// Initially, two symbolic thread, 
unsigned GPUConfig::SymBlockSize[3] = {1,1,1};
// Currently, use the GTX 480 standard configuration
unsigned GPUConfig::SymMaxBlockSize[3] = {2048, 2048, 128};

unsigned GPUConfig::num_blocks = GPUConfig::GridSize[0] * GPUConfig::GridSize[1] * GPUConfig::GridSize[2];
unsigned GPUConfig::block_size = GPUConfig::BlockSize[0] * GPUConfig::BlockSize[1] * GPUConfig::BlockSize[2];
unsigned GPUConfig::num_threads = GPUConfig::block_size * GPUConfig::num_blocks;

unsigned GPUConfig::sym_num_blocks = GPUConfig::SymGridSize[0] * GPUConfig::SymGridSize[1] * GPUConfig::SymGridSize[2];
unsigned GPUConfig::sym_block_size = GPUConfig::SymBlockSize[0] * GPUConfig::SymBlockSize[1] * GPUConfig::SymBlockSize[2];
unsigned GPUConfig::sym_num_threads = GPUConfig::sym_block_size * GPUConfig::sym_num_blocks;

unsigned GPUConfig::warpsize = 32;
unsigned GPUConfig::check_level = 0;
unsigned GPUConfig::verbose = 0;

namespace runtime {
  extern cl::opt<bool> Emacs;
}

//********************************************************************************************

const llvm::GlobalValue* CUDAUtil::getGlobalValue(const llvm::Value* v) {
  if (const llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(v)) {
    if (ce->getOpcode() == llvm::Instruction::GetElementPtr) {
      // std::cout << "arg0: " << ce->getOperand(0)->getNameStr() << "\n";
      return dyn_cast<llvm::GlobalValue> (ce->getOperand(0));
    }
  }
  else if (const llvm::GlobalValue* e = dyn_cast<llvm::GlobalValue>(v)) {
    return e;
  }
  return NULL;
}

GPUConfig::CTYPE CUDAUtil::getCType(const llvm::Value* v, bool is_GPU_mode) {
  // first handle the case where v is a composite expression
  if (const llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(v)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast) {
      // std::cout << *ce << " : " << *(ce->getOperand(0)) << std::endl;
      return getCType(ce->getOperand(0), is_GPU_mode);
    }
  }
  // now v is a atomic expression
  const llvm::GlobalValue* v1 = getGlobalValue(v);
  if (v1) {
    if (v1->hasSection()) {
      std::string s = v1->getSection();
      if (s == "__shared__")
	return GPUConfig::SHARED;
      else if (s == "__device__")
	return GPUConfig::DEVICE;
      else if (s == "__constant__")
        return GPUConfig::DEVICE;
    }
    else if (v1->getName().str() == "threadIdx") {
      return GPUConfig::LOCAL;
    }
    else if (v1->getName().str() == "blockIdx") {
      return GPUConfig::SHARED;
    } 
    else {
      return GPUConfig::HOST;
    }
  }
  
  if (is_GPU_mode)
    return GPUConfig::LOCAL;
  else
    return GPUConfig::HOST;
}

GPUConfig::CTYPE CUDAUtil::getUpdatedCType(const llvm::Value* v, bool is_GPU_mode) {
  // first handle the case where v is a composite expression
  if (const llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(v)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast) {
      // std::cout << *ce << " : " << *(ce->getOperand(0)) << std::endl;
      return getUpdatedCType(ce->getOperand(0), is_GPU_mode);
    }
  }
  // now v is a atomic expression
  const llvm::GlobalValue* v1 = getGlobalValue(v);
  if (v1) {
    if (v1->hasSection()) {
      std::string s = v1->getSection();
      if (s == "__shared__")
	return GPUConfig::SHARED;
      else if (s == "__device__")
	return GPUConfig::DEVICE;
      else if (s == "__constant__")
        return GPUConfig::DEVICE;
    }
    else if (v1->getName().str() == "threadIdx") {
      return GPUConfig::LOCAL;
    }
    else if (v1->getName().str() == "blockIdx") {
      return GPUConfig::SHARED;
    }
    else if (v1->getName().str() == "blockDim") {
      return GPUConfig::HOST;
    }
    else if (v1->getName().str() == "girdDim") {
      return GPUConfig::HOST;
    }
    else {
      return GPUConfig::UNKNOWN;
    }
  }
  
  return GPUConfig::UNKNOWN;
}

std::string CUDAUtil::getCTypeStr(GPUConfig::CTYPE tp) {
  switch (tp) {
  case GPUConfig::LOCAL :
    return "LOCAL";
  case GPUConfig::SHARED :
    return "SHARED";
  case GPUConfig::DEVICE :
    return "DEVICE";
  case GPUConfig::HOST :
    return "HOST";
  case GPUConfig::CONSTANT :
    return "CONSTANT";
  default:
    return "UNKNOWN";
  }
}

//*****************************************************************

ThreadInfo::ThreadInfo() : cur_tid(0), cur_bid(0), cur_wid(0), cur_warp_start_tid(0), 
                           cur_warp_end_tid(0), sym_cur_bid(0), sym_cur_tid(0), 
                           kernel_call(false), is_GPU_mode(false), 
                           is_Atomic_op(0),
                           just_enter_GPU_mode(false), 
                           allEndKernel(false), escapeFromBranch(false), 
                           warpInBranch(false), sym_warp_num(0), 
                           sym_block_num(0), sym_tdc_eval(0), 
                           builtInFork(false), thread_id_mo(0), 
                           block_id_mo(0), sym_bdim_mo(0), 
                           block_size_os(0), grid_size_os(0) {
  start_time = clock();
  end_time = start_time;
}

ThreadInfo::ThreadInfo(KInstIterator _pc) : cur_tid(0), cur_bid(0), cur_wid(0), 
                                            cur_warp_start_tid(0), 
                                            cur_warp_end_tid(0), 
                                            sym_cur_bid(0), sym_cur_tid(0), 
                                            kernel_call(false), 
                                            is_GPU_mode(false), 
                                            is_Atomic_op(0),
                                            just_enter_GPU_mode(false), 
                                            allEndKernel(false), 
                                            escapeFromBranch(false), 
                                            warpInBranch(false),
                                            sym_warp_num(0), sym_block_num(0),
                                            sym_tdc_eval(0),
                                            builtInFork(false), thread_id_mo(0), 
                                            block_id_mo(0), sym_bdim_mo(0), 
                                            block_size_os(0), grid_size_os(0) {
  start_time = clock();
  end_time = start_time;
  std::vector<BarrierInfo> bVec; 
  for (unsigned i = 0; i < get_num_threads(); i++) {
    PCs.push_back(_pc);
    prevPCs.push_back(_pc);
    numBars.push_back(std::make_pair(bVec, false));
  }
}

ThreadInfo::ThreadInfo(const ThreadInfo& info) : cur_tid(info.cur_tid), cur_bid(info.cur_bid), 
                                                 cur_wid(info.cur_wid), cur_warp_start_tid(info.cur_warp_start_tid), 
                                                 cur_warp_end_tid(info.cur_warp_end_tid), 
                                                 sym_cur_bid(info.sym_cur_bid), sym_cur_tid(info.sym_cur_tid), 
                                                 start_time(info.start_time), end_time(info.end_time), 
                                                 PCs(info.PCs), prevPCs(info.prevPCs), numBars(info.numBars),
                                                 kernel_call(info.kernel_call), 
                                                 is_GPU_mode(info.is_GPU_mode), 
                                                 is_Atomic_op(info.is_Atomic_op),
                                                 just_enter_GPU_mode(info.just_enter_GPU_mode),
                                                 allEndKernel(info.allEndKernel),
                                                 escapeFromBranch(info.escapeFromBranch), 
                                                 warpInBranch(info.warpInBranch),
                                                 executeSet(info.executeSet), 
                                                 sym_warp_num(info.sym_warp_num), 
                                                 sym_block_num(info.sym_block_num),
                                                 sym_tdc_eval(info.sym_tdc_eval),
                                                 builtInFork(info.builtInFork), 
                                                 symExecuteSet(info.symExecuteSet),
                                                 symParaTreeVec(info.symParaTreeVec) {
  thread_id_mo = info.thread_id_mo;
  block_id_mo = info.block_id_mo;
  block_size_os = info.block_size_os;
  grid_size_os = info.grid_size_os;
  sym_bdim_mo = info.sym_bdim_mo;
  sym_gdim_mo = info.sym_gdim_mo;
}

unsigned ThreadInfo::lastTidInCurrentWarp(std::vector<CorrespondTid>& cTidSets) {
  unsigned i = cur_warp_start_tid;
  if (i == GPUConfig::num_threads-1) return i;

  while (cTidSets[i].warpNum == cTidSets[i+1].warpNum) {
    i++;
    if (i == GPUConfig::num_threads-1) break;
  }
  return i;
}  

static bool threadsInWarpEncounterImplicitBarrier(std::vector<CorrespondTid> &cTidSets, 
                                                  unsigned sTid, unsigned eTid) {
  for (unsigned i = sTid; i <= eTid; ++i) {
    if (!cTidSets[i].syncEncounter)
      return false;
  }  
  return true;
}

static bool existThreadsInWarpEncounterSyncthreads(std::vector<CorrespondTid> &cTidSets, 
                                                   unsigned sTid, unsigned eTid) {
  for (unsigned i = sTid; i <= eTid; ++i) {
    if (cTidSets[i].barrierEncounter)
      return true;
  }
  return false;
}

static void findMismatchExplicitAndImplicitBarriers(std::vector<CorrespondTid> &cTidSets,
                                                    unsigned sTid, unsigned eTid) {
  std::vector<unsigned> sameTid;
  std::vector<unsigned> diffTid;

  for (unsigned i = sTid; i <= eTid; i++) {
    if (cTidSets[i].barrierEncounter) {
      sameTid.push_back(i);
    } else {
      diffTid.push_back(i);
    }
  }

  std::cout << "Threads (";
  for (unsigned i = 0; i < sameTid.size(); i++) {
    if (i == sameTid.size() - 1)
      std::cout << sameTid[i]; 
    else 
      std::cout << sameTid[i] << ", "; 
  }
  std::cout << ") wait at the explicit __syncthreads()" << std::endl;

  std::cout << "Threads (";
  for (unsigned i = 0; i < diffTid.size(); i++) {
    if (i == diffTid.size() - 1)
      std::cout << diffTid[i]; 
    else 
      std::cout << diffTid[i] << ", "; 
  }
  std::cout << ") wait at the re-convergence point, not __syncthreads()" << std::endl;
  
  sameTid.clear();
  diffTid.clear();
}

static bool findNearestBranchDivRegion(std::vector<BranchDivRegionSet> &branchDivRegionSets, unsigned &brNum) { 
  bool findBrNum = false;
  unsigned size = branchDivRegionSets.size();
  for (int i = size-1; i >= 0; i--) {
    if (!branchDivRegionSets[i].allSync) {
      brNum = i;
      findBrNum = true;
      break; 
    }
  }
  return findBrNum;
}

bool ThreadInfo::allThreadsInWarpEncounterBarrier(std::vector<CorrespondTid> &cTidSets) {
  for (unsigned i = cur_warp_start_tid; i <= cur_warp_end_tid; i++) {
    if (!cTidSets[i].barrierEncounter)
      return false;
  }
  return true;
}

static bool allThreadsSynchronizedForSpecificBranch(std::vector< std::vector<unsigned> > &nonSyncSets) {
  for (unsigned i = 0; i < nonSyncSets.size(); i++) {
    if (nonSyncSets[i].size() != 0)
      return false;
  }
  return true;
}

static bool allThreadsSynchronizedForAllBranches(std::vector<BranchDivRegionSet> &branchDivRegionSets) {
  for (unsigned i = 0; i < branchDivRegionSets.size(); i++) {
    if (!branchDivRegionSets[i].allSync)
      return false;
  }
  return true;
}

void ThreadInfo::updateStateAfterBarriers(std::vector<CorrespondTid> &cTidSets, 
                                          std::vector<BranchDivRegionSet> &branchDivRegionSets) {
  for (int i = branchDivRegionSets.size()-1; i >= 0; i--) {
    if (allThreadsSynchronizedForSpecificBranch(branchDivRegionSets[i].nonSyncSets))
      branchDivRegionSets[i].allSync = true;
  }
  // All Threads are synchronized through explicit or implicit barriers 
  if (allThreadsSynchronizedForAllBranches(branchDivRegionSets)) {
    for (unsigned i = cur_warp_start_tid; i <= cur_warp_end_tid; i++)
      cTidSets[i].inBranch = false; 

    warpInBranch = false;
    escapeFromBranch = true;
    executeSet.clear();
  }
}

void ThreadInfo::incTid(std::vector<CorrespondTid> &cTidSets, 
                        std::vector<BranchDivRegionSet> &branchDivRegionSets, 
                        bool &newBI, bool &moveToNextWarp, bool &deadlockFound) {
  Gklee::Logging::enterFunc( std::string("newBI?:") + std::to_string( newBI ), __PRETTY_FUNCTION__ );
  unsigned block_size = GPUConfig::block_size;
  if (is_GPU_mode) {
    // Find next thread which does not encounter synchronization ...
    bool ateb = allThreadsInWarpEncounterBarrier(cTidSets);
    Gklee::Logging::outItem( std::to_string( ateb ), "all threads in warp encounter bar?" );
    if (ateb) {
      deadlockFound = foundMismatchBarrierWithinTheBlock(cTidSets, cur_warp_start_tid, 
                                                         cur_warp_end_tid, cur_bid);
      warpInBranch = false;
      moveToNextWarp = true;
      if (cur_warp_end_tid == GPUConfig::num_threads - 1) {
        GKLEE_INFO << "Moving to next barrier from warp " 
                   << cur_wid << std::endl;
        cur_tid = 0;
        cur_wid = 0;
        cur_warp_start_tid = 0;
        cur_warp_end_tid = lastTidInCurrentWarp(cTidSets);
        newBI = true;
	Gklee::Logging::outItem( std::string( "lastTID:" ) + 
				 std::to_string( cur_warp_end_tid ),
				 "finished warp" );
      } else {
        cur_wid = cur_wid + 1; 
        GKLEE_INFO << "Moving from warp " << cur_wid-1 
                   << " to warp " << cur_wid << std::endl;
        for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
          if (cTidSets[i].warpNum == cur_wid) {
            cur_warp_start_tid = i;
            cur_warp_end_tid = lastTidInCurrentWarp(cTidSets);
            break;
          }
        }
        //std::cout << "cur_wid: " << cur_wid << ", cur_warp_start_tid: "
        //          << cur_warp_start_tid << ", cur_warp_end_tid: "
        //          << cur_warp_end_tid << std::endl; 
        cur_tid = cur_warp_start_tid;
	Gklee::Logging::outItem( std::string( "new curTid:" ) +
				 std::to_string( cur_tid ),
				 "advancing cur warp" );
      }
    } else {
      if (warpInBranch) {
	Gklee::Logging::outItem( std::string( "executeSetsize:" ) +
				 std::to_string( executeSet.size() ),
				 "warp in branch" );
        if (executeSet.size() == 0) {
          unsigned brNum = 0;
          bool findDivRegion = findNearestBranchDivRegion(branchDivRegionSets, brNum);
          if (findDivRegion) {
            //std::cout << "Br Inst: " << std::endl;
            //branchDivRegionSets[brNum].brInst->dump();
            std::vector< std::vector<unsigned> > &nonSyncSets = branchDivRegionSets[brNum].nonSyncSets;
            for (unsigned i = 0; i < nonSyncSets.size(); i++) {
              if (nonSyncSets[i].size() != 0) {
                executeSet = nonSyncSets[i];
                break;
              }
            }
          }
          assert(findDivRegion && "Divergence Region Not Found, GKLEE's problem\n");
        }
        /*std::cout << "The execute set in incTid: " << std::endl;
        for (unsigned j = 0; j < executeSet.size(); j++) {
          std::cout << executeSet[j] << " ";
        }
        std::cout << std::endl;*/

        std::vector<unsigned>::iterator it = executeSet.begin();
        cur_tid = *it;
        executeSet.erase(it); 
      } else {
        unsigned i = 0;
        unsigned start_pos;
        if (escapeFromBranch) 
          start_pos = cur_warp_start_tid;
        else 
          start_pos = cur_tid + 1;

        for (i = start_pos; i <= cur_warp_end_tid; i++) {
          if (!cTidSets[i].barrierEncounter)
            break; 
        }
        // If cur_tid is the last tid of the current warp. 
        if (i == cur_warp_end_tid+1) {
          for (unsigned j = cur_warp_start_tid; j <= cur_warp_end_tid; j++) {
            if (!cTidSets[j].barrierEncounter) {
              cur_tid = j;
              break;
            }
          }
        }
        else cur_tid = i;

        // potential deadlock ...
        if (threadsInWarpEncounterImplicitBarrier(cTidSets, cur_warp_start_tid, cur_warp_end_tid)) {
          if (existThreadsInWarpEncounterSyncthreads(cTidSets, cur_warp_start_tid, cur_warp_end_tid)) {
            deadlockFound = true;
            findMismatchExplicitAndImplicitBarriers(cTidSets, cur_warp_start_tid, cur_warp_end_tid);
          }
          for (unsigned i = cur_warp_start_tid; i <= cur_warp_end_tid; i++)
            cTidSets[i].syncEncounter = false;
        }
      }
    }
    if (escapeFromBranch) escapeFromBranch = false;
  }
  else  // only one thread
    cur_tid = 0;

  cur_bid = cur_tid / block_size;
  Gklee::Logging::outItem( std::string( "cur tid:bid;" ) +
			  std::to_string( cur_tid ) + ":" + 
			   std::to_string( cur_bid ),
			   "on exit" );
  Gklee::Logging::exitFunc();
}

void ThreadInfo::incTid() {
  unsigned block_size = UseSymbolicConfig? GPUConfig::sym_block_size :
                                           GPUConfig::block_size;
  if (is_GPU_mode) {
    if (cur_tid == get_num_threads() - 1)
      cur_tid = 0;
    else
      cur_tid++;
  }
  else  // only one thread
    cur_tid = 0;
  cur_bid = cur_tid / block_size;
}

static bool allSymbolicThreadsInSetEncounterBarrier(std::vector<CorrespondTid> &cTidSets, 
                                                    std::vector<unsigned> &set) {
  for (unsigned i = 0; i < set.size(); i++) {
    unsigned tid = set[i];
    if (!cTidSets[tid].barrierEncounter)
      return false; 
  }
  return true;
}

static void findSymbolicTidFromParaTree(std::vector<CorrespondTid> &cTidSets, 
                                        ParaTree &paraTree, unsigned &sym_cur_tid) {
  ParaTreeNode *current = paraTree.getCurrentNode();
  bool tidFound = false;
 
  if (current) {
    ParaTreeNode *tmp = current;
    while (!tidFound && tmp != NULL) {
      if (!tmp->allSync) {
        std::vector<ParaConfig> &configVec = tmp->successorConfigVec;
        for (unsigned i = 0; i < configVec.size(); i++) {
          if (!configVec[i].syncEncounter) {
            sym_cur_tid = configVec[i].sym_tid;
            tmp->whichSuccessor = i;
            tidFound = true;
            break;
          }
        }
      } else tmp = tmp->parent;
    }
  } else {
    tidFound = true; // keep using the current tid
  }

  assert(tidFound && "sym tid not found!");
}

void ThreadInfo::dumpSymExecuteSet() {
  for (unsigned i = 0; i < symExecuteSet.size(); i++) {
    std::cout << symExecuteSet[i] << " "; 
  }
  std::cout << std::endl;
}

void ThreadInfo::incParametricFlow(std::vector<CorrespondTid> &cTidSets, 
                                   ParaTree &paraTree, bool &newBI) {
  // explore a symbolic thread, and a new thread required ...
  if (allSymbolicThreadsInSetEncounterBarrier(cTidSets, symExecuteSet)) {
    // The end of this barrier interval ...
    symExecuteSet.clear();
    symParaTreeVec.erase(symParaTreeVec.begin());
    if (symParaTreeVec.empty()) {
      sym_cur_tid = 0;
      symExecuteSet.push_back(sym_cur_tid);
      warpInBranch = false; 
      newBI = true; // start a new BI ...
    } else {
      sym_cur_tid = symParaTreeVec[0];
      symExecuteSet.push_back(sym_cur_tid);
      warpInBranch = false;
      newBI = false;
    }
  } else {
    if (sym_tdc_eval) {
      ParaTreeNode *current = paraTree.getCurrentNode();
      std::vector<ParaConfig> &configVec = current->successorConfigVec;
      unsigned size = configVec.size();
      if (configVec[size-1].sym_tid == sym_cur_tid) {
        sym_tdc_eval = 0;
        sym_cur_tid = configVec[0].sym_tid;
      } else {
        unsigned i = 0;
        for (; i < configVec.size(); i++) {
          if (configVec[i].sym_tid == sym_cur_tid)
            break;
        }
        assert((i+1) < configVec.size() && "i is wrong");
        sym_cur_tid = configVec[i+1].sym_tid;
      }
    } else {
      // Use execute set to choose thread, always the first thread ...
      findSymbolicTidFromParaTree(cTidSets, paraTree, sym_cur_tid);
    }
    //std::cout << "The new tid: " << sym_cur_tid << std::endl;
    //setConcreteConfig(paraTree);
  } 
}

bool ThreadInfo::foundMismatchBarrierWithinTheBlock(std::vector<CorrespondTid> &cTidSets, 
                                                    unsigned bStart, unsigned bEnd, unsigned bNum) {
  bool hasMismatch = false;
  for (unsigned i = bStart+1; i <= bEnd; i++) {
    if (numBars[i].second != numBars[i-1].second) {
      // Definitely proves the barrier sequences explored by those 
      // two threads are different ...
      GKLEE_INFO << "Thread " << i << " and Thread " << i-1
                 << " encounter different barrier sequences"
                 << std::endl;
      if (numBars[i].second) 
        GKLEE_INFO << "Thread" << i << " hits the end of kernel, but Thread "
                   << i-1 << " encounters the __syncthreads() barrier!" 
                   << std::endl;  
      else 
        GKLEE_INFO << "Thread" << i-1 << " hits the end of kernel, but Thread "
                   << i << " encounters the __syncthreads() barrier!" 
                   << std::endl;  

      if (Emacs) std::cout << std::flush << "emacs:dbs:" 
			   << cTidSets[i].rBid << ":" <<
  		 cTidSets[i].rTid << ":::" << cTidSets[i-1].rBid << ":" <<
		 cTidSets[i-1].rTid << "::" << std::endl;
      hasMismatch = true;
      break;
    } else {
      if (numBars[i].first.size() != numBars[i-1].first.size()) {  
        // The number of barriers explored by two threads are different 
        GKLEE_INFO << "Thread " << i << " and Thread " << i-1
                   << " explore barrier sequences with different length, "
                   << "violating the property that barriers have to be textually aligned!" 
                   << std::endl;  
        if (Emacs) std::cout << std::flush << "emacs:bsdl:" 
			     << cTidSets[i].rBid << ":" << 
  	  	   cTidSets[i].rTid << ":::" << cTidSets[i-1].rBid << ":" <<
		   cTidSets[i-1].rTid << "::" << std::endl;
        hasMismatch = true;
        break;
      } else {
        std::vector<BarrierInfo> &bVec1 = numBars[i-1].first;
        std::vector<BarrierInfo> &bVec2 = numBars[i].first;
        for (unsigned j = 0; j < bVec1.size(); j++) {
          if (bVec1[j].filePath.compare(bVec2[j].filePath) != 0 || bVec1[j].line != bVec2[j].line) {
            hasMismatch = true;
            GKLEE_INFO << "Thread " << i-1 << " __syncthread : <" 
                       << bVec1[j].filePath << ", " << bVec1[j].line << ">" << std::endl;
            GKLEE_INFO << "Thread " << i << " __syncthread : <" 
                       << bVec2[j].filePath << ", " << bVec2[j].line << ">" << std::endl;
            GKLEE_INFO << "Thread " << i << " and Thread " << i-1
                       << " encounter different barrier sequences, "
                       << "violating the property that barriers have to be textually aligned!" 
                       << std::endl;  
	    if (Emacs) {
	      std::cout << std::flush << "emacs:dbs:" << cTidSets[i].rBid << ":" 
	     	        << cTidSets[i].rTid 
		        << bVec2[j].filePath << ":" << bVec2[j].line 
		        << ":"
		        << cTidSets[i-1].rBid << ":"<< cTidSets[i-1].rTid << ":" 
		        << bVec1[j].filePath << ":" << bVec1[j].line << std::endl;
	    }
            break;
          }
        }
        if (hasMismatch) break;
      }
    }
  }

  return hasMismatch;
}

void ThreadInfo::synchronizeBarrierInfo(ParaTreeNode *current) {
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  unsigned sTid = configVec[0].sym_tid;
  for (unsigned i = 1; i < configVec.size(); i++) {
    unsigned tid = configVec[i].sym_tid;
    numBars[tid] = numBars[sTid];  
  } 
}

// check barrier mismatches (deadlocks)
bool ThreadInfo::hasMismatchBarrier(std::vector<CorrespondTid> &cTidSets) {
  unsigned realTid = cTidSets[cur_tid].rTid;
  // The first thread in each block will skip checking.
  if (GPUConfig::check_level == 0 
       || realTid == 0)  // skip checking
    return false;

  if (GPUConfig::verbose > 1) {
    std::cout << "\nStart checking mismatch barriers... \n";
    std::cout << "Barrier counts:\n";
    for (unsigned i = 0; i < get_num_threads(); i++) {
      std::cout << "<" << numBars[i].first.size() << "," << 
	(numBars[i].second ? "true" : "false") << "> ";
    }
    std::cout << std::endl;
  }

  bool hasDeadlock = false;
  for (unsigned i = 0; i < GPUConfig::num_blocks; i++) {
    unsigned start = i * GPUConfig::block_size;
    unsigned end = (i+1) * GPUConfig::block_size - 1;

    if (foundMismatchBarrierWithinTheBlock(cTidSets, start, end, i)) {
      hasDeadlock = true;
      break;
    }
  }
  return hasDeadlock;
}
