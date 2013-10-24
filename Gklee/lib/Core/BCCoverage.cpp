#include "Executor.h"
#include "CUDA.h"
#include "BCCoverage.h"
#include "llvm/Support/CommandLine.h"

#include <string>
#include <iostream>

// ***
using namespace llvm;
using namespace klee;

namespace runtime {
  extern cl::opt<bool> BCCoverageLevel; 
  extern cl::opt<bool> UseSymbolicConfig;  
}

using namespace runtime;

// ****************************************************************************************
// Bytecode coverage for CUDA
// ****************************************************************************************

void CovInfo::clear() {
  coveredInsts.clear();
  coveredFullBrans.clear();
  accumInsts.clear();
  visitedInsts.clear();
  trueBranSet.clear();
  falseBranSet.clear();
}

// Insert the CovInfo w.r.t each barrier interval
void BICovInfo::atBarrier(unsigned BI_index) {
  if (infos.size() <= BI_index) {
    CovInfo info;
    CovInfo::InstSet s;
    unsigned num_threads = (UseSymbolicConfig)? GPUConfig::sym_num_threads : 
                                                GPUConfig::num_threads;
    for (unsigned i = 0; i < num_threads; i++) {
      info.visitedInsts.push_back(s);
      info.trueBranSet.push_back(s);
      info.falseBranSet.push_back(s);
    }
    infos.push_back(info);
  }
}

/*******************************************************************************
  Runtime coverage information
 *******************************************************************************/

void BCCoverage::initPerThreadCov() {
  BICovInfo covInfo;
  if (covInfo.infos.size() == 0) {
    covInfo.atBarrier(0);
  }
  covInfoVec.push_back(covInfo);
}

void BCCoverage::atBarrier(unsigned kernelNum, unsigned BI_index) { 
  BICovInfo &covInfo = covInfoVec[kernelNum-1]; 
  covInfo.atBarrier(BI_index); 
}

void BCCoverage::stepInstruction(ExecutionState &es, llvm::Instruction *inst) {
  if (es.tinfo.is_GPU_mode) {
    unsigned kNum = es.getKernelNum()-1;
    unsigned tid = es.tinfo.get_cur_tid();
    unsigned BI_index = es.tinfo.getNumBars(tid);
    CovInfo::InstSet& is = covInfoVec[kNum].getCurInfo(BI_index).visitedInsts[tid];
    is.insert(inst);
    covInfoVec[kNum].getCurInfo(BI_index).coveredInsts.insert(inst);
  }
}

void BCCoverage::coverFullBranch(ExecutionState *es) {
  if (es->tinfo.is_GPU_mode) {
    unsigned kNum = es->getKernelNum()-1;
    llvm::Instruction *inst = es->getPrevPC()->inst;
    unsigned tid = es->tinfo.get_cur_tid();
    unsigned BI_index = es->tinfo.getNumBars(tid);
    covInfoVec[kNum].getCurInfo(BI_index).coveredFullBrans.insert(inst);
  }
}

void BCCoverage::markTakenBranch(ExecutionState *es, bool is_left) {
  if (es->tinfo.is_GPU_mode) {
    unsigned kNum = es->getKernelNum()-1;
    llvm::Instruction *inst = es->getPrevPC()->inst;
    unsigned tid = es->tinfo.get_cur_tid();
    unsigned BI_index = es->tinfo.getNumBars(tid);
    // std::cout << "branch inst: " << *inst << std::endl;
    // std::cout << "thread " << tid << "'s branch: " << (is_left ? "left" : "right") 
    // 	      << std::endl;
    if (is_left)
      covInfoVec[kNum].getCurInfo(BI_index).trueBranSet[tid].insert(inst);
    else
      covInfoVec[kNum].getCurInfo(BI_index).falseBranSet[tid].insert(inst);
  }
}

static void dumpACPTCoverage(BICovInfo &covInfo) {
  unsigned avg_inst_cov = 0;
  unsigned avg_bran_cov = 0;

  CovInfo::InstSet tbran_s, fbran_s;
  CovInfo::InstSet tCoveredInsts;

  CovInfo::ThreadInstSet tBrSets, fBrSets, tCovSets;
  CovInfo::InstSet is;
  for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
    tBrSets.push_back(is); 
    fBrSets.push_back(is);
    tCovSets.push_back(is);
  }

  std::cout << "---------- ACPT ----------\n";
  for (unsigned bi = 0; bi < covInfo.infos.size(); bi++) {
    // compute the branch number
    for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
      CovInfo::InstSet& s = covInfo.infos[bi].trueBranSet[i];
      for (CovInfo::InstSet::iterator ii = s.begin(); ii != s.end(); ii++) {
        tbran_s.insert(*ii);
        tBrSets[i].insert(*ii);
      }

      s = covInfo.infos[bi].falseBranSet[i];
      for (CovInfo::InstSet::iterator ii = s.begin(); ii != s.end(); ii++) {
	fbran_s.insert(*ii);	  
        fBrSets[i].insert(*ii);
      }

      s = covInfo.infos[bi].visitedInsts[i];
      for (CovInfo::InstSet::iterator ii = s.begin(); ii != s.end(); ii++) {
        tCovSets[i].insert(*ii);
      }
    }
     
    CovInfo::InstSet &covInsts = covInfo.infos[bi].coveredInsts;
    for (CovInfo::InstSet::iterator ii = covInsts.begin(); 
         ii != covInsts.end(); ii++)
      tCoveredInsts.insert(*ii);
  }

  unsigned total_branches = tbran_s.size() + fbran_s.size();
  unsigned total_insts = tCoveredInsts.size();

  for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
    unsigned k = tCovSets[i].size() * 100 / total_insts;
    std::cout << "thread " << i << " line: " << k << "%; ";
    avg_inst_cov += k;

    k = (tBrSets[i].size() + fBrSets[i].size()) * 100 / total_branches;
    std::cout << " branch: " << k << "%\n";
    avg_bran_cov += k;
  }

  avg_inst_cov /= GPUConfig::num_threads;
  avg_bran_cov /= GPUConfig::num_threads;

  std::cout << "Coverage for all threads: \n";
  std::cout << "Lines: " << avg_inst_cov << "%\n";
  std::cout << "Taken branches: " << avg_bran_cov << "%\n";
}

static void dumpPERBICoverage(BICovInfo &covInfo) {
  unsigned avg_inst_cov = 0;
  unsigned avg_bran_cov = 0;
    
  // multiple BIs but only one thread
  std::vector<TCIBVec> TCIBSet;
  TCIBVec vec;  
  CovInfo::InstSet inst_s, tbran_s, fbran_s;

  std::cout << "---------- PER_BI ----------\n";
  for (unsigned bi = 0; bi < covInfo.infos.size(); bi++) {
    for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
      CovInfo::InstSet& instSet = covInfo.infos[bi].visitedInsts[i];
      for (CovInfo::InstSet::iterator ii = instSet.begin(); 
           ii != instSet.end(); ii++)
        inst_s.insert(*ii);

      CovInfo::InstSet& tSet = covInfo.infos[bi].trueBranSet[i];
      for (CovInfo::InstSet::iterator ii = tSet.begin(); 
           ii != tSet.end(); ii++)
        tbran_s.insert(*ii);

      CovInfo::InstSet& fSet = covInfo.infos[bi].falseBranSet[i];
      for (CovInfo::InstSet::iterator ii = fSet.begin(); 
           ii != fSet.end(); ii++)
        fbran_s.insert(*ii);	  
    }
    std::cout << "In this BI, the total lines of code accessed by all threads (Without duplicate): "
              << inst_s.size() << std::endl;
    std::cout << "In this BI, the total branches accessed by all threads (Without duplicate): "
              << tbran_s.size() + fbran_s.size() << std::endl;

    for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
      unsigned instNum = covInfo.infos[bi].visitedInsts[i].size();
      unsigned instCov = 0;
      if (inst_s.size())
        instCov = (100 * instNum) / inst_s.size();
      else 
        instCov = 0; 
         
      unsigned branNum = covInfo.infos[bi].trueBranSet[i].size() + 
                         covInfo.infos[bi].falseBranSet[i].size();  
      unsigned branCov = 0;
      if (tbran_s.size() + fbran_s.size())
        branCov = (100 * branNum) / (tbran_s.size() + fbran_s.size());
      else 
        branCov = 0;

      vec.push_back(ThreadCovInfoBI(instCov, branCov));
    }
    avg_inst_cov += inst_s.size();
    avg_bran_cov += tbran_s.size() + fbran_s.size(); 
    inst_s.clear();
    tbran_s.clear();
    fbran_s.clear();
    TCIBSet.push_back(vec);
  }
         
  unsigned avgInstCov = 0;
  unsigned avgBranCov = 0;
  unsigned maxInstCov = 0;
  unsigned maxBranCov = 0;
  unsigned maxInstThread = 0;
  unsigned maxBranThread = 0;

  for (unsigned i = 0; i < GPUConfig::num_threads; i++) {  
    unsigned threadAvgInstCov = 0; 
    unsigned threadAvgBranCov = 0; 

    for (unsigned j = 0; j < TCIBSet.size(); j++) {
      if (TCIBSet[j][i].instCovPercent) {
        threadAvgInstCov += TCIBSet[j][i].instCovPercent;  
      }
      if (TCIBSet[j][i].branCovPercent) {
        threadAvgBranCov += TCIBSet[j][i].branCovPercent;
      }
    } 

    threadAvgInstCov /= TCIBSet.size();  
    threadAvgBranCov /= TCIBSet.size();  
    std::cout << "Across all barriers, the thread " << i << " line: "
              << threadAvgInstCov << "%, branch: " << threadAvgBranCov
              << "%" << std::endl; 

    avgInstCov += threadAvgInstCov;
    avgBranCov += threadAvgBranCov;

    if (threadAvgInstCov > maxInstCov) {
      maxInstCov = threadAvgInstCov;
      maxInstThread = i;
    }

    if (threadAvgBranCov > maxBranCov) {
      maxBranCov = threadAvgBranCov;
      maxBranThread = i;
    }
  }
  avgInstCov /= GPUConfig::num_threads;
  avgBranCov /= GPUConfig::num_threads;
    
  std::cout << "Among all threads, the average line cov: " 
            << avgInstCov << "%" << std::endl; 
  std::cout << "Among all threads, the average branch cov: " 
            << avgBranCov << "%" << std::endl; 
  std::cout << "Among all threads, the maximum line cov: " 
            << maxInstCov << "%" << ", and the maximum thread: " 
            << maxInstThread << std::endl; 
  std::cout << "Among all threads, the maximum branch cov: " << maxBranCov << "%"
            << ", and the maximum thread: " << maxBranThread << std::endl; 
  /***********************************/
  std::cout << "Coverage for BIs: \n";
  std::cout << "Covered lines: " << avg_inst_cov << "\n";
  std::cout << "Taken branches: " << avg_bran_cov << "\n";
}

/* byte code coverage per thread
   The original KLEE method:
   instruction coverage: 100.*SCov/(SCov+SUnc),
   branch coverage: 100.*(2*BFull+BPart)/(2.*BTot).
   We use a different method here
*/
void BCCoverage::computePerThreadCoverage() {
  if (BCCoverageLevel && !UseSymbolicConfig) {
    for (unsigned i = 0; i < covInfoVec.size(); i++) {
      dumpACPTCoverage(covInfoVec[i]);
      dumpPERBICoverage(covInfoVec[i]);
    }
  }
}
