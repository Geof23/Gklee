#include "AddressSpace.h"
#include "Memory.h"
#include "TimingSolver.h"

#include "klee/Expr.h"
#include "klee/TimerStatIncrementer.h"
#include "klee/Constraints.h"
#include <iostream>
#include <fstream>

#include "CUDA.h"
#include <assert.h>

using namespace llvm;
using namespace klee;

namespace runtime {
  extern cl::opt<bool> DumpDetailSolution;
  extern cl::opt<bool> Emacs;
}

using namespace runtime;

static void updateWarpDefVecConsider(WarpDefVec &, MemoryAccessVec &, 
                                     std::vector<CorrespondTid> &, bool);

static void updateWarpDefVec(WarpDefVec &, MemoryAccessVec &, 
                             std::vector<CorrespondTid> &, unsigned, bool);

static void dumpTmpMemorySet(MemoryAccessVec &);

// return true if bank conflict exists...
static bool checkReadBankConflictExprsCap1x(klee::ref<Expr> &addr1, klee::ref<Expr> &addr2, 
                                            Executor &executor, ExecutionState &state, 
                                            unsigned BankNum, klee::ref<Expr> &bankSeq, 
                                            klee::ref<Expr> &bcCond, unsigned &queryNum) {
  klee::ref<Expr> origEq = EqExpr::create(addr1, addr2);
  bool result = false;
  bool unknown = false;
  bool success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, origEq, result, unknown);
  queryNum++;
  if (success) {
    if (result) return false; // broadcast...
  }
 
  klee::ref<Expr> tmpExpr = NeExpr::create(addr1, addr2);
  klee::ref<Expr> bankSize = ConstantExpr::create(BankNum * 4, addr1->getWidth());
  klee::ref<Expr> wordSize = ConstantExpr::create(4, addr1->getWidth());
  klee::ref<Expr> a1 = UDivExpr::create(URemExpr::create(addr1, bankSize), wordSize);
  klee::ref<Expr> a2 = UDivExpr::create(URemExpr::create(addr2, bankSize), wordSize);
  klee::ref<Expr> expr = EqExpr::create(a1, a2);
  klee::ref<Expr> andExpr = AndExpr::create(tmpExpr, expr);

  success = AddressSpaceUtil::evaluateQueryMustBeFalse(executor, state, andExpr, result, unknown);
  queryNum++;
  if (success) {
    if (!result) {
      if (unknown) bcCond = AndExpr::create(bcCond, andExpr); 
      // try the first one ...
      bankSeq = a1;
    }
    return !result;
  }
  return false; 
}

// return true if bank conflict exists...
static bool checkWriteBankConflictExprsCap1x(klee::ref<Expr> &addr1, klee::ref<Expr> &addr2, 
                                             Executor &executor, ExecutionState &state, 
                                             unsigned BankNum, klee::ref<Expr> &bankSeq, 
                                             klee::ref<Expr> &bcCond, unsigned &queryNum) {
  klee::ref<Expr> bankSize = ConstantExpr::create(BankNum * 4, addr1->getWidth());
  klee::ref<Expr> wordSize = ConstantExpr::create(4, addr1->getWidth());
  klee::ref<Expr> a1 = UDivExpr::create(URemExpr::create(addr1, bankSize), wordSize);
  klee::ref<Expr> a2 = UDivExpr::create(URemExpr::create(addr2, bankSize), wordSize);
  klee::ref<Expr> expr = EqExpr::create(a1, a2);

  bool result = false;
  bool unknown = false;
  bool success = AddressSpaceUtil::evaluateQueryMustBeFalse(executor, state, expr, result, unknown);
  queryNum++;
 
  if (success) {
    if (!result) {
      if (unknown) bcCond = AndExpr::create(bcCond, expr);
      // try the first one ...
      bankSeq = a1;
    }
    return !result;
  }
  return false;
}

static bool checkBankConflictExprsCap2x(klee::ref<Expr> &addr1, klee::ref<Expr> &addr2, 
                                        Executor &executor, ExecutionState &state,
                                        unsigned BankNum, klee::ref<Expr> &bankSeq, 
                                        klee::ref<Expr> &bcCond, unsigned &queryNum) {
  // Eliminate the same word ...
  klee::ref<Expr> wordSize = ConstantExpr::create(4, addr1->getWidth());
  klee::ref<Expr> a1 = UDivExpr::create(addr1, wordSize);
  klee::ref<Expr> a2 = UDivExpr::create(addr2, wordSize);
  klee::ref<Expr> expr = EqExpr::create(a1, a2);

  bool result = false;
  bool unknown = false;
  bool success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, expr, result, unknown);
  queryNum++;
  if (success) {
    if (result) return false; // broadcast...
  }

  klee::ref<Expr> tmpExpr = NeExpr::create(a1, a2);
  klee::ref<Expr> bankSize = ConstantExpr::create(BankNum * 4, addr1->getWidth());
  klee::ref<Expr> b1 = UDivExpr::create(URemExpr::create(addr1, bankSize), wordSize);
  klee::ref<Expr> b2 = UDivExpr::create(URemExpr::create(addr2, bankSize), wordSize);

  klee::ref<Expr> andExpr = AndExpr::create(tmpExpr, EqExpr::create(b1, b2));
  success = AddressSpaceUtil::evaluateQueryMustBeFalse(executor, state, andExpr, result, unknown);
  queryNum++;
  if (success) {
    if (!result) {
      if (unknown) bcCond = AndExpr::create(bcCond, andExpr);
      // try the first one...
      bankSeq = b1;
    }
    return !result;
  }
  return false;
}

static void dumpBankConflictCap1x(Executor &executor, ExecutionState &state, 
                                  klee::ref<Expr> &bcCond, const MemoryAccess &ma1, 
                                  const MemoryAccess &ma2, 
                                  bool isWrite, klee::ref<Expr> &bankSeq) {
  std::string str = (isWrite)? "W-W bank conflict" : "R-R bank conflict";
  GKLEE_INFO2 << "********** CAPACITY 1.x Bank Conflict **********" << std::endl;

  GKLEE_INFO << "Threads " << ma1.tid << " and " << ma2.tid
             << " incur a " << str;
  if (DumpDetailSolution) {
    std::cout << " on bank " ;
    bankSeq->print(std::cout);
  }
  std::cout << std::endl;
  ma1.dump(executor, state, bcCond);
  ma2.dump(executor, state, bcCond);
  
  GKLEE_INFO2 << "************************************************" << std::endl;
}
 
static bool checkBankConflictCap1x(Executor &executor, ExecutionState &state, 
                                   MemoryAccessVec &rwSet, std::vector<CorrespondTid> &cTidSets, 
                                   std::vector<InstAccessSet> &instAccessSets,
                                   std::vector<klee::RefDivRegionSetVec> &divRegionSets,
                                   std::vector<SameInstVec> &sameInstSets,
                                   bool isWrite, klee::ref<Expr> &bcCond, WarpDefVec &bcWDVec, 
                                   unsigned &queryNum) {
  MemoryAccessVec tmpRWSet = rwSet;
  bool hasBC = false;

  while (!tmpRWSet.empty()) {
    MemoryAccessVec bcRWSet; 

    AddressSpaceUtil::constructTmpRWSet(executor, state, tmpRWSet, bcRWSet, cTidSets, 
                                        instAccessSets, divRegionSets, sameInstSets, 
                                        GPUConfig::warpsize/2);

    updateWarpDefVecConsider(bcWDVec, bcRWSet, cTidSets, isWrite);

    for (MemoryAccessVec::const_iterator ii = bcRWSet.begin(); 
         ii != bcRWSet.end(); ii++) {
      klee::ref<Expr> addr1 =  ii->offset;
      MemoryAccessVec::const_iterator jj = ii;
      jj++;
      for (; jj != bcRWSet.end(); jj++) {
        klee::ref<Expr> addr2 = jj->offset;
        klee::ref<Expr> bankSeq;
        bool hasConflict = isWrite ?
                        checkWriteBankConflictExprsCap1x(addr1, addr2, executor, state, GPUConfig::warpsize/2, bankSeq, bcCond, queryNum) :
                        checkReadBankConflictExprsCap1x(addr1, addr2, executor, state, GPUConfig::warpsize/2, bankSeq, bcCond, queryNum);
        if (hasConflict) {
          dumpBankConflictCap1x(executor, state, bcCond, 
                                *ii, *jj, isWrite, bankSeq);
          hasBC = true;
          break;
        }
      }

      if (hasBC) break;
    }

    if (hasBC) {
      updateWarpDefVec(bcWDVec, bcRWSet, cTidSets, 1, isWrite);
      tmpRWSet.clear();
      bcRWSet.clear();
      break;
    }
    
    bcRWSet.clear();
  }

  return hasBC;
}

static void dumpBankConflictCap2x(Executor &executor, ExecutionState &state, 
                                  klee::ref<Expr> bcCond, const MemoryAccess &ma1, 
                                  const MemoryAccess &ma2, 
                                  bool isWrite, klee::ref<Expr> &bankSeq) {
  if(Emacs) AddressSpace::dumpEmacsInfoVect(ma1.bid, ma2.bid, ma1.tid, 
					    ma2.tid, ma1.instr, ma2.instr, 
					    isWrite ? "wwbc" : "rrbc");
  std::string str = (isWrite)? "W-W bank conflict" : "R-R bank conflict";
  GKLEE_INFO2 << "********** CAPACITY 2.x Bank Conflict **********" << std::endl;

  GKLEE_INFO << "Threads " << ma1.tid << " and " << ma2.tid
             << " incur a " << str;
  if (DumpDetailSolution) {
    std::cout << " on bank ";
    bankSeq->print(std::cout);
  }
  std::cout << std::endl;
  ma1.dump(executor, state, bcCond);
  ma2.dump(executor, state, bcCond);

  GKLEE_INFO2 << "************************************************" << std::endl;
}
 
static bool checkBankConflictCap2x(Executor &executor, ExecutionState &state,
                                   MemoryAccessVec &rwSet, std::vector <CorrespondTid> &cTidSets, 
                                   std::vector<InstAccessSet> &instAccessSets,
                                   std::vector<klee::RefDivRegionSetVec> &divRegionSets, 
                                   std::vector<SameInstVec> &sameInstSets,
                                   bool isWrite, klee::ref<Expr> &bcCond, WarpDefVec &bcWDVec, 
                                   unsigned &queryNum) {
  MemoryAccessVec tmpRWSet = rwSet;
  bool hasBC = false;

  while (!tmpRWSet.empty()) {
    MemoryAccessVec bcRWSet;

    AddressSpaceUtil::constructTmpRWSet(executor, state, tmpRWSet, bcRWSet, cTidSets,
                      instAccessSets, divRegionSets, sameInstSets,
                      GPUConfig::warpsize);
    updateWarpDefVecConsider(bcWDVec, bcRWSet, cTidSets, isWrite);
    bool hasViolation = false;

    for (MemoryAccessVec::const_iterator ii = bcRWSet.begin(); ii != bcRWSet.end(); ii++) {
      klee::ref<Expr> addr1 = ii->offset;
      MemoryAccessVec::const_iterator jj = ii;
      jj++;
      for (; jj != bcRWSet.end(); jj++) {
        klee::ref<Expr> addr2 = jj->offset;
        klee::ref<Expr> bankSeq;
        bool hasConflict = checkBankConflictExprsCap2x(addr1, addr2, executor, state, 
                                                       GPUConfig::warpsize, bankSeq, 
                                                       bcCond, queryNum); 

        if (hasConflict) {
          dumpBankConflictCap2x(executor, state, bcCond, 
                                *ii, *jj, isWrite, bankSeq);
          hasViolation = true;
          hasBC = true;
        }
      }
      if (hasBC) break; 
    }

    if (hasViolation) {
      updateWarpDefVec(bcWDVec, bcRWSet, cTidSets, 1, isWrite);
      break;
    }

    bcRWSet.clear();
  }
  return hasBC;
}

bool AddressSpace::hasBankConflict(Executor &executor, ExecutionState &state,
                                   unsigned capability, std::vector<CorrespondTid> &cTidSets, 
                                   std::vector<InstAccessSet> &instAccessSets,
                                   std::vector<klee::RefDivRegionSetVec> &divRegionSets,
                                   std::vector<SameInstVec> &sameInstVecSets,
                                   klee::ref<Expr> &bcCond, WarpDefVec &bcWDVec, 
                                   bool &Consider, unsigned &queryNum) {
  bool hasReadBC = false;
  bool hasWriteBC = false;

  if (readSet.empty() && writeSet.empty()) Consider = false;
  else Consider = true; 
  
  if (capability == 0 || capability == 1) {
    // 1.x
    // ReadSet first...
    if (readSet.empty()) {
      GKLEE_INFO << "The read set is empty in bank conflict checking for capability 1.x"
                 << std::endl; 
    } else {
      hasReadBC = checkBankConflictCap1x(executor, state, readSet, cTidSets, 
                                         instAccessSets, divRegionSets, sameInstVecSets, 
                                         false, bcCond, bcWDVec, queryNum);  
    }
    // WriteSet ...
    if (writeSet.empty()) {
      GKLEE_INFO << "The write set is empty in bank conflict checking for capability 1.x"
                 << std::endl; 
    } else {
      hasWriteBC = checkBankConflictCap1x(executor, state, writeSet, cTidSets, 
                                          instAccessSets, divRegionSets, sameInstVecSets, 
                                          true, bcCond, bcWDVec, queryNum);
    }
  } else {
    // 2.x
    // ReadSet first...
    if (readSet.empty()) {
      GKLEE_INFO << "The read set is empty in bank conflict checking for capability 2.x"
                 << std::endl; 
    } else {
      hasReadBC = checkBankConflictCap2x(executor, state, readSet, cTidSets, 
                                         instAccessSets, divRegionSets, sameInstVecSets, 
                                         false, bcCond, bcWDVec, queryNum);
    }
    // WriteSet ...
    if (writeSet.empty()) {
      GKLEE_INFO << "The write set is empty in bank conflict checking for capability 2.x"
                 << std::endl; 
    } else {
      hasWriteBC = checkBankConflictCap2x(executor, state, writeSet, cTidSets, 
                                          instAccessSets, divRegionSets, sameInstVecSets, 
                                          true, bcCond, bcWDVec, queryNum);
    }
  }
 
  return (hasReadBC || hasWriteBC);
}

//****************************************************************************************************

// For testing purpose
static void dumpTmpMemorySet(MemoryAccessVec &tmpRWSet) {
  MemoryAccessVec::const_iterator vit;
  GKLEE_INFO << "++++++++++ Dump tmp memory access vector: ++++++++++"
             << std::endl;
  for (vit = tmpRWSet.begin(); vit != tmpRWSet.end(); vit++) {
    vit->dump();
  }
  GKLEE_INFO << std::endl;
}

static int getSegmentSize(Expr::Width width, unsigned &wordsize, unsigned capability) {
  wordsize = width / 8;

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

static void dumpMemoryCoalescingCap0Fail(Executor &executor, ExecutionState &state, 
                                         klee::ref<Expr> noMCCond, MemoryAccess &access1, 
                                         MemoryAccess &access2, MemoryAccessVec &tmpRWSet,  
                                         klee::ref<Expr> &lbound, klee::ref<Expr> &ubound, 
                                         unsigned reason, unsigned warpNum, unsigned wordsize) {
  GKLEE_INFO2 << "********** CAPACITY 1.0 or 1.1 half-warp ( " << warpNum << " ) **********" << std::endl;
  if(Emacs) AddressSpace::dumpEmacsInfoVect(access1.bid, access2.bid,
					    access1.tid, access2.tid, 
					    access1.instr, access2.instr, "noncoalesc");

  if (GPUConfig::verbose > 0) {
    GKLEE_INFO << "The lower bound of the segment accessed by this half-warp: " << std::endl;
    lbound->dump();
    GKLEE_INFO << "The upper bound of the segment accessed by this half-warp: " << std::endl;
    ubound->dump();
    dumpTmpMemorySet(tmpRWSet);

    GKLEE_INFO << "thread (" << access1.bid << "," << access1.tid << ") info: "
               << std::endl;
    access1.dump(executor, state, noMCCond);
    if (reason == 0) {
      GKLEE_INFO << "thread (" << access2.bid << "," << access2.tid << ") info: "
                 << std::endl;
      access2.dump(executor, state, noMCCond);
    }
  }

  GKLEE_INFO << "The word size in this half warp : " << wordsize << std::endl; 

  if (reason) {
    GKLEE_INFO << "Address accessed by the thread (" << access1.bid << "," 
               << access1.tid << ") exceeds this segment bound." << std::endl;
  }
  else {
    GKLEE_INFO << "Thread (" << access1.bid << "," << access1.tid << ")" 
               << "'s memory region violates the sequential rule: threads must access words sequentially" 
                << std::endl;
  }
  GKLEE_INFO << "So 16 memory transactions needed!" << std::endl;

  GKLEE_INFO2 << "**********************************************************************" << std::endl;
}

void dumpMemoryCoalescingCap0Success(MemoryAccessVec &tmpRWSet, klee::ref<Expr> &lbound, klee::ref<Expr> &ubound,
                                     unsigned warpNum, unsigned wordsize) {
  GKLEE_INFO2 << "********** CAPACITY 1.0 or 1.1 half-warp ( " << warpNum << " ) **********" << std::endl;

  if (GPUConfig::verbose > 0) {
    GKLEE_INFO << "The lower bound of the segment accessed by this half-warp: " << std::endl;
    lbound->dump();
    GKLEE_INFO << "The upper bound of the segment accessed by this half-warp: " << std::endl;
    ubound->dump();
    dumpTmpMemorySet(tmpRWSet);
  }

  GKLEE_INFO << "The word size in this half warp : " << wordsize << std::endl; 

  GKLEE_INFO << "All memory accesses reside in one segment, so one memory transaction starting at ";
  lbound->dump();
  GKLEE_INFO << std::endl;

  GKLEE_INFO2 << "************************************************************************" << std::endl;
}

static void updateWarpDefVecConsider(WarpDefVec &vec, MemoryAccessVec &tmpRWVec, 
                                     std::vector<CorrespondTid> &cTidSets, bool isWrite) {
   // Find which warp it is
   unsigned tid = tmpRWVec.begin()->tid;
   unsigned warpId = cTidSets[tid].warpNum;    
   vec[warpId].consider = true;

   if (isWrite) vec[warpId].instWriteTotal++;
   else vec[warpId].instReadTotal++;
}

static void updateWarpDefVec(WarpDefVec &vec, MemoryAccessVec &tmpRWVec, 
                             std::vector<CorrespondTid> &cTidSets, unsigned set, bool isWrite) {
   // Find which warp it is
   unsigned tid = tmpRWVec.begin()->tid;
   unsigned warpId = cTidSets[tid].warpNum;    
   vec[warpId].occur = set;

   if (isWrite) vec[warpId].instWriteOccur++;
   else vec[warpId].instReadOccur++;
}

static bool checkMemoryCoalescingCap0Size(Executor &executor, ExecutionState &state, 
                                          MemoryAccessVec &tmpRWSet, std::vector<CorrespondTid> &cTidSets, 
                                          unsigned halfWarpNum, unsigned segSize, 
                                          unsigned threadNum, unsigned wordsize, 
                                          klee::ref<Expr> &noMCCond, unsigned &queryNum) {
  klee::ref<Expr> lbound;
  klee::ref<Expr> ubound;
  klee::ref<Expr> baseAddr = tmpRWSet[0].mo->getBaseExpr();
  klee::ref<Expr> segSizeExpr = ConstantExpr::create(segSize, baseAddr->getWidth());
  klee::ref<Expr> segNumExpr;
  bool result = false;
  bool success = false;
  bool hasViolation = false;
  klee::ref<Expr> cond;
  unsigned segWarpNum = 0; // The number of different segments all threads in a half 
                           // warp will access 

  for (unsigned i = 0; i < tmpRWSet.size(); i++) {
    klee::ref<Expr> tmpSegNumExpr = UDivExpr::create(tmpRWSet[i].offset, segSizeExpr);
    if (segWarpNum == 0) {
      segWarpNum++;
      segNumExpr = tmpSegNumExpr; 
      lbound = MulExpr::create(segSizeExpr, segNumExpr); 
      ubound = AddExpr::create(lbound, segSizeExpr); 
    } else {
      klee::ref<Expr> cond = EqExpr::create(segNumExpr, tmpSegNumExpr);
      bool unknown = false;

      success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
      queryNum++;
      if (success) {
        // memory access exceeds the segment bound...
        if (!result) {
          hasViolation = true;
          if (unknown) noMCCond = AndExpr::create(noMCCond, Expr::createIsZero(cond));
          dumpMemoryCoalescingCap0Fail(executor, state, noMCCond, 
                                       tmpRWSet[i], tmpRWSet[i], tmpRWSet, 
                                       lbound, ubound, 1, halfWarpNum, wordsize);
          segWarpNum++;
          break;
        }
      }
    }
    // Violation of the sequential rule... 
    klee::ref<Expr> remExpr = URemExpr::create(tmpRWSet[i].offset, segSizeExpr);  
    klee::ref<Expr> idxExpr = UDivExpr::create(remExpr, ConstantExpr::create(wordsize, baseAddr->getWidth()));
    klee::ref<Expr> tidExpr = ConstantExpr::create(cTidSets[tmpRWSet[i].tid].rTid, baseAddr->getWidth());
    klee::ref<Expr> remTidExpr = URemExpr::create(tidExpr, ConstantExpr::create(threadNum, baseAddr->getWidth()));
    cond = EqExpr::create(remTidExpr, idxExpr);
          
    bool unknown = false;
    success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
    queryNum++;
    if (success) {
      if (!result) {
        hasViolation = true;
        if (unknown) noMCCond = AndExpr::create(noMCCond, Expr::createIsZero(cond));
        if (i == tmpRWSet.size()-1)
          dumpMemoryCoalescingCap0Fail(executor, state, noMCCond, 
                                       tmpRWSet[i], tmpRWSet[i], tmpRWSet, 
                                       lbound, ubound, 0, halfWarpNum, wordsize);
        else 
          dumpMemoryCoalescingCap0Fail(executor, state, noMCCond, 
                                       tmpRWSet[i], tmpRWSet[i+1], tmpRWSet, 
                                       lbound, ubound, 0, halfWarpNum, wordsize);
        break;
      }
    }
  } 
  // The memory coalescing fulfills...
  if (!hasViolation) {
    dumpMemoryCoalescingCap0Success(tmpRWSet, lbound, ubound, halfWarpNum, wordsize);
  }

  return hasViolation;
}

static bool checkMemoryCoalescingCap0(Executor &executor, ExecutionState &state, 
                                      MemoryAccessVec &rwSet, std::vector<CorrespondTid> &cTidSets, 
                                      std::vector<InstAccessSet> &instAccessSets,
                                      std::vector<klee::RefDivRegionSetVec> &divRegionSets,
                                      std::vector<SameInstVec> &sameInstVecSets,
                                      klee::ref<Expr> &noMCCond, WarpDefVec &nomcWDVec, 
                                      unsigned &queryNum, bool isWrite) {
  bool hasCoalescing = true;
  unsigned halfWarpNum = 0; // The sequence number of half warps
   
  while(!rwSet.empty()) {
    bool hasViolation = false;
    MemoryAccessVec tmpRWSet; 
    AddressSpaceUtil::constructTmpRWSet(executor, state, rwSet, tmpRWSet, cTidSets, 
                      instAccessSets, divRegionSets, sameInstVecSets, GPUConfig::warpsize/2);
    updateWarpDefVecConsider(nomcWDVec, tmpRWSet, cTidSets, isWrite);

    // Constructed the tmpRWSet...
    MemoryAccessVec::const_iterator tii = tmpRWSet.begin();
    halfWarpNum++;
    unsigned wordsize = 0;

    int segSize = getSegmentSize(tii->width, wordsize, 0);
    if (segSize > 0) {
      if (segSize == 256) {
        // The word size is 16 bytes, needs two 128 segments
        MemoryAccessVec vec1;
        MemoryAccessVec vec2;   
        
        MemoryAccessVec::iterator vi = tmpRWSet.begin();
        vec1.push_back(MemoryAccess(*vi));
        
        for (unsigned i = 1; i<tmpRWSet.size(); i++) {
          MemoryAccess tmpAccess(tmpRWSet[i]);
          if (cTidSets[tmpRWSet[i].tid].rTid/8 == cTidSets[vec1[0].tid].rTid/8) 
            vec1.push_back(tmpAccess);
          else 
            vec2.push_back(tmpAccess);
        }

        if (vec1.size() > 0) {
          hasViolation = checkMemoryCoalescingCap0Size(executor, state, vec1, cTidSets, 
                                                       halfWarpNum, 128, GPUConfig::warpsize/4, 
                                                       wordsize, noMCCond, queryNum);
          if (!hasViolation) {
            if (vec2.size() > 0)
              hasViolation = checkMemoryCoalescingCap0Size(executor, state, vec2, cTidSets, 
                                                           halfWarpNum, 128, GPUConfig::warpsize/4,
                                                           wordsize, noMCCond, queryNum);
          }
        }
        vec1.clear();
        vec2.clear();
      }
      else {
        if (tmpRWSet.size() > 0) {
          hasViolation = checkMemoryCoalescingCap0Size(executor, state, tmpRWSet, cTidSets, 
                                                       halfWarpNum, segSize, GPUConfig::warpsize/2, 
                                                       wordsize, noMCCond, queryNum);
        }
      }
    } else {
      GKLEE_INFO << "Threads do not have the aligned word size (4, 8, or 16), \
                    so 16 memory transactions needed!" << std::endl;
    }
    // non-coalesced accesses happen ...
    if (hasViolation) {
      updateWarpDefVec(nomcWDVec, tmpRWSet, cTidSets, 1, isWrite);
      hasCoalescing = false;
    }
 
    tmpRWSet.clear();
  }
  return hasCoalescing;
}

// Memory coalescing 0, we assume base addresses accessed 
// by threads within half warp are aligned to the beginning of 
// the segment...
bool AddressSpace::hasMemoryCoalescingCap0(Executor &executor, ExecutionState &state, 
                                           std::vector<CorrespondTid> &cTidSets, 
                                           std::vector<InstAccessSet> &instAccessSets,
                                           std::vector<klee::RefDivRegionSetVec> &divRegionSets,
                                           std::vector<SameInstVec> &sameInstVecSets,
                                           klee::ref<Expr> &noMCCond, WarpDefVec &nomcWDVec, 
                                           bool &Consider, unsigned &queryNum) {
  // Default ...
  bool hasReadCoalescing = true;
  bool hasWriteCoalescing = true;

  if (readSet.empty() && writeSet.empty()) Consider = false;
  else Consider = true; 
  
  // First handle readSet...
  if (readSet.empty())
    GKLEE_INFO << "The read set for memory coalescing cap 0 is empty" << std::endl;
  else
    hasReadCoalescing = checkMemoryCoalescingCap0(executor, state, readSet, cTidSets, 
                                                  instAccessSets, divRegionSets, sameInstVecSets,
                                                  noMCCond, nomcWDVec, queryNum, false);
  // Then writeSet...
  if (writeSet.empty())
    GKLEE_INFO << "The write set for memory coalescing cap 0 is empty" << std::endl;
  else
    hasWriteCoalescing = checkMemoryCoalescingCap0(executor, state, writeSet, cTidSets, 
                                                   instAccessSets, divRegionSets, sameInstVecSets, 
                                                   noMCCond, nomcWDVec, queryNum, true);

  return (hasReadCoalescing && hasWriteCoalescing);
}

void reduceSegmentSize(Executor &executor, ExecutionState &state, 
                       klee::ref<Expr> &lbound, klee::ref<Expr> &ubound, int &size) {
  //Solver::Validity result;
  bool result;
  klee::ref<Expr> tmpDiv1 = UDivExpr::create(lbound, ConstantExpr::create(64, lbound->getWidth()));
  klee::ref<Expr> tmpDiv2 = UDivExpr::create(ubound, ConstantExpr::create(64, ubound->getWidth()));

  klee::ref<Expr> cond = EqExpr::create(tmpDiv1, tmpDiv2);
  bool unknown = false;
  bool success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
  if (success) {
    if (result) {
      tmpDiv1 = UDivExpr::create(lbound, ConstantExpr::create(32, lbound->getWidth()));
      tmpDiv2 = UDivExpr::create(ubound, ConstantExpr::create(32, ubound->getWidth()));

      cond = EqExpr::create(tmpDiv1, tmpDiv2);
      success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
      if (success) {
        size = result ? 32 : 64;
      } else {
        size = 128;
      }
    } else {
      size = 128;
    }
  } else {
    size = 128;
  }
}

void dumpMemoryCoalescingCap1(Executor &executor, ExecutionState &state, 
                              unsigned segSize, klee::ref<Expr> &segSizeExpr, 
                              klee::ref<Expr> &numExpr, klee::ref<Expr> &lbound, klee::ref<Expr> &ubound, 
                              unsigned threadNum, bool hasViolation, 
                              unsigned accessNum, unsigned totalAccessNum) {
  GKLEE_INFO2 << "+++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

  klee::ref<Expr> start = MulExpr::create(numExpr, segSizeExpr);
  klee::ref<Expr> end = AddExpr::create(start, segSizeExpr);
  start->dump();
  end->dump();

  if (GPUConfig::verbose > 0) {
    GKLEE_INFO << "lower bound of memory access: " <<std::endl; 
    lbound->dump();
    GKLEE_INFO << "upper bound of memory access: " <<std::endl; 
    ubound->dump();
  }
  int size = 0;
  reduceSegmentSize(executor, state, lbound, ubound, size); 
  
  if (hasViolation){
    GKLEE_INFO << "This request from " << threadNum << " threads is not coalesced and split into "
               << totalAccessNum << " memory segments, and this is the " << accessNum+1 
               << "th memory segment (" << segSize << " Bytes) then reduced to " << size 
               << " Bytes" << std::endl;
   // if(Emacs) dumpEmacsInfoVect(access1.tid, access2.tid, 
   // 			      access.instr, "noncoalesc");

  } else 
    GKLEE_INFO << "This request is coalesced into one memory segment (" << segSize 
               << " Bytes) then reduced to " << size << " Bytes" << std::endl;

  GKLEE_INFO2 << "+++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

static bool checkMemoryCoalescingCap1(Executor &executor, ExecutionState &state, 
                                      MemoryAccessVec &rwSet, std::vector<CorrespondTid> &cTidSets,
                                      std::vector<InstAccessSet> &instAccessSets, 
                                      std::vector<klee::RefDivRegionSetVec> &divRegionSets, 
                                      std::vector<SameInstVec> &sameInstVecSets,
                                      klee::ref<Expr> &noMCCond, WarpDefVec &nomcWDVec, 
                                      unsigned &queryNum, bool isWrite) {
  bool hasCoalescing = true;
  unsigned halfWarpNum = 0; // The sequence number of half warps

  while(!rwSet.empty()) {
    bool hasViolation = false;
    MemoryAccessVec tmpRWSet; 
    AddressSpaceUtil::constructTmpRWSet(executor, state, rwSet, tmpRWSet, cTidSets,
                                        instAccessSets, divRegionSets, sameInstVecSets, GPUConfig::warpsize/2);
    updateWarpDefVecConsider(nomcWDVec, tmpRWSet, cTidSets, isWrite);
 
    halfWarpNum++;
    // Constructed the tmpRWSet...
    MemoryAccessVec::const_iterator tii = tmpRWSet.begin();
    unsigned wordsize = 0;
    int segSize = getSegmentSize(tii->width, wordsize, 1);

    GKLEE_INFO2 << "********** CAPACITY 1.2 or 1.3 Inst By Half-Warp ( " << halfWarpNum << " ) **********" << std::endl;
    if (segSize == -1) {
      GKLEE_INFO << "CAPACITY 1.2 or 1.3: the word size can not be recognized, so move on to another half-warp instruction!" << std::endl;
      break;
    }
    klee::ref<Expr> baseAddr = tmpRWSet[0].mo->getBaseExpr();
    klee::ref<Expr> segSizeExpr = ConstantExpr::create(segSize, baseAddr->getWidth());    
    std::vector < klee::ref<Expr> > segNumExprVec;
    std::vector < klee::ref<Expr> > lboundVec;
    std::vector < klee::ref<Expr> > uboundVec; 
    std::vector <unsigned> threadNumVec;

    unsigned segWarpNum = 0; // The number of different segments all threads in a half 
                             // warp will access 

    for (unsigned i = 0; i < tmpRWSet.size(); i++) {
      // Ensure the access is in bound of the segment...
      klee::ref<Expr> tmpSegNumExpr = UDivExpr::create(tmpRWSet[i].offset, segSizeExpr);

      if (segWarpNum == 0) {
        segWarpNum++;
        segNumExprVec.push_back(tmpSegNumExpr);
        lboundVec.push_back(tmpRWSet[i].offset);
        uboundVec.push_back(tmpRWSet[i].offset);
        threadNumVec.push_back(1);
      } else {
        unsigned diffNum = 0;
        unsigned j = 0;
        bool result = false;
        bool success = false;

        for (; j < segNumExprVec.size(); j++) {
          klee::ref<Expr> cond = EqExpr::create(segNumExprVec[j], tmpSegNumExpr);
          bool unknown = false;
          success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
          queryNum++;
          if (success) {
            if (!result) {
              if (unknown) noMCCond = AndExpr::create(noMCCond, Expr::createIsZero(cond)); 
              diffNum++; 
            }
            else break;
          }
        }
        
        if (diffNum == segNumExprVec.size()) {
          hasViolation = true;
          segNumExprVec.push_back(tmpSegNumExpr);
          lboundVec.push_back(tmpRWSet[i].offset);
          uboundVec.push_back(tmpRWSet[i].offset);
          threadNumVec.push_back(1);
          segWarpNum++;
        }
        else {
          // update the lbound and ubound
          bool unknown = false;
          klee::ref<Expr> ucond = UgtExpr::create(tmpRWSet[i].offset, uboundVec[j]); 
          success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, ucond, result, unknown);
          queryNum++;
          if (success) {
            if (result)
              uboundVec[j] = tmpRWSet[i].offset;
          }

          klee::ref<Expr> lcond = UltExpr::create(tmpRWSet[i].offset, lboundVec[j]); 
          success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, lcond, result, unknown);
          queryNum++;
          if (success) {
            if (result)
              lboundVec[j] = tmpRWSet[i].offset;
          }
          
          threadNumVec[j]++;
        }
      }
    }
    // For the left threads...
    for (unsigned k = 0; k < segNumExprVec.size(); k++)
      dumpMemoryCoalescingCap1(executor, state, segSize, segSizeExpr, segNumExprVec[k], 
                               lboundVec[k], uboundVec[k], threadNumVec[k],
                               hasViolation, k, segNumExprVec.size());
    // non-coalesced accesses happen ...
    if (hasViolation) {
      updateWarpDefVec(nomcWDVec, tmpRWSet, cTidSets, 1, isWrite);
      hasCoalescing = false;
    }

    // clear these vectors...
    tmpRWSet.clear();
    segNumExprVec.clear();
    lboundVec.clear();
    uboundVec.clear();
    threadNumVec.clear();
  }
  return hasCoalescing;
}

// Memory coalescing 1, we also assume the first element accessed 
// by first thread in half warp is aligned to the beginning of the 
// segment...
bool AddressSpace::hasMemoryCoalescingCap1(Executor &executor, ExecutionState &state,
                                           std::vector<CorrespondTid> &cTidSets, 
                                           std::vector<InstAccessSet> &instAccessSets,
                                           std::vector<klee::RefDivRegionSetVec> &divRegionSets, 
                                           std::vector<SameInstVec> &sameInstVecSets,
                                           klee::ref<Expr> &noMCCond, WarpDefVec &nomcWDVec, 
                                           bool &Consider, unsigned &queryNum) {
  bool hasReadCoalescing = true;
  bool hasWriteCoalescing = true;

  if (readSet.empty() && writeSet.empty()) Consider = false;
  else Consider = true; 

  // Frist handle readSet...
  if (readSet.empty())
    GKLEE_INFO << "The read set for memory coalescing cap 1 is empty" << std::endl;
  else
    hasReadCoalescing = checkMemoryCoalescingCap1(executor, state, readSet, cTidSets, 
                                                  instAccessSets, divRegionSets, sameInstVecSets, 
                                                  noMCCond, nomcWDVec, queryNum, false);
  // Then writeSet...
  if (writeSet.empty())
    GKLEE_INFO << "The write set for memory coalescing cap 1 is empty" << std::endl;
  else
    hasWriteCoalescing = checkMemoryCoalescingCap1(executor, state, writeSet, cTidSets, 
                                                   instAccessSets, divRegionSets, sameInstVecSets, 
                                                   noMCCond, nomcWDVec, queryNum, true);
  
  return (hasReadCoalescing && hasWriteCoalescing);
}

void dumpMemoryCoalescingCap2Begin(unsigned warpNum, unsigned wordsize, 
                                   unsigned k, unsigned reqNum) {
  GKLEE_INFO << "The word size accessed by threads: " << wordsize 
             << ", the " << k+1 << "th request over total " << reqNum 
             << " requests" << std::endl;
}

void dumpMemoryCoalescingCap2Body(klee::ref<Expr> &lbound, klee::ref<Expr> &ubound, 
                                  unsigned threadNum, bool hasViolation, 
                                  unsigned accessNum, unsigned totalAccessNum) {
  GKLEE_INFO2 << "+++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  if (hasViolation) {
    GKLEE_INFO << "This request is not coalesced and split into " 
               << totalAccessNum << " memory segments, and this is the " << accessNum+1 
               << "th memory segment (128 Bytes) accessed by " 
               << threadNum << " threads" << std::endl;
  }
  else
    GKLEE_INFO << "This request is coalesced into one memory segment (128 Bytes) accessed by "
               << threadNum << " threads" << std::endl;

  if (GPUConfig::verbose > 0) {
    GKLEE_INFO << "lower bound of this segment: " <<std::endl; 
    lbound->dump();
    GKLEE_INFO << "upper bound of this segment: " <<std::endl; 
    ubound->dump();
  }
  
  GKLEE_INFO2 << "+++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

static bool checkMemoryCoalescingCap2(Executor &executor, ExecutionState &state, 
                                      MemoryAccessVec &rwSet, std::vector<CorrespondTid> &cTidSets, 
                                      std::vector<InstAccessSet> &instAccessSets, 
                                      std::vector<klee::RefDivRegionSetVec> &divRegionSets, 
                                      std::vector<SameInstVec> &sameInstVecSets,
                                      klee::ref<Expr> &noMCCond, WarpDefVec &nomcWDVec, 
                                      unsigned &queryNum, bool isWrite) {
  bool hasCoalescing = true;
  unsigned warpNum = 0; // The sequence number of half warps

  while(!rwSet.empty()) {
    bool hasViolation = false;
    MemoryAccessVec tmpRWSet; 
    AddressSpaceUtil::constructTmpRWSet(executor, state, rwSet, tmpRWSet, cTidSets, 
                                        instAccessSets, divRegionSets, sameInstVecSets, 
                                        GPUConfig::warpsize);
    updateWarpDefVecConsider(nomcWDVec, tmpRWSet, cTidSets, isWrite);

    unsigned delimit = cTidSets[tmpRWSet.begin()->tid].rTid / GPUConfig::warpsize;

    warpNum++;
    // Constructed the tmpRWSet...
    unsigned wordsize = 0;
    MemoryAccessVec::const_iterator tii = tmpRWSet.begin();
    int reqNum = getSegmentSize(tii->width, wordsize, 2);

    GKLEE_INFO2 << "********** CAPACITY 2.x Inst By Whole Warp ( " << warpNum << " ) **********" << std::endl;
    if (reqNum == -1) {
      GKLEE_INFO << "CAPACITY 2.x: the word size can not be recognized, so quit the checking procedure!" << std::endl;
      break;
    }

    std::vector < MemoryAccessVec > reqSets;
    MemoryAccessVec vec;
    for (int i = 0; i < reqNum; i++) {
      // Split the 32 threads into sub-warps...
      reqSets.push_back(vec);
    }
  
    for (unsigned i = 0; i < tmpRWSet.size(); i++) {
      unsigned tmpTid;
      if (delimit != 0)
        tmpTid = cTidSets[tmpRWSet[i].tid].rTid % GPUConfig::warpsize;
      else
        tmpTid = cTidSets[tmpRWSet[i].tid].rTid;

      unsigned divisor = GPUConfig::warpsize / reqNum; 
      unsigned a = tmpTid / divisor; 
      reqSets[a].push_back(tmpRWSet[i]);
    }

    for (int k = 0; k < reqNum; k++) {
      // cache line is 128...
      unsigned segSize = 128;
      klee::ref<Expr> baseAddr = tmpRWSet[0].mo->getBaseExpr(); 
      klee::ref<Expr> segSizeExpr = ConstantExpr::create(segSize, baseAddr->getWidth());
      klee::ref<Expr> lbound;
      klee::ref<Expr> ubound;
      std::vector < klee::ref<Expr> > segNumExprVec;
      std::vector < klee::ref<Expr> > lboundVec; 
      std::vector < klee::ref<Expr> > uboundVec; 
      std::vector <unsigned> threadNumVec;

      unsigned segWarpNum = 0; // The number of different segments all threads in a half 
                               // warp will access 

      MemoryAccessVec &tmpReqSet = reqSets[k];
      for (unsigned i = 0; i < tmpReqSet.size(); i++) {
        // ensure the access is in bound of segment...
        klee::ref<Expr> tmpSegNumExpr = UDivExpr::create(tmpReqSet[i].offset, segSizeExpr);

        if (segWarpNum == 0) {
          segWarpNum++;
          segNumExprVec.push_back(tmpSegNumExpr);
          lbound = MulExpr::create(segSizeExpr, tmpSegNumExpr); 
          ubound = AddExpr::create(lbound, segSizeExpr); 
          lboundVec.push_back(lbound);
          uboundVec.push_back(ubound);
          threadNumVec.push_back(1);
        } else {
          unsigned diffNum = 0;
          bool result = false;
          bool success = false;
          unsigned j = 0;

          for (; j < segNumExprVec.size(); j++) {
            klee::ref<Expr> cond = EqExpr::create(segNumExprVec[j], tmpSegNumExpr);
            bool unknown = false;
            success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
            queryNum++;
            if (success) {
              if (!result) {
                if (unknown) noMCCond = AndExpr::create(noMCCond, Expr::createIsZero(cond));
                diffNum++;
              }
              else break;
            }
          }

          if (diffNum == segNumExprVec.size()) {
            hasViolation = true;
            segNumExprVec.push_back(tmpSegNumExpr);
            lbound = MulExpr::create(segSizeExpr, tmpSegNumExpr); 
            ubound = AddExpr::create(lbound, segSizeExpr); 
            lboundVec.push_back(lbound);
            uboundVec.push_back(ubound);
            threadNumVec.push_back(1);
            segWarpNum++;
          } else {
            threadNumVec[j]++;
          }
        }
      }

      dumpMemoryCoalescingCap2Begin(warpNum, wordsize, k, reqNum);
      for (unsigned m = 0; m < segNumExprVec.size(); m++) {
        dumpMemoryCoalescingCap2Body(lboundVec[m], uboundVec[m], threadNumVec[m], 
                                     hasViolation, m, segNumExprVec.size());
      }
      // clear these vectors...
      tmpReqSet.clear();
      reqSets[k].clear();
      segNumExprVec.clear();
      lboundVec.clear();
      uboundVec.clear();
      threadNumVec.clear();
    }
    // non-coalesced accesses happen ...
    if (hasViolation) {
      updateWarpDefVec(nomcWDVec, tmpRWSet, cTidSets, 1, isWrite);
      hasCoalescing = false;
    }

    tmpRWSet.clear();
  }
  return hasCoalescing;
}

bool AddressSpace::hasMemoryCoalescingCap2(Executor &executor, ExecutionState &state,
                                           std::vector<CorrespondTid> &cTidSets, 
                                           std::vector<InstAccessSet> &instAccessSets, 
                                           std::vector<klee::RefDivRegionSetVec> &divRegionSets, 
                                           std::vector<SameInstVec> &sameInstVecSets, 
                                           klee::ref<Expr> &noMCCond, WarpDefVec &nomcWDVec, 
                                           bool &Consider, unsigned &queryNum) {
  bool hasReadCoalescing = true;
  bool hasWriteCoalescing = true;

  if (readSet.empty() && writeSet.empty()) Consider = false;
  else Consider = true; 

  // Frist handle readSet...
  if (readSet.empty())
    GKLEE_INFO << "The read set for memory coalescing cap 2 is empty" << std::endl;
  else 
    hasReadCoalescing = checkMemoryCoalescingCap2(executor, state, readSet, cTidSets, 
                                                  instAccessSets, divRegionSets, sameInstVecSets, 
                                                  noMCCond, nomcWDVec, queryNum, false);
  // Then writeSet...
  if (writeSet.empty())
    GKLEE_INFO << "The write set for memory coalescing cap 2 is empty" << std::endl;
  else
    hasWriteCoalescing = checkMemoryCoalescingCap2(executor, state, writeSet, cTidSets, 
                                                   instAccessSets, divRegionSets, sameInstVecSets, 
                                                   noMCCond, nomcWDVec, queryNum, true);
  
  return (hasReadCoalescing && hasWriteCoalescing);
}

bool HierAddressSpace::hasMemoryCoalescing(Executor &executor, ExecutionState &state, 
                                           std::vector<CorrespondTid> &cTidSets,
                                           unsigned capability) {
  std::string str;
  if (capability == 0)
    str = "1.0 or 1.1"; 
  else if (capability == 1)
    str = "1.2 or 1.3";
  else
    str = "2.x";
  GKLEE_INFO << "\n********** Start checking memory coalescing at DeviceMemory at capability: " 
            << str << " **********\n";

  bool hasCoalescing = true;
  WarpDefVec nomcWDVec;
  unsigned tmpWarpId = 0;

  nomcWDVec.push_back(WarpDefectInfo(cTidSets.begin()->warpNum, cTidSets.begin()->rBid)); 
  for (std::vector<CorrespondTid>::iterator ii = cTidSets.begin(); ii != cTidSets.end(); ii++) {
    if (ii->warpNum != tmpWarpId) {
      tmpWarpId = ii->warpNum;
      nomcWDVec.push_back(WarpDefectInfo(ii->warpNum, ii->rBid));
    } 
  } 

  bool Consider = false;
  klee::ref<Expr> nonMCCond = ConstantExpr::create(1, Expr::Bool);
  if (capability == 0) {
    hasCoalescing = deviceMemory.hasMemoryCoalescingCap0(executor, state, cTidSets, instAccessSets, 
                                                         divRegionSets, sameInstVecSets, 
                                                         nonMCCond, nomcWDVec, Consider, 
                                                         mcQueryNum);
  } else if (capability == 1) {
    hasCoalescing = deviceMemory.hasMemoryCoalescingCap1(executor, state, cTidSets, instAccessSets, 
                                                         divRegionSets, sameInstVecSets, 
                                                         nonMCCond, nomcWDVec, Consider, 
                                                         mcQueryNum);
  } else {
    hasCoalescing = deviceMemory.hasMemoryCoalescingCap2(executor, state, cTidSets, instAccessSets, 
                                                         divRegionSets, sameInstVecSets, 
                                                         nonMCCond, nomcWDVec, Consider, 
                                                         mcQueryNum);
  }
  nonMCCondComb = AndExpr::create(nonMCCondComb, nonMCCond);

  if (Consider) {
    nomcWDSet.push_back(nomcWDVec);
    deviceMemory.numMCBI++;
    if (hasCoalescing) {
      deviceMemory.numMC++;
    } else {
      hasNoMC = true;
    }
  } else nomcWDVec.clear();

  return hasCoalescing;
}

static void concludeWarpDivergStatistics(SameInstVec &sameSets, std::vector<InstAccessSet> &instSets, 
                                         std::vector<klee::RefDivRegionSetVec> &divRegionSets, unsigned warpNum) {
  GKLEE_INFO << "In warp " << warpNum << ", threads are diverged into following sub-sets: " 
             << std::endl;
    
  for (unsigned k = 0; k < sameSets.size(); k++) {
    GKLEE_INFO << "Set " << k << ":" << std::endl;
    GKLEE_INFO << "Threads: ";
    std::vector<unsigned>::iterator ii;
    for (ii = sameSets[k].begin(); ii != sameSets[k].end(); ii++) {
      std::cout << *ii; 
      if (ii != sameSets[k].end()-1)
        std::cout << ", "; 
    }
    std::cout << std::endl;
  }
  // Output some verbose difference ... 
  if (GPUConfig::verbose > 0) {
    if (sameSets.size() == 1) {
      GKLEE_INFO << "Only one set, no divergence exists!" << std::endl; 
    }
    else {
      GKLEE_INFO << "We only pick 0 and 1 subsets to illustrate the difference" 
                 << std::endl;
      unsigned tid1 = *(sameSets[0].begin());
      unsigned tid2 = *(sameSets[1].begin());
      GKLEE_INFO << "Thread " << tid1 << " in subset 0 and thread " << tid2 
                 << " in subset 1 result in warp divergence in the same warp!"
                 << std::endl; 
    }
  }
}

bool HierAddressSpace::hasWarpDivergence(std::vector<CorrespondTid> &cTidSets) {
  // First check whether these threads in the same warp..
  unsigned warpNum = 0;
  bool hasDiverge = false;

  GKLEE_INFO << "\n********** Start checking warp divergence **********\n";

  WarpDefVec wdWDVec;
  unsigned tmpWarpId = 0;

  wdWDVec.push_back(WarpDefectInfo(cTidSets.begin()->warpNum, cTidSets.begin()->rBid)); 
  for (std::vector<CorrespondTid>::iterator ii = cTidSets.begin(); ii != cTidSets.end(); ii++) {
    if (ii->warpNum != tmpWarpId) {
      tmpWarpId = ii->warpNum;
      wdWDVec.push_back(WarpDefectInfo(ii->warpNum, ii->rBid));
    }
  }

  for (unsigned i = 0; i<sameInstVecSets.size(); i++, warpNum++) {
    wdWDVec[warpNum].consider = true;
    concludeWarpDivergStatistics(sameInstVecSets[i], instAccessSets, 
                                 divRegionSets, warpNum);
    if (sameInstVecSets[i].size() != 1) { 
      // warp divergence occurs .. 
      wdWDVec[warpNum].occur = 1;
      hasDiverge = true;
    }
  }

  // For each BI
  numWDBI++;
  if (hasDiverge)
    numWD++;
  wdWDSet.push_back(wdWDVec);

  GKLEE_INFO2 << "*****************************************************\n";
  return hasDiverge;
}

// races in all the address spaces
bool HierAddressSpace::hasBankConflict(Executor &executor, ExecutionState &state, 
                                       std::vector<CorrespondTid> &cTidSets, 
                                       unsigned capability) {
  WarpDefVec bcWDVec;
  unsigned tmpWarpId = 0;

  bcWDVec.push_back(WarpDefectInfo(cTidSets.begin()->warpNum, cTidSets.begin()->rBid)); 
  for (std::vector<CorrespondTid>::iterator ii = cTidSets.begin(); ii != cTidSets.end(); ii++) {
    if (ii->warpNum != tmpWarpId) {
      tmpWarpId = ii->warpNum;
      bcWDVec.push_back(WarpDefectInfo(ii->warpNum, ii->rBid));
    }
  }

  unsigned i = 0; 
  bool bc = false;
  bool Consider = false;
  bool totalConsider = false;

  for (std::vector<AddressSpace>::iterator ii = sharedMemories.begin(); 
       ii != sharedMemories.end(); ii++)
  {
    GKLEE_INFO << "\n********** Start checking bank conflicts at SharedMemory " 
               << i++ << " **********\n";
    if (GPUConfig::verbose > 0)
      ii->dump(true);

    klee::ref<Expr> bcCond = ConstantExpr::create(1, Expr::Bool); 
    bc = ii->hasBankConflict(executor, state, capability, cTidSets, 
                             instAccessSets, divRegionSets,
                             sameInstVecSets, bcCond, bcWDVec, 
                             Consider, bcQueryNum);
    bcCondComb = AndExpr::create(bcCondComb, bcCond);
    
    if (Consider) {
      totalConsider = true;
      ii->numBCBI++;
      if (bc) {
        hasBC = true;
        ii->numBC++;
      }
    }
  }

  if (totalConsider) bcWDSet.push_back(bcWDVec);
  else bcWDVec.clear();

  return bc;
}

void HierAddressSpace::getMCRate(unsigned &mcWarpNum, unsigned &mcWarpSum, 
                                 unsigned &mcBINum, unsigned &mcBISum) {
  if (deviceMemory.numMCBI) {
    unsigned bi = 0;
    unsigned mcWarpTotal = 0; 
    unsigned warpTotal = 0;
  
    unsigned mcInstReadOccur = 0;
    unsigned mcInstReadTotal = 0;

    unsigned mcInstWriteOccur = 0;
    unsigned mcInstWriteTotal = 0;

    for (std::vector<WarpDefVec>::iterator ii = nomcWDSet.begin(); 
         ii != nomcWDSet.end(); ii++, bi++) {
      WarpDefVec &nomcVec = *(ii);
      // warp level 
      unsigned occurNum = 0;
      unsigned totalNum = 0;
      // instruction level
      unsigned readOccurNum = 0;
      unsigned readTotalNum = 0;

      unsigned writeOccurNum = 0;
      unsigned writeTotalNum = 0;

      for (WarpDefVec::iterator jj = nomcVec.begin(); jj != nomcVec.end(); jj++) {
        //GKLEE_INFO << "Warp (" << jj->warpID << ") : " << jj->consider << std::endl; 
        if (jj->consider) {
          //GKLEE_INFO << "Warp (" << jj->warpID << ") : " << jj->consider << std::endl; 
          totalNum++;
          readOccurNum += jj->instReadOccur;
          writeOccurNum += jj->instWriteOccur;
          readTotalNum += jj->instReadTotal; 
          writeTotalNum += jj->instWriteTotal; 

          if (jj->occur) occurNum++;
        }
      }

      mcWarpTotal += (totalNum-occurNum); 
      warpTotal += totalNum;

      mcInstReadOccur += (readTotalNum - readOccurNum);
      mcInstWriteOccur += (writeTotalNum - writeOccurNum);
      
      mcInstReadTotal += readTotalNum;
      mcInstWriteTotal += writeTotalNum;
    }
    mcWarpNum += mcWarpTotal;
    mcWarpSum += warpTotal;

    GKLEE_INFO << "Across " << bi << " BIs, the total num of read instructions with MC: " 
               << mcInstReadOccur << ", the total number of read instructions: " 
               << mcInstReadTotal << std::endl;

    GKLEE_INFO << "Across " << bi << " BIs, the total num of write instructions with MC: " 
               << mcInstWriteOccur << ", the total number of write instructions: " 
               << mcInstWriteTotal << std::endl;

    GKLEE_INFO << "Across " << bi << " BIs, the total num of warps with MC: " << mcWarpTotal
               << ", the total num of warps: " << warpTotal << std::endl;

    GKLEE_INFO << "num of BIs with MC: " << deviceMemory.numMC << ", num of BIs: " 
               << deviceMemory.numMCBI << std::endl;

    mcBINum += deviceMemory.numMC;
    mcBISum += deviceMemory.numMCBI;
  }
  //GKLEE_INFO << "The mc query number: " << mcQueryNum << std::endl;
}

void HierAddressSpace::getBCRate(unsigned &bcWarpNum, unsigned &bcWarpSum,
                                 unsigned &bcBINum, unsigned &bcBISum) {
  unsigned i = 0;

  unsigned bi = 0;
  unsigned bcWarpTotal = 0;
  unsigned warpTotal = 0;

  unsigned bcInstOccur = 0;
  unsigned bcInstTotal = 0;

  for (std::vector<WarpDefVec>::iterator ii = bcWDSet.begin(); 
       ii != bcWDSet.end(); ii++, bi++) {
    WarpDefVec &bcVec = *(ii);
    // warp level
    unsigned occurNum = 0;
    unsigned totalNum = 0;

    for (WarpDefVec::iterator jj = bcVec.begin(); jj != bcVec.end(); jj++) {
      if (jj->consider) {
        totalNum++;
        bcInstOccur += jj->instReadOccur;
        bcInstOccur += jj->instWriteOccur;
        bcInstTotal += jj->instReadTotal;
        bcInstTotal += jj->instWriteTotal; 
        if (jj->occur) occurNum++;
      }
    }
    bcWarpTotal += occurNum;
    warpTotal += totalNum;
  }
  bcWarpNum += bcWarpTotal;
  bcWarpSum += warpTotal;

  GKLEE_INFO << "Across " << bi << " BIs, the total num of instructions with BC: " 
             << bcInstOccur << ", the total num of instructions: " << bcInstTotal 
             << std::endl;

  GKLEE_INFO << "Across " << bi << " BIs, the total num of warps with BC: " 
             << bcWarpTotal << ", the total num of warps: " << warpTotal 
             << std::endl;

  // I think the divisor "warpTotal" can not be 0...
  unsigned tmpBCBINum = 0;
  unsigned tmpBCBISum = 0;
  for (std::vector<AddressSpace>::iterator ii = sharedMemories.begin(); 
       ii != sharedMemories.end(); ii++, i++) {
    if (ii->numBCBI) {
      GKLEE_INFO << "In shared memory " << i << ", num of BIs with BC: " 
                 << ii->numBC << ", num of BIs: " << ii->numBCBI << std::endl; 
      tmpBCBINum += ii->numBC;
      tmpBCBISum += ii->numBCBI;
    }
  }
  bcBINum += tmpBCBINum / i;
  bcBISum += tmpBCBISum / i;
  //GKLEE_INFO << "The bc query num: " << bcQueryNum << std::endl;
}

void HierAddressSpace::getWDRate(unsigned &wdWarpNum, unsigned &wdWarpSum,
                                 unsigned &wdBINum, unsigned &wdBISum) {
  if (numWDBI) {
    unsigned bi = 0;
    unsigned wdWarpTotal = 0;
    unsigned warpTotal = 0;

    for (std::vector<WarpDefVec>::iterator ii = wdWDSet.begin(); 
         ii != wdWDSet.end(); ii++, bi++) {
      WarpDefVec &wdVec = *(ii);
      unsigned occurNum = 0;
      unsigned totalNum = 0;

      for (WarpDefVec::iterator jj = wdVec.begin(); jj != wdVec.end(); jj++) {
        if (jj->consider) {
          totalNum++;
          if (jj->occur) occurNum++;
        }
      }
      wdWarpTotal += occurNum;
      warpTotal += totalNum;
    }
    wdWarpNum += wdWarpTotal;
    wdWarpSum += warpTotal;

    GKLEE_INFO << "Across " << bi << " BIs, the total num of warps with WD: " << wdWarpTotal
               << ", the total num of warps: " << warpTotal << std::endl;

    GKLEE_INFO << "Num of BIs with WD: " << numWD << ", num of BIs: " 
               << numWDBI << std::endl;
    wdBINum += numWD;
    wdBISum += numWDBI;
  }
}
