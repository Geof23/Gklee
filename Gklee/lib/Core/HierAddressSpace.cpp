//===-- HierAddressSpace.cpp ----------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Executor.h"
#include "AddressSpace.h"
#include "CoreStats.h"
#include "Memory.h"
#include "TimingSolver.h"

#include "klee/Expr.h"
#include "klee/TimerStatIncrementer.h"
#include "klee/Constraints.h"
#include "klee/util/Assignment.h"
#include "klee/Internal/Module/KInstruction.h"
#include <iostream>
#include <fstream>

#include "llvm/Support/raw_ostream.h"
#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
#include "llvm/Analysis/DebugInfo.h" 
#else
#include "llvm/DebugInfo.h"
#endif

#include "CUDA.h"
#include <assert.h>

using namespace llvm;
using namespace klee;

namespace runtime {
  cl::opt<bool>
  DumpDetailSolution("dump-detail-solution", 
                     cl::desc("Dump the intermediate solution for the memory access"),
                     cl::init(false)); 
  extern cl::opt<bool> UseSymbolicConfig;
  extern cl::opt<bool> Emacs;
  extern cl::opt<bool> SimdSchedule;
}

using namespace runtime;

//********************************************************************************************
// Pretty printing of memory accesses
//********************************************************************************************

static bool accessSameMemoryRegion(Executor &, ExecutionState &,
                                   ref<Expr>, ref<Expr>);

static bool isInSameHalfOrEntireWarp(std::vector<CorrespondTid> &, unsigned, unsigned, 
                                     unsigned, unsigned, unsigned);

static bool inRegionBound(unsigned, unsigned, unsigned);
static bool hasSameExecTrace(std::vector<CorrespondTid> &, 
                             std::vector<SameInstVec> &, 
                             unsigned, unsigned &, 
                             unsigned, unsigned &);
static int findDivRegionNum(RefDivRegionSetVec &, unsigned);

static bool isBothAtomic(const MemoryAccess &access1, 
                         const MemoryAccess &access2) {
  return access1.isAtomic && access2.isAtomic;
}

bool AddressSpaceUtil::evaluateQueryMustBeTrue(Executor &executor, 
                                               ExecutionState &state, 
                                               ref<Expr> &expr, bool &result, 
                                               bool &unknown) {
  // In this case, result false case includes 'false', 'unknown'
  bool success = executor.solver->mustBeTrue(state, expr, result);

  if (result) 
    unknown = false; 
  else {
    bool known = false;
    bool success1 = executor.solver->mustBeFalse(state, expr, known);
    if (success1) {
      unknown = known? false : true;
    } else {
      unknown = false;
    }
  }

  return success;
}

bool AddressSpaceUtil::evaluateQueryMustBeFalse(Executor &executor, ExecutionState &state, 
                                                ref<Expr> &expr, bool &result, bool &unknown) {
  // In this case, result false case includes 'true', 'unknown'
  bool success = executor.solver->mustBeFalse(state, expr, result); 

  if (result) 
    unknown = false; 
  else {
    bool known = false;
    bool success1 = executor.solver->mustBeTrue(state, expr, known);
    if (success1) {
      unknown = known? false : true;
    } else {
      unknown = false;
    }
  }

  return success;
}

bool AddressSpaceUtil::isTwoInstIdentical(llvm::Instruction *inst1, 
                                          llvm::Instruction *inst2) {
  std::string func1Name = inst1->getParent()->getParent()->getName().str();
  std::string func2Name = inst2->getParent()->getParent()->getName().str();
  llvm::BasicBlock *bb1 = inst1->getParent();
  llvm::BasicBlock *bb2 = inst2->getParent();

  return func1Name.compare(func2Name) == 0 
           && bb1 == bb2
             && inst1->isIdenticalTo(inst2); 
}

static void extractInstFromSourceCode(MDNode *N) {
  DILocation Loc(N);               // DILocation is in DebugInfo.h
  unsigned Line = Loc.getLineNumber();
  StringRef File = Loc.getFilename();
  StringRef Dir = Loc.getDirectory();
  std::cout << "Instruction Line: " << Line << ", In File: " 
            << File.str() << ", With Dir Path: " << Dir.str() 
            << std::endl;

  std::string filePath = Dir.str() + "/" + File.str();
  ifstream src(filePath.data(), ifstream::in);
  if (src.is_open()) {
    unsigned num = 0;
    std::string cLine;
    do {
      getline(src, cLine);
      num++;
    } while (num != Line);
    
    std::cout << "[File: " << filePath << ", Line: " << Line
              << ", Inst: " << cLine << "]" << std::endl;
  } else {
    std::cout << "Can not open file!" << std::endl;
  }
}

void MemoryAccess::dump() const {
  std::cout << "[GKLEE] Inst: " << std::endl;
  if (MDNode *N = instr->getMetadata("dbg")) {  // Here I is an LLVM instruction
    extractInstFromSourceCode(N);
    instr->dump();
  } else {
    instr->dump();
  }

  if (is_write) 
    std::cout << "<W: ";
  else 
    std::cout << "<R: ";

  if (mo->name.compare("unnamed") != 0)
    std::cout << mo->name << ", ";
  else
    std::cout << mo->address << ", ";

  offset->print(std::cout);
  std::cout << ":";
  val->print(std::cout);
  std::cout << ", b" << bid << ", t" << tid << "> " << std::endl;
}

void MemoryAccess::dump(Executor &executor, ExecutionState &state, 
                        ref<Expr> cond) const {
  std::cout << "[GKLEE] Inst: " << std::endl;
  if (MDNode *N = instr->getMetadata("dbg")) {  // Here I is an LLVM instruction
    extractInstFromSourceCode(N);
    instr->dump();
  } else { 
    instr->dump();
  }

  std::vector<std::pair<std::string, std::vector<unsigned char> > > res;
  executor.getConditionSolution(state, cond, res);
  // construct the binding 
  std::vector<const Array*> objects;
  std::vector<std::vector<unsigned char> > values;
  for (unsigned i = 0; i != state.symbolics.size(); ++i) {
    objects.push_back(state.symbolics[i].second);
    values.push_back(res[i].second);
  }

  Assignment *binding = new Assignment(objects, values);  

  if (is_write)
    std::cout << "<W: ";
  else 
    std::cout << "<R: ";

  if (mo->name.compare("unnamed") != 0)
    std::cout << mo->name << ", ";
  else
    std::cout << mo->address << ", ";

  if (!isa<ConstantExpr>(offset)) {
    ref<Expr> cOffset = binding->evaluate(offset);
    cOffset->print(std::cout);
  } else {
    offset->print(std::cout);
  }

  std::cout << ":";
  if (!isa<ConstantExpr>(val)) {
    ref<Expr> cVal = binding->evaluate(val);
    cVal->print(std::cout);
  } else {
    val->print(std::cout);
  }

  if (UseSymbolicConfig) 
    std::cout << "> " << std::endl;
  else
    std::cout << ", b" << bid << ", t" << tid << "> " << std::endl;  

  if (DumpDetailSolution) {
    std::cout << "Unconcretized offset: " << std::endl;
    offset->print(std::cout);
    if (is_write) {
      std::cout << "Unconcretized value: " << std::endl;
      val->print(std::cout);
    } 
    std::cout << "With concretized inputs: " << std::endl;
    for (unsigned i = 0; i < res.size(); i++) {
      std::cout << res[i].first << " : ";  
      for (unsigned j = 0; j < res[i].second.size(); j++) {
        std::cout << (unsigned)res[i].second[j] << " ";
      }
      std::cout << std::endl;
    }
  }

  delete binding;
}

void InstAccess::dump() const {
  std::cout << "<b" << bid << ", t" << tid << ">:  ";
  if (MDNode *N = inst->getMetadata("dbg")) {
    extractInstFromSourceCode(N);
  } else {
    inst->dump();
  }
  std::cout << std::endl;
}

AddressSpace &AddressSpace::operator=(const AddressSpace& b) {
  if (this != &b) { 
    cowKey = ++b.cowKey;
    objects = b.objects;
    ctype = b.ctype;
    numBCBI = b.numBCBI;
    numBC = b.numBC;
    numMCBI = b.numMCBI;
    numMC = b.numMC; 
    readSet = b.readSet;
    writeSet = b.writeSet;
    MemAccessSets = b.MemAccessSets;
    MemAccessSetsPureCS = b.MemAccessSetsPureCS;
    symGlobalReadSets = b.symGlobalReadSets;
    symGlobalWriteSets = b.symGlobalWriteSets;
  }
  return *this;
}

//****************************************************************************************************
// Conflict checking
//****************************************************************************************************

// return true if a conflict is found
bool checkConflictExprs(Executor &executor, ExecutionState &state, 
                        ref<Expr> &raceCond, unsigned &queryNum, 
                        ref<Expr> &addr1, Expr::Width width1, 
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

  // the fast path
  if (klee::ConstantExpr *CE = dyn_cast<klee::ConstantExpr>(expr)) {
    queryNum++;
    return CE->isTrue() ? true : false;
  }

  // consult the solver
  bool result;
  bool unknown = false;
  bool success = AddressSpaceUtil::evaluateQueryMustBeFalse(executor, state, expr, result, unknown);
  queryNum++;
  if (success) {
    if (!result) {
      if (unknown) raceCond = AndExpr::create(raceCond, expr);
    }
    return !result;
  }
  return false;
}

//****************************************************************************************************
// Check Races / Volatile Missing 
//****************************************************************************************************

static bool belongToSameWarp(std::vector<CorrespondTid> &cTidSets, 
                             const MemoryAccessVec &vec1, 
                             const MemoryAccessVec &vec2, unsigned warpsize) {
  return isInSameHalfOrEntireWarp(cTidSets, vec1.begin()->bid, vec1.begin()->tid, 
                                  vec2.begin()->bid, vec2.begin()->tid, warpsize);
}

static bool belongToSameBlockNotSameWarp(const MemoryAccessVec &vec1, 
                                         const MemoryAccessVec &vec2, 
                                         std::vector<CorrespondTid> &cTidSets) {
  if (vec1.begin()->bid == vec2.begin()->bid) {
    if (cTidSets[vec1.begin()->tid].warpNum != cTidSets[vec2.begin()->tid].warpNum)
      return true;
  }
  return false;
}

static bool checkVolatileMissing(Executor &executor, ExecutionState &state, 
                                 MemoryAccessVec &readVec, MemoryAccessVec &writeVec, 
                                 unsigned mark, ref<Expr> &vmCond) {
    bool vmissing = false;
    unsigned vmQueryNum = 0;
  
    // check the potential Read-Write sharing first 
    for (MemoryAccessVec::const_iterator ii = writeVec.begin(); ii != writeVec.end(); ii++) {
      ref<Expr> addr1 = ii->offset;
      unsigned tid1 = ii->tid;
      Expr::Width width1 = ii->width;
      for (MemoryAccessVec::const_iterator jj = readVec.begin(); jj != readVec.end(); jj++) {
        unsigned tid2 = jj->tid;
        if (tid1 != tid2) {
          ref<Expr> addr2 = jj->offset;
	  Expr::Width width2 = jj->width;
	  if (accessSameMemoryRegion(executor, state, ii->mo->getBaseExpr(), jj->mo->getBaseExpr()) &&
              checkConflictExprs(executor, state, vmCond, vmQueryNum, 
                                 addr1, width1, addr2, width2)) {
            if (mark == 0)
	      GKLEE_INFO2 << "Threads " << tid2 << " and " << tid1
	      	          << " have a Read-Write sharing on " << std::endl;
            else if (mark == 1)
	      GKLEE_INFO2 << "Threads " << tid2 << " and " << tid1
	      	          << " have a Write-Write sharing on " << std::endl;
            else 
	      GKLEE_INFO2 << "Threads " << tid2 << " and " << tid1
	      	          << " have a Read-Read sharing on " << std::endl;
	    ii->dump(executor, state, vmCond);
            jj->dump(executor, state, vmCond);

            GKLEE_INFO2 << "These two threads access common memory location, "
                        << "it is better to set shared variables as volatile!" << std::endl;
	    //ADD EMACS INFO CALL HERE!!!
 	    if (Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, ii->instr, jj->instr, "mv");
            vmissing = true;
	  }
        }
      }
    }

    return vmissing;
}

bool AddressSpace::hasVolatileMissing(Executor &executor, ExecutionState &state, 
                                      std::vector<CorrespondTid> &cTidSets, 
                                      std::vector<InstAccessSet> &instAccessSets, 
                                      std::vector<RefDivRegionSetVec> &divRegionSets, 
                                      std::vector<SameInstVec> &sameInstVecSets, 
                                      ref<Expr> &vmCond) {
  MemoryAccessVec tmpReadSet = readSet;
  MemoryAccessVec tmpWriteSet = writeSet;    

  std::vector<MemoryAccessVec> vmReadSets;
  std::vector<MemoryAccessVec> vmWriteSets;

  bool volatilemiss = false;

  // Handle read set first
  while (!tmpReadSet.empty()) {
    MemoryAccessVec vmReadVec;
    AddressSpaceUtil::constructTmpRWSet(executor, state, tmpReadSet, 
                                        vmReadVec, cTidSets, 
                                        instAccessSets, divRegionSets, 
                                        sameInstVecSets, GPUConfig::warpsize);
    vmReadSets.push_back(vmReadVec);
  } 

  // Handle write set
  while(!tmpWriteSet.empty()) {
    MemoryAccessVec vmWriteVec;
    AddressSpaceUtil::constructTmpRWSet(executor, state, tmpWriteSet, 
                                        vmWriteVec, cTidSets, 
                                        instAccessSets, divRegionSets, 
                                        sameInstVecSets, GPUConfig::warpsize);  
    vmWriteSets.push_back(vmWriteVec);
  }

  // Read-Write sharing 
  for (std::vector<MemoryAccessVec>::iterator ii = vmReadSets.begin();
       ii != vmReadSets.end(); ii++) {
    for (std::vector<MemoryAccessVec>::iterator jj = vmWriteSets.begin(); 
         jj != vmWriteSets.end(); jj++) {
      if (belongToSameWarp(cTidSets, *ii, *jj, GPUConfig::warpsize)) {
        if (checkVolatileMissing(executor, state, *ii, *jj, 0, vmCond)) {
          volatilemiss = true;
          break;
        }
      }
    }
    if (volatilemiss) break;
  }

  // write-write sharing
  for (std::vector<MemoryAccessVec>::iterator ii = vmWriteSets.begin();
       ii != vmWriteSets.end(); ii++) {
    std::vector<MemoryAccessVec>::iterator jj = ii;
    jj++;
    for (;jj != vmWriteSets.end(); jj++) {
      if (belongToSameWarp(cTidSets, *ii, *jj, GPUConfig::warpsize)) {
        if (checkVolatileMissing(executor, state, *ii, *jj, 1, vmCond)) {
          volatilemiss = true;
          break;
        }
      }
    }
    if (volatilemiss) break;
  }

  for (unsigned i = 0; i<vmReadSets.size(); i++) 
    vmReadSets[i].clear();
  vmReadSets.clear();

  for (unsigned i = 0; i<vmWriteSets.size(); i++) 
    vmWriteSets[i].clear();
  vmWriteSets.clear();

  return volatilemiss;
}

static bool checkValuesSame(Executor &executor, ExecutionState &state, 
                            const ref<Expr> &val1, const ref<Expr> &val2) {
  bool result = false;
  bool unknown = false;
  ref<Expr> expr = EqExpr::create(val1, val2); 

  bool success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, expr, result, unknown);
  if (success) {
    return result;
  } 
  return false;
}

static bool checkWWRace(Executor &executor, ExecutionState &state, 
                        MemoryAccessVec &vec1, MemoryAccessVec &vec2, 
                        bool withinwarp, ref<Expr> &raceCond, 
                        unsigned &queryNum) {
  if (!accessSameMemoryRegion(executor, state, 
                              vec1.begin()->mo->getBaseExpr(), 
                              vec2.begin()->mo->getBaseExpr())) 
    return false;
  
  if (withinwarp) {
    for (MemoryAccessVec::iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
      unsigned tid1 = ii->tid;
      Expr::Width width1 = ii->width;
      MemoryAccessVec::iterator jj = ii;
      jj++;
      for (; jj != vec1.end(); jj++) {
        unsigned tid2 = jj->tid;
        Expr::Width width2 = jj->width;
        
        if (!isBothAtomic(*ii, *jj) 
             && checkConflictExprs(executor, state, raceCond, queryNum,
                                   ii->offset, width1, jj->offset, width2)) {
          bool benign = false;
          if (!checkValuesSame(executor, state, ii->val, jj->val)) {
	    GKLEE_INFO2 << "Within a warp, threads " << tid1 << " and " << tid2
	  	        << " incur a Write-Write race (Actual) on " << std::endl;
          } else {
            // If the written values are same, 'benign' race conditions...
	    GKLEE_INFO2 << "Within a warp, threads " << tid2 << " and " << tid1
	  	        << " incur a Write-Write race with the same value (Benign) on " 
                        << std::endl;
            benign = true;
          }
	  ii->dump(executor, state, raceCond);
          jj->dump(executor, state, raceCond);
	  if (Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, 
                                                     ii->instr, jj->instr, benign ? "wwrwb" : "wwrw");
          if (benign) return false;
	  else return true;
        }
      }
    }
  } else {
    for (MemoryAccessVec::iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
      unsigned tid1 = ii->tid;
      Expr::Width width1 = ii->width;

      for (MemoryAccessVec::iterator jj = vec2.begin(); jj != vec2.end(); jj++) {
        unsigned tid2 = jj->tid;
        Expr::Width width2 = jj->width;
        if (!isBothAtomic(*ii, *jj)
             && checkConflictExprs(executor, state, raceCond, queryNum, 
                                   ii->offset, width1, jj->offset, width2)) {
          bool benign = false;
          if (!checkValuesSame(executor, state, ii->val, jj->val)) {
	    GKLEE_INFO2 << "Across different warps, threads " << tid1 << " and " << tid2
	   	        << " incur a Write-Write race (Actual) on " << std::endl;
          } else {
            // If the written values are same, 'benign' race conditions...
	    GKLEE_INFO2 << "Across different warps, threads " << tid1 << " and " << tid2
	  	        << " incur a Write-Write race with same value (Benign) on " << std::endl;
            benign = true;
          }
	  ii->dump(executor, state, raceCond);
          jj->dump(executor, state, raceCond);
	  if (Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, ii->instr, jj->instr, benign? "wwraw" : "wwrawb");
          if (benign) return false;
	  else return true;
        }
      }
    }
  }
  return false;
}

static bool checkRWRace(Executor &executor, ExecutionState &state, 
                        MemoryAccessVec &vec1, MemoryAccessVec &vec2, 
                        ref<Expr> &raceCond, unsigned &queryNum) {
  if (!accessSameMemoryRegion(executor, state, vec1.begin()->mo->getBaseExpr(), 
                              vec2.begin()->mo->getBaseExpr())) 
    return false;
  
  // Definitely different warps...
  for (MemoryAccessVec::iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
    unsigned tid1 = ii->tid;
    Expr::Width width1 = ii->width;

    for (MemoryAccessVec::iterator jj = vec2.begin(); jj != vec2.end(); jj++) {
      unsigned tid2 = jj->tid;
      Expr::Width width2 = jj->width;
      if (!isBothAtomic(*ii, *jj)
           && checkConflictExprs(executor, state, raceCond, queryNum, 
                                 ii->offset, width1, jj->offset, width2)) {
        GKLEE_INFO2 << "Across different warps, threads " << tid1 << " and " << tid2
		    << " incur a Read-Write race (Actual) on " << std::endl;
	ii->dump(executor, state, raceCond);
        jj->dump(executor, state, raceCond);
	if(Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, ii->instr, jj->instr, "rwraw");
	return true;
      }
    }
  }
  return false;
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

static bool checkWWRacePureCS(Executor &executor, ExecutionState &state, 
                              MemoryAccessVec &vec1, MemoryAccessVec &vec2, 
                              bool withinBlock, ref<Expr> &raceCond, 
                              unsigned &queryNum) {
  if (withinBlock) {
    for (MemoryAccessVec::iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
      ref<Expr> base1 = ii->mo->getBaseExpr();
      ref<Expr> offset1 = ii->offset;
      unsigned tid1 = ii->tid;
      Expr::Width width1 = ii->width;

      MemoryAccessVec::iterator jj = ii;
      jj++;
      for (; jj != vec2.end(); jj++) {
        ref<Expr> base2 = jj->mo->getBaseExpr(); 
        ref<Expr> offset2 = jj->offset;
        unsigned tid2 = jj->tid;
        Expr::Width width2 = jj->width;

        if (tid1 != tid2) {
          if (!isBothAtomic(*ii, *jj)
               && accessSameMemoryRegion(executor, state, base1, base2)
                && checkConflictExprs(executor, state, raceCond, queryNum, 
                                      offset1, width1, offset2, width2)
                  && ii->val != jj->val
                    && fenceRelation(*ii, *jj, withinBlock)) {
 	    if (Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, 
	    			                       ii->instr, jj->instr, "ww");
            GKLEE_INFO2 << "Under the pure canonical schedule, within the same block, "
                        << "threads " << tid1 << " and " << tid2
                        << " incur a Write-Write race (Actual) on ";
            ii->dump(executor, state, raceCond);
	    jj->dump(executor, state, raceCond);
            return true;
          }
        }
      }
    }
  } else {
    for (MemoryAccessVec::iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
      ref<Expr> base1 = ii->mo->getBaseExpr();
      ref<Expr> offset1 = ii->offset;
      unsigned tid1 = ii->tid;
      Expr::Width width1 = ii->width;

      for (MemoryAccessVec::iterator jj = vec2.begin(); jj != vec2.end(); jj++) {
        ref<Expr> base2 = jj->mo->getBaseExpr(); 
        ref<Expr> offset2 = jj->offset;
        unsigned tid2 = jj->tid;
        Expr::Width width2 = jj->width;

        if (tid1 != tid2) {
          if (!isBothAtomic(*ii, *jj)
               && accessSameMemoryRegion(executor, state, base1, base2)
                && checkConflictExprs(executor, state, raceCond, queryNum, 
                                      offset1, width1, offset2, width2)
                 && ii->val != jj->val) {
 	    if (Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, 
	    			                       ii->instr, jj->instr, "ww");
            GKLEE_INFO2 << "Under the pure canonical schedule, across different blocks, "
                        << "threads " << tid1 << " and " << tid2
                        << " incur a Write-Write race (Actual) on ";
            ii->dump(executor, state, raceCond);
	    jj->dump(executor, state, raceCond);
            return true;
          }
        }
      } 
    }
  }
  return false;
}

static bool checkRWRacePureCS(Executor &executor, ExecutionState &state, 
                              MemoryAccessVec &vec1, MemoryAccessVec &vec2, 
                              bool withinBlock, ref<Expr> &raceCond, unsigned &queryNum) {
  // check the Read-Write conflict first 
  for (MemoryAccessVec::iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
    ref<Expr> base1 = ii->mo->getBaseExpr();
    ref<Expr> offset1 = ii->offset;
    unsigned tid1 = ii->tid;
    Expr::Width width1 = ii->width;

    for (MemoryAccessVec::iterator jj = vec2.begin(); jj != vec2.end(); jj++) {
      ref<Expr> base2 = jj->mo->getBaseExpr();
      ref<Expr> offset2 = jj->offset;
      unsigned tid2 = jj->tid;
      Expr::Width width2 = jj->width;

      if (tid1 != tid2) {
        if (!isBothAtomic(*ii, *jj)
             && accessSameMemoryRegion(executor, state, base1, base2) 
              && checkConflictExprs(executor, state, raceCond, queryNum, 
                                    offset1, width1, offset2, width2)
                && fenceRelation(*ii, *jj, withinBlock)) {
          GKLEE_INFO2 << "Threads " << tid1 << " and " << tid2
                      << " incur a Write-Read race (Actual) on ";
          ii->dump(executor, state, raceCond);
	  jj->dump(executor, state, raceCond);
 	  if (Emacs) AddressSpace::dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, 
                                                     ii->instr, jj->instr, "rw");
	  return true;
        }
      }
    }
  }
  return false;
}

static bool noThreadIntersection(const MemoryAccessVec &vec1, 
                                 const MemoryAccessVec &vec2) {
  for (MemoryAccessVec::const_iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
    unsigned tid1 = ii->tid; 
    for (MemoryAccessVec::const_iterator jj = vec2.begin(); jj != vec2.end(); jj++) {
      unsigned tid2 = jj->tid;

      if (tid1 == tid2) {
        return false;
      } 
    }
  }
  return true;
}

// Ensure that instructions belong to different BBs ... 
static bool belongToDifferentBB(const MemoryAccessVec &vec1, const MemoryAccessVec &vec2) {
  llvm::Instruction *inst1 = vec1.begin()->instr;
  llvm::Instruction *inst2 = vec2.begin()->instr;

  std::string func1Name = inst1->getParent()->getParent()->getName().str();
  std::string func2Name = inst2->getParent()->getParent()->getName().str();
  llvm::BasicBlock *bb1 = inst1->getParent();
  llvm::BasicBlock *bb2 = inst2->getParent();

  if (func1Name.compare(func2Name) == 0) {
    if (bb1 == bb2) 
      return false;
    else 
      return true;
  } else return true;
}
 
//static 
void AddressSpace::dumpEmacsInfoVect(unsigned bid1, unsigned bid2, 
				     unsigned tid1, unsigned tid2, 
				     const Instruction* i, const Instruction* i2,
				     std::string errorInfo){
  MDNode *N = NULL;
  if(i) N = i->getMetadata("dbg");
  MDNode *N2 = NULL;
  if(i2) N2 = i2->getMetadata("dbg");
  if(N && N2){
    DILocation Loc(N); 
    DILocation Loc2(N2);
    GKLEE_INFO << std::flush << "emacs:" << errorInfo << ":" 
	      << bid1 << ":"
	      << tid1 << ":"
	      << Loc.getDirectory().str() << "/" 
	      << Loc.getFilename().str() << ":"
	      << Loc.getLineNumber() << ":"
	      << bid2 << ":" 
	      << tid2 << ":"
	      << Loc2.getDirectory().str() << "/" 
	      << Loc2.getFilename().str() << ":"
	      << Loc2.getLineNumber() 
	      << std::endl << std::flush;
  }else if(N){
      DILocation Loc(N);
      GKLEE_INFO << std::flush << "emacs:" << errorInfo << ":" 
		<< bid1 << ":"
		<< tid1 << ":"
		<< Loc.getDirectory().str() << "/" 
		<< Loc.getFilename().str() << ":"
		<< Loc.getLineNumber() << ":"
		<< bid2 << ":"
		<< tid2 << "::" 
		<< std::endl << std::flush;
  }else{
    GKLEE_INFO << "Error getting location info for Emacs for " << errorInfo 
	      << ":t1:" << tid1 << ":t2:" << tid2 << ":b1:" << bid1 << ":b2:" << bid2 << std::endl;
  }
}

bool AddressSpace::belongToSameDivergenceRegion(const MemoryAccess &access1, 
                                                const MemoryAccess &access2,
                                                std::vector<CorrespondTid> &cTidSets,
                                                std::vector< std::vector<BranchDivRegionSet> > &divRegionSets,
                                                std::vector<SameInstVec> &sameInstSets) {
  unsigned tid1 = access1.tid;
  unsigned tid2 = access2.tid;

  unsigned repTid1 = 0;
  unsigned repTid2 = 0;  
  bool accessSame = hasSameExecTrace(cTidSets, sameInstSets, 
                                     tid1, repTid1, tid2, repTid2);
  if (accessSame)
    return false;
  else {
    unsigned warpNum = cTidSets[tid1].warpNum;
    std::vector<BranchDivRegionSet> &branchDivRegionSet = divRegionSets[warpNum];

    for (unsigned i = 0; i < branchDivRegionSet.size(); i++) {
      int idx1 = -1;
      int idx2 = -1;

      std::vector<BranchDivRegionVec> &branchSets = branchDivRegionSet[i].branchSets;
      for (unsigned j = 0; j < branchSets.size(); j++) {
        std::vector<BranchDivRegion> &divRegionVec = branchSets[j].branchDivRegionVec;

        for (unsigned k = 0; k < divRegionVec.size(); k++) {
          if (divRegionVec[k].tid == access1.tid
               && inRegionBound(divRegionVec[k].regionStart,
                                divRegionVec[k].regionEnd,
                                access1.instSeqNum))
            idx1 = j;

          if (divRegionVec[k].tid == access2.tid
               && inRegionBound(divRegionVec[k].regionStart,
                                divRegionVec[k].regionEnd,
                                access2.instSeqNum))
            idx2 = j;
        }
      }

      if (idx1 == -1 || idx2 == -1)
        return false;
      else
        return idx1 != idx2;
    }

    return false;
  }
}
  
// Check whether these two instructions belong to same conditional range... 
bool AddressSpace::checkDivergeBranchRace(Executor &executor, ExecutionState &state,
                                          const MemoryAccessVec &vec1, const MemoryAccessVec &vec2,
                                          std::vector<CorrespondTid> &cTidSets,
                                          std::vector<InstAccessSet> &accessSets, 
                                          std::vector< std::vector<BranchDivRegionSet> > &warpsBranchDivRegionSets,
                                          std::vector<SameInstVec> &sameInstSets,
                                          bool isWW, ref<Expr> &raceCond, unsigned &queryNum) {
  for (MemoryAccessVec::const_iterator ii = vec1.begin(); ii != vec1.end(); ii++) {
    ref<Expr> addr1 = AddExpr::create(ii->mo->getBaseExpr(), ii->offset);
    unsigned tid1 = ii->tid;
    Expr::Width width1 = ii->width;

    for (MemoryAccessVec::const_iterator jj = vec2.begin(); jj != vec2.end(); jj++) {
      if (belongToSameDivergenceRegion(*ii, *jj, cTidSets, warpsBranchDivRegionSets, sameInstSets)) {
        ref<Expr> addr2 = AddExpr::create(jj->mo->getBaseExpr(), jj->offset);
        unsigned tid2 = jj->tid;
        Expr::Width width2 = jj->width;
        if (!isBothAtomic(*ii, *jj)
             && accessSameMemoryRegion(executor, state, ii->mo->getBaseExpr(), jj->mo->getBaseExpr())
              && checkConflictExprs(executor, state, raceCond, queryNum, 
                                    addr1, width1, addr2, width2)) {
          bool benign = false;
          if (isWW) {
            if (!checkValuesSame(executor, state, ii->val, jj->val)) {
	      GKLEE_INFO2 << "Within a warp, because of branch divergence, threads " << tid2 << " and " << tid1
	       	          << " incur a Write-Write race (Actual) on " << std::endl;;
            } else {
	      GKLEE_INFO2 << "Within a warp, because of branch divergence, threads " << tid2 << " and " << tid1
	       	          << " incur a Write-Write race with the same value (Benign) on " << std::endl;;
              benign = true;
            }
          } else {
	    GKLEE_INFO2 << "Within a warp, because of branch divergence, threads " << tid2 << " and " << tid1
	    	        << " incur a Read-Write race (Actual) on " << std::endl;;
          }
	  ii->dump(executor, state, raceCond);
          jj->dump(executor, state, raceCond);	  
	  if (Emacs){
	    if(isWW) dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, ii->instr, jj->instr, benign ? "wwbdb" : "wwbd");
	    else dumpEmacsInfoVect(ii->bid, jj->bid, tid1, tid2, ii->instr, jj->instr, "rwbd");
	  }
          if (benign) return false;
          else return true;
        }
      }
    }
  }
  return false;
}

void AddressSpace::constructGlobalMemAccessSet(Executor &executor, ExecutionState &state,
                                               std::vector<CorrespondTid> &cTidSets, 
                                               std::vector<InstAccessSet> &instAccessSets, 
                                               std::vector<RefDivRegionSetVec> &divRegionSets, 
                                               std::vector<SameInstVec> &sameInstVecSets, 
                                               unsigned BINum) {
  if (SimdSchedule) {
    MemoryAccessVec tmpReadSet = readSet;
    MemoryAccessVec tmpWriteSet = writeSet;

    unsigned tmpWarpNum = 0;
    for (unsigned i = 0; i < cTidSets.size(); i++) {
      if (cTidSets[i].warpNum == tmpWarpNum) {
        MemAccessSets[tmpWarpNum].push_back(MemoryAccessSet(cTidSets[i].rBid, tmpWarpNum, BINum));
        tmpWarpNum++;
      }
    }
    // read set
    while (!tmpReadSet.empty()) {
      MemoryAccessVec readVec;
      AddressSpaceUtil::constructTmpRWSet(executor, state, tmpReadSet, readVec, cTidSets, 
                                          instAccessSets, divRegionSets, sameInstVecSets, 
                                          GPUConfig::warpsize);  
      unsigned tid = readVec.begin()->tid;
      unsigned warpNum = cTidSets[tid].warpNum;
      unsigned size = MemAccessSets[warpNum].size()-1;
      assert(MemAccessSets[warpNum][size].biNum == BINum && "constructGlobalMemAccessSet failed");
      MemAccessSets[warpNum][size].readVecSet.push_back(readVec);
    }
    // write set
    while (!tmpWriteSet.empty()) {
      MemoryAccessVec writeVec;
      AddressSpaceUtil::constructTmpRWSet(executor, state, tmpWriteSet, writeVec, cTidSets, 
                                          instAccessSets, divRegionSets, sameInstVecSets, 
                                          GPUConfig::warpsize);
      unsigned tid = writeVec.begin()->tid;
      unsigned warpNum = cTidSets[tid].warpNum;
      unsigned size = MemAccessSets[warpNum].size()-1;
      assert(MemAccessSets[warpNum][size].biNum == BINum && "constructGlobalMemAccessSet failed");
      MemAccessSets[warpNum][size].writeVecSet.push_back(writeVec);
    }
  } else {
    for (unsigned i = 0; i < MemAccessSetsPureCS.size(); i++)
      MemAccessSetsPureCS[i].push_back(MemoryAccessSetPureCS(i, BINum));

    unsigned size = MemAccessSetsPureCS[0].size()-1;
    // read set
    for (MemoryAccessVec::iterator vi = readSet.begin(); 
         vi != readSet.end(); vi++) {
      MemAccessSetsPureCS[(*vi).bid][size].readSet.push_back(MemoryAccess(*vi)); 
    }   

    // write set
    for (MemoryAccessVec::iterator vi = writeSet.begin(); 
         vi != writeSet.end(); vi++) {
      MemAccessSetsPureCS[(*vi).bid][size].writeSet.push_back(MemoryAccess(*vi)); 
    }
  }
}

static bool checkSetRaceInGlobal(Executor &executor, ExecutionState &state, 
                                 std::vector<MemoryAccessVec> &VecSet1, 
                                 std::vector<MemoryAccessVec> &VecSet2, 
                                 ref<Expr> &raceCond, bool isWW, unsigned &queryNum) {
  bool race = false;

  for (std::vector<MemoryAccessVec>::iterator ii = VecSet1.begin();
       ii != VecSet1.end(); ii++) { 
    for (std::vector<MemoryAccessVec>::iterator jj = VecSet2.begin(); 
         jj != VecSet2.end(); jj++) {
      if (isWW) {
        if (checkWWRace(executor, state, *ii, *jj, false, raceCond, queryNum)) 
          race = true;
      } else {
        if (checkRWRace(executor, state, *ii, *jj, raceCond, queryNum)) 
          race = true;
      }
      if (race) break;
    }
    if (race) break;
  }
  return race;
}

bool AddressSpace::hasRaceInGlobalWithinSameBlockPureCS(Executor &executor, ExecutionState &state) {
  bool hasRace = false;

  for (unsigned i = 0; i < MemAccessSetsPureCS.size(); i++) {
    unsigned size = MemAccessSetsPureCS[i].size();
    if (size > 0) {
      ref<Expr> cond = ConstantExpr::create(1, Expr::Bool);
      unsigned queryNum = 0;
      unsigned BINum = MemAccessSetsPureCS[i][size-1].biNum;
      if (checkWWRacePureCS(executor, state, MemAccessSetsPureCS[i][size-1].writeSet,
                            MemAccessSetsPureCS[i][size-1].writeSet, true, cond, queryNum)) {
        GKLEE_INFO << "Under pure canonical schedule, a write-write race is found from BI " 
                   << BINum << " of the block " << i << std::endl;
        hasRace = true;
        break;
      }
      if (checkRWRacePureCS(executor, state, MemAccessSetsPureCS[i][size-1].readSet, 
                            MemAccessSetsPureCS[i][size-1].writeSet, true, cond, queryNum)) {
        GKLEE_INFO << "Under pure canonical schedule, a read-write race is found from BI "
                   << BINum << " of the block " << i << std::endl; 
        hasRace = true;
        break;
      }
    }
  }

  return hasRace;
}

bool AddressSpace::hasRaceInGlobalAcrossBlocksPureCS(Executor &executor, ExecutionState &state) {
  bool hasRace = false;

  for (std::vector<MemoryAccessSetVecPureCS>::iterator ii = MemAccessSetsPureCS.begin();
       ii != MemAccessSetsPureCS.end(); ii++) {
    std::vector<MemoryAccessSetVecPureCS>::iterator jj = ii;
    jj++;
    for (; jj != MemAccessSetsPureCS.end(); jj++) {
      for (unsigned i = 0; i < (*ii).size(); i++) {
        for (unsigned j = 0; j < (*jj).size(); j++) {
          ref<Expr> cond = ConstantExpr::create(1, Expr::Bool);
          unsigned num = 0;
          if (checkWWRacePureCS(executor, state, (*ii)[i].writeSet,
                                (*jj)[j].writeSet, false, cond, num)) {
            GKLEE_INFO << "One thread at BI " << (*ii)[i].biNum << " of Block "
                      << (*ii)[i].bid << " incurs a write-write race with the thread at BI "
                      << (*jj)[j].biNum << " of Block " << (*jj)[j].bid << std::endl;
            hasRace = true;
          }
          if (checkRWRacePureCS(executor, state, (*ii)[i].readSet,
                                (*jj)[j].writeSet, false, cond, num)) {
            GKLEE_INFO << "One thread at BI " << (*ii)[i].biNum << " of Block "
                      << (*ii)[i].bid << " incurs a read-write race with the thread at BI "
                      << (*jj)[j].biNum << " of Block " << (*jj)[j].bid << std::endl;
            hasRace = true;
          }
          if (checkRWRacePureCS(executor, state, (*ii)[i].writeSet,
                                (*jj)[j].readSet, false, cond, num)) {
            GKLEE_INFO << "One thread at BI " << (*ii)[i].biNum << " of Block "
                      << (*ii)[i].bid << " incurs a write-read race with the thread at BI "
                      << (*jj)[j].biNum << " of Block " << (*jj)[j].bid << std::endl;
            hasRace = true;
          }
          if (hasRace) break;
        }
        if (hasRace) break;
      }
      if (hasRace) break;
    }
    if (hasRace) break;
  }

  return hasRace;
}

bool AddressSpace::hasRaceInGlobalWithinSameBlock(Executor &executor, ExecutionState &state,
                                                  std::vector<CorrespondTid> &cTidSets, 
                                                  std::vector<InstAccessSet> &instAccessSets, 
                                                  std::vector<RefDivRegionSetVec> &divRegionSets, 
                                                  std::vector< std::vector<BranchDivRegionSet> > &warpsBranchDivRegionSets,
                                                  std::vector<SameInstVec> &sameInstVecSets, 
                                                  ref<Expr> &raceCond, unsigned &queryNum, unsigned BINum) {
  ref<Expr> expr;

  bool race  = false;
  // within a warp first...
  for (unsigned i = 0; i < MemAccessSets.size(); i++) {
    unsigned size = MemAccessSets[i].size();
    if (size > 0) {
      if (MemAccessSets[i][size-1].biNum == BINum) {
        std::vector<MemoryAccessVec> &rReadSets = MemAccessSets[i][size-1].readVecSet;
        std::vector<MemoryAccessVec> &rWriteSets = MemAccessSets[i][size-1].writeVecSet;
        // write-write ...
        for (std::vector<MemoryAccessVec>::iterator ii = MemAccessSets[i][size-1].writeVecSet.begin(); 
             ii != MemAccessSets[i][size-1].writeVecSet.end(); ii++) {
          if (checkWWRace(executor, state, *ii, *ii, true, raceCond, queryNum)) { 
            race = true;
            break;
          }
        }
        // write-write, different branches within same condition ...
        for (std::vector<MemoryAccessVec>::const_iterator ii = rWriteSets.begin();
             ii != rWriteSets.end(); ii++) {
          std::vector<MemoryAccessVec>::const_iterator jj = ii;
          jj++;
          for (; jj != rWriteSets.end(); jj++) {
            if (noThreadIntersection(*ii, *jj)
                  && belongToDifferentBB(*ii, *jj)) {
              if (checkDivergeBranchRace(executor, state, *ii, *jj, 
                                         cTidSets, instAccessSets, 
                                         warpsBranchDivRegionSets,
                                         sameInstVecSets, 
                                         true, raceCond, queryNum)) { 
                race = true; 
                break;
              }
            }
          } 
          if (race) break;
        }
        // read-write, different branches within same condition ... 
        for (std::vector<MemoryAccessVec>::const_iterator ii = rWriteSets.begin();
             ii != rWriteSets.end(); ii++) {
          for (std::vector<MemoryAccessVec>::const_iterator jj = rReadSets.begin(); 
               jj != rReadSets.end(); jj++) {
            if (noThreadIntersection(*ii, *jj)
                   && belongToDifferentBB(*ii, *jj)) {
              if (checkDivergeBranchRace(executor, state, *ii, *jj, 
                                         cTidSets, instAccessSets, 
                                         warpsBranchDivRegionSets,
                                         sameInstVecSets, 
                                         false, raceCond, queryNum)) { 
                race = true; 
                break;
              }
            }
          }
          if (race) break;
        }
      }
    }
    if (race) break;
  }

  // Across different warps within the same block...
  for (unsigned i = 0; i < MemAccessSets.size(); i++) {
    unsigned j = i+1;
    for (; j < MemAccessSets.size(); j++) {
      if (MemAccessSets[i].size() > 0 && MemAccessSets[j].size() > 0) {
        if (MemAccessSets[i].begin()->bid == MemAccessSets[j].begin()->bid) {
          unsigned size_i = MemAccessSets[i].size();
          unsigned size_j = MemAccessSets[j].size();

          if (MemAccessSets[i][size_i-1].biNum == BINum && 
                MemAccessSets[j][size_j-1].biNum == BINum) {
            // write-write 
            if (checkSetRaceInGlobal(executor, state, MemAccessSets[i][size_i-1].writeVecSet, 
                                     MemAccessSets[j][size_j-1].writeVecSet, raceCond, true, queryNum))
              race = true;
            // read-write
            if (checkSetRaceInGlobal(executor, state, MemAccessSets[i][size_i-1].readVecSet, 
                                     MemAccessSets[j][size_j-1].writeVecSet, raceCond, false, queryNum))
              race = true;
            if (checkSetRaceInGlobal(executor, state, MemAccessSets[i][size_i-1].writeVecSet, 
                                     MemAccessSets[j][size_j-1].readVecSet, raceCond, false, queryNum))
              race = true;

            break;
          }
        }
      }
      if (race) break;
    }
    if (race) break;
  }

  return race;
}

bool AddressSpace::hasRaceInGlobalAcrossBlocks(Executor &executor, ExecutionState &state,
                                               std::vector<CorrespondTid> &cTidSets, 
                                               ref<Expr> &raceCond, unsigned &queryNum) {
  ref<Expr> expr;
  bool race  = false;

  // between different warps across different blocks...
  for (unsigned i = 0; i < MemAccessSets.size(); i++) {
    unsigned j = i+1;
    for (; j < MemAccessSets.size(); j++) {
      if (MemAccessSets[i].size() > 0 && MemAccessSets[j].size() > 0) {
        if (MemAccessSets[i].begin()->bid != MemAccessSets[j].begin()->bid) {
          for (unsigned k = 0; k < MemAccessSets[i].size(); k++) {
            for (unsigned m = 0; m < MemAccessSets[j].size(); m++) {
              // write-write
              if (checkSetRaceInGlobal(executor, state, MemAccessSets[i][k].writeVecSet,
                                       MemAccessSets[j][m].writeVecSet, raceCond, true, queryNum))
                race = true;
              // read-write
              if (checkSetRaceInGlobal(executor, state, MemAccessSets[i][k].readVecSet, 
                                       MemAccessSets[j][m].writeVecSet, raceCond, false, queryNum))
                race = true;

              if (checkSetRaceInGlobal(executor, state, MemAccessSets[i][k].writeVecSet, 
                                       MemAccessSets[j][m].readVecSet, raceCond, false, queryNum))
                race = true;

              break;
            }
            if (race) break;
          }
        }
      }
      if (race) break;
    }
    if (race) break;
  }

  // clear those two sets...
  for (unsigned i = 0; i<MemAccessSets.size(); i++) 
    MemAccessSets[i].clear();

  return race;
}

// races in a single address space
bool AddressSpace::hasRaceInShare(Executor &executor, ExecutionState &state,
                                  std::vector<CorrespondTid> &cTidSets, 
                                  std::vector<InstAccessSet> &instAccessSets, 
                                  std::vector<RefDivRegionSetVec> &divRegionSets, 
                                  std::vector<SameInstVec> &sameInstVecSets, 
                                  std::vector< std::vector<BranchDivRegionSet> > &warpsBranchDivRegionSets,
                                  ref<Expr> &raceCond, unsigned &queryNum) {
  ref<Expr> expr;

  MemoryAccessVec tmpReadSet = readSet;
  MemoryAccessVec tmpWriteSet = writeSet;

  std::vector<MemoryAccessVec> rReadSets;
  std::vector<MemoryAccessVec> rWriteSets;

  bool race  = false;
  bool hasRace = false;
  // Handle read set first... 
  while (!tmpReadSet.empty()) {
    MemoryAccessVec vmReadVec;
    AddressSpaceUtil::constructTmpRWSet(executor, state, tmpReadSet, vmReadVec, cTidSets, 
                                        instAccessSets, divRegionSets, sameInstVecSets, 
                                        GPUConfig::warpsize);  
    rReadSets.push_back(vmReadVec);
  }

  // Handle write set
  while (!tmpWriteSet.empty()) {
    MemoryAccessVec vmWriteVec;
    AddressSpaceUtil::constructTmpRWSet(executor, state, tmpWriteSet, vmWriteVec, cTidSets, 
                                        instAccessSets, divRegionSets, sameInstVecSets, 
                                        GPUConfig::warpsize);
    rWriteSets.push_back(vmWriteVec);
  }
      
  // within a warp first...
  // write-write ...
  for (std::vector<MemoryAccessVec>::iterator ii = rWriteSets.begin(); 
       ii != rWriteSets.end(); ii++) {
    if (checkWWRace(executor, state, *ii, *ii, true, raceCond, queryNum)) { 
      race = true;
      break;
    }
  }

  // write-write, different branches within same condition ...
  for (std::vector<MemoryAccessVec>::iterator ii = rWriteSets.begin();
       ii != rWriteSets.end(); ii++) {
    std::vector<MemoryAccessVec>::iterator jj = ii;
    jj++;
    for (; jj != rWriteSets.end(); jj++) {
      if (belongToSameWarp(cTidSets, *ii, *jj, GPUConfig::warpsize) 
           && noThreadIntersection(*ii, *jj)
            && belongToDifferentBB(*ii, *jj)) {
        if (checkDivergeBranchRace(executor, state, *ii, *jj, 
                                   cTidSets, instAccessSets, 
                                   warpsBranchDivRegionSets,
                                   sameInstVecSets, 
                                   true, raceCond, queryNum)) {
          race = true; 
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) {
      hasRace = false;
      break;
    }
  }

  // read-write, different branches within same condition ... 
  for (std::vector<MemoryAccessVec>::const_iterator ii = rWriteSets.begin();
       ii != rWriteSets.end(); ii++) {
    for (std::vector<MemoryAccessVec>::const_iterator jj = rReadSets.begin(); 
         jj != rReadSets.end(); jj++) {
      if (belongToSameWarp(cTidSets, *ii, *jj, GPUConfig::warpsize) 
           && noThreadIntersection(*ii, *jj)
            && belongToDifferentBB(*ii, *jj)) {
        if (checkDivergeBranchRace(executor, state, *ii, *jj, 
                                   cTidSets, instAccessSets, 
                                   warpsBranchDivRegionSets,
                                   sameInstVecSets, 
                                   false, raceCond, queryNum)) {
          race = true; 
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) {
      hasRace = false;
      break;
    }
  }
 
  // Across different warps...
  // write-write... 
  for (std::vector<MemoryAccessVec>::iterator ii = rWriteSets.begin();
       ii != rWriteSets.end(); ii++) {
    std::vector<MemoryAccessVec>::iterator jj = ii;
    jj++;
    for (; jj != rWriteSets.end(); jj++) {
      if (belongToSameBlockNotSameWarp(*ii, *jj, cTidSets)) {
        if (checkWWRace(executor, state, *ii, *jj, false, raceCond, queryNum)) {
          race = true;
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) {
      hasRace = false;
      break;
    }
  }
  // read-write
  for (std::vector<MemoryAccessVec>::iterator ii = rReadSets.begin();
       ii != rReadSets.end(); ii++) {
    for (std::vector<MemoryAccessVec>::iterator jj = rWriteSets.begin(); 
         jj != rWriteSets.end(); jj++) {
      if (belongToSameBlockNotSameWarp(*ii, *jj, cTidSets)) {
        if (checkRWRace(executor, state, *ii, *jj, raceCond, queryNum)) { 
          race = true;
          hasRace = true;
          break;
        }
      }
    }
    if (hasRace) {
      hasRace = false;
      break;
    }
  }
  // clear those two sets...
  for (unsigned i = 0; i<rReadSets.size(); i++) 
    rReadSets[i].clear();
  rReadSets.clear();

  for (unsigned i = 0; i<rWriteSets.size(); i++) 
    rWriteSets[i].clear();
  rWriteSets.clear();

  return race;
}

// races under pure canonical schedule 
bool AddressSpace::hasRaceInSharePureCS(Executor &executor, ExecutionState &state, 
                                        ref<Expr> &raceCond, unsigned &queryNum) {
  bool wwRace = false;
  bool rwRace = false;
  // check the Read-Write conflict first 
  wwRace = checkRWRacePureCS(executor, state, writeSet, readSet, true, raceCond, queryNum);
  // check the Write-Write conflict then
  rwRace = checkWWRacePureCS(executor, state, writeSet, writeSet, true, raceCond, queryNum);

  return wwRace || rwRace;
}

// races in all the address spaces
bool HierAddressSpace::hasRaceInShare(Executor &executor, ExecutionState &state, 
                                      std::vector<CorrespondTid> &cTidSets,
                                      ref<Expr> &raceCond) {
  if (GPUConfig::check_level == 0)  // skip checking
    return false;

  unsigned bid = 0;
  for (std::vector<AddressSpace>::iterator ii = sharedMemories.begin(); 
       ii != sharedMemories.end(); ii++) {
    GKLEE_INFO << "\n********** Start checking races at SharedMemory " 
              << bid << " **********\n";
    if (GPUConfig::verbose > 0) 
      ii->dump(true);

    bool race = false;
    if (!SimdSchedule)
      race = ii->hasRaceInSharePureCS(executor, state, raceCond, raceQueryNum);
    else
      race = ii->hasRaceInShare(executor, state, cTidSets, 
                                instAccessSets, divRegionSets, 
                                sameInstVecSets, warpsBranchDivRegionSets,
                                raceCond, raceQueryNum);
    bid++;
    ii->clearAccessSet();
    if (race) return true;    
  }
  return false;
}

// races in the device or CPU address spaces 
bool HierAddressSpace::hasRaceInGlobal(Executor &executor, ExecutionState &state, 
                                       std::vector<CorrespondTid> &cTidSets, 
                                       ref<Expr> &raceCond, unsigned BINum, 
                                       bool is_end_GPU_barrier) {
  //unsigned warpsize = 32;
  if (GPUConfig::check_level > 1) {
    // Device memory 
    GKLEE_INFO << "\n********** Start checking races at Device Memory ********** \n";
    if (GPUConfig::verbose > 0) {
      deviceMemory.dump(true);
    }

    if (!SimdSchedule) {
      if (deviceMemory.hasRaceInGlobalWithinSameBlockPureCS(executor, state))
        return true;

      if (is_end_GPU_barrier) {
        bool hasRace = deviceMemory.hasRaceInGlobalAcrossBlocksPureCS(executor, state);
        deviceMemory.clearGlobalMemoryAccessSets();
        if (hasRace)
          return true;
      }

      deviceMemory.clearAccessSet();
    } else {
      if (deviceMemory.hasRaceInGlobalWithinSameBlock(executor, state, cTidSets, 
                                                      instAccessSets, divRegionSets, 
                                                      warpsBranchDivRegionSets,
                                                      sameInstVecSets, raceCond, 
                                                      raceQueryNum, BINum))
        return true;
      if (is_end_GPU_barrier) {
        bool hasRace = deviceMemory.hasRaceInGlobalAcrossBlocks(executor, state, cTidSets, raceCond, raceQueryNum);
        deviceMemory.clearGlobalMemoryAccessSets();
        if (hasRace)
          return true;
      }
      deviceMemory.clearAccessSet();
    }

    // CPU memory 
    /*GKLEE_INFO << "\n************************************** Start checking races at CPU Memory ************************************** \n";
    if (GPUConfig::verbose > 0) {
      cpuMemory.dump(true);
    }
    if (cpuMemory.hasRaceInGlobalWithinSameBlock(solver, constr, cTidSets, instAccessSets, divRegionSets, 
                                                 sameInstVecSets, raceCond, raceQueryNum, BINum))
      return true;
    if (is_end_GPU_barrier) {
      bool hasRace = cpuMemory.hasRaceInGlobalAcrossBlocks(solver, constr, cTidSets, raceCond, raceQueryNum);
      if (hasRace)
        return true;
    }
    cpuMemory.clearAccessSet();*/
  }
  return false;
}

static bool accessSameMemoryRegion(Executor &executor, ExecutionState &state, 
                                   ref<Expr> baseAddr1, ref<Expr> baseAddr2) {
  ref<Expr> cond = EqExpr::create(baseAddr1, baseAddr2);
  bool result;
  bool unknown = false;
  bool success = AddressSpaceUtil::evaluateQueryMustBeTrue(executor, state, cond, result, unknown);
  if (success) {
    return result;
  }
  return false;
}

static bool checkExploredInstSame(InstAccessSet &set1, InstAccessSet &set2) {
  unsigned size = set1.size();

  for (unsigned i = 0; i<size; i++) {
    if (!AddressSpaceUtil::isTwoInstIdentical(set1[i].inst, set2[i].inst))
      return false; 
  }
  return true;
}

static bool isTwoInstAccessSetSame(InstAccessSet &set1, InstAccessSet &set2) {
  if (set1.size() != set2.size())
    return false;
  else
    return checkExploredInstSame(set1, set2);
}

static bool isInSameHalfOrEntireWarp(std::vector <CorrespondTid> &cTidSets, 
                                     unsigned bid1, unsigned tid1, 
                                     unsigned bid2, unsigned tid2, 
                                     unsigned warpsize) {
  if (bid1 != bid2)
    return false;
  else {
    if (cTidSets[tid1].rTid/warpsize != cTidSets[tid2].rTid/warpsize)
      return false;
    else 
      return true;
  }
}

static bool findSameThreadFromTmpSet(MemoryAccessVec &tmpRWSet, unsigned tid) {
  for (unsigned i = 0; i<tmpRWSet.size(); i++) {
    if (tmpRWSet[i].tid == tid)
      return true;
  }
  return false;
}
 
static bool inRegionBound(unsigned lBound, unsigned rBound, unsigned instNum) {
  return (lBound <= instNum && instNum <= rBound);
}

static bool checkSameOrDivRegion(RefDivRegionSetVec &divRegionSet, 
                                 unsigned otherTid, 
                                 unsigned instNum, unsigned bound, 
                                 unsigned &divNum, bool &above, unsigned &relPos) {
  bool same = false;

  for (unsigned i = 0; i < divRegionSet.size(); i++) {
    if (divRegionSet[i].otherTid == otherTid) {
      unsigned idx = 0;
      for (DivRegionSet::iterator ii = divRegionSet[i].regionSet.begin();
           ii != divRegionSet[i].regionSet.end(); ii++, idx++) {
        if (!ii->isEmpty && inRegionBound(ii->startIdx, ii->endIdx, instNum)) {
          // find this instruction from divergent region ...
          divNum = idx;
          break;
        }
        // For each divergence region ..
        if (ii->startIdx != 0) {
          unsigned start = 0;
          if (idx != 0)
            start = (divRegionSet[i].regionSet[idx-1].isEmpty)? 
                                     divRegionSet[i].regionSet[idx-1].endIdx: 
                                     divRegionSet[i].regionSet[idx-1].endIdx+1;

          if (inRegionBound(start, ii->startIdx-1, instNum)) {
            divNum = idx;
            same = true;
            above = true;
            relPos = instNum - start;
            break;
          }
        }
        // Only for the last divergence region ..
        if (idx == divRegionSet[i].regionSet.size()-1 
              && ii->endIdx != bound) {
          unsigned start = (ii->isEmpty)? ii->endIdx : ii->endIdx+1;
          if (inRegionBound(start, bound, instNum)) {
            divNum = idx;
            same = true;
            above = false;
            relPos = instNum - start;
            break;
          }
        }
      }
      break;
    }
  }
 
  return same;
}

static bool hasSameExecTrace(std::vector<CorrespondTid> &cTidSets, 
                             std::vector<SameInstVec> &sameInstVecSets, 
                             unsigned tid1, unsigned &repTid1, 
                             unsigned tid2, unsigned &repTid2) {
  unsigned warpNum = cTidSets[tid1].warpNum;
  unsigned idx = 0;
  unsigned tidSet1 = 0;
  unsigned tidSet2 = 0;

  for (SameInstVec::iterator ii = sameInstVecSets[warpNum].begin(); 
       ii != sameInstVecSets[warpNum].end(); ii++, idx++) {
    for (std::vector<unsigned>::iterator jj = ii->begin();
         jj != ii->end(); jj++) {
      if (*jj == tid1)
        tidSet1 = idx;
      if (*jj == tid2)
        tidSet2 = idx; 
    }
  }

  repTid1 = sameInstVecSets[warpNum][tidSet1][0];
  repTid2 = sameInstVecSets[warpNum][tidSet2][0];

  if (tidSet1 == tidSet2)
    return true;
  else
    return false;
}

static int findDivRegionNum(RefDivRegionSetVec &divRegionSet, unsigned otherTid) {
  for (unsigned i = 0; i < divRegionSet.size(); i++) {
    if (divRegionSet[i].otherTid == otherTid)
      return divRegionSet[i].regionSet.size(); 
  }
  return -1; 
}

static bool isInMatchRegion(std::vector<CorrespondTid> &cTidSets, 
                            std::vector<InstAccessSet> &instAccessSets, 
                            std::vector<RefDivRegionSetVec> &divRegionSets, 
                            std::vector<SameInstVec> &sameInstVecSets,
                            const MemoryAccess &access1, 
                            const MemoryAccess &access2) {
  unsigned tid1 = access1.tid;
  unsigned tid2 = access2.tid; 

  unsigned repTid1 = 0;
  unsigned repTid2 = 0;
 
  if (hasSameExecTrace(cTidSets, sameInstVecSets, tid1, repTid1, tid2, repTid2)) {
    if (access1.instSeqNum == access2.instSeqNum) return true;
    else return false;
  } else {
    findDivRegionNum(divRegionSets[tid1], repTid2);
    findDivRegionNum(divRegionSets[tid2], repTid1);

    unsigned divNum1 = 0;
    bool above1 = false;
    unsigned relPos1 = 0;
    bool same1 = checkSameOrDivRegion(divRegionSets[tid1], repTid2, access1.instSeqNum, 
                                      instAccessSets[tid1].size()-1, divNum1, above1, relPos1);

    unsigned divNum2 = 0;
    bool above2 = false;
    unsigned relPos2 = 0;
    bool same2 = checkSameOrDivRegion(divRegionSets[tid2], repTid1, access2.instSeqNum, 
                                      instAccessSets[tid2].size()-1, divNum2, above2, relPos2);

    if (same1 != same2) 
      return false; 
    else {
      if (same1) {
        // "access1" and "access2" both access the common parts. 
        if (GPUConfig::verbose > 0) {
          GKLEE_INFO << "(tid1, repTid2, divNum1, above1, relPos1): " << "(" << tid1 
                    << ", " << repTid2 << ", " << divNum1 << ", " << above1 << ", " << relPos1 
                    << ")" << std::endl; 
          GKLEE_INFO << "(tid2, resTid1, divNum2, above2, relPos2): " << "(" << tid2
                    << ", " << repTid1 << ", " << divNum2 
                    << ", " << above2 << ", " << relPos2 << ")" << std::endl; 
        }
        if (divNum1 == divNum2 && above1 == above2 && relPos1 == relPos2) 
          return true;
        else return false; 
      } else return false;
    }
  }
}

void AddressSpaceUtil::constructTmpRWSet(Executor &executor, ExecutionState &state, 
                                         MemoryAccessVec &rwSet, MemoryAccessVec &tmpRWSet, 
                                         std::vector<CorrespondTid> &cTidSets, 
                                         std::vector<InstAccessSet> &instAccessSets, 
                                         std::vector<RefDivRegionSetVec> &divRegionSets, 
                                         std::vector<SameInstVec> &sameInstVecSets, 
                                         unsigned warpsize) {
  if (rwSet.empty()) return;
  MemoryAccessVec::iterator begin = rwSet.begin();
  MemoryAccess *tmpAccess = new MemoryAccess(*begin);
  for (MemoryAccessVec::iterator ii = rwSet.begin(); ii != rwSet.end();) {
    // Same memory region, same instruction
    // within same half or entire warp, only one thread with 
    // same id included in the set...  
    if (isInSameHalfOrEntireWarp(cTidSets, tmpAccess->bid, tmpAccess->tid, ii->bid, ii->tid, warpsize)
         && AddressSpaceUtil::isTwoInstIdentical(tmpAccess->instr, ii->instr)
          && isInMatchRegion(cTidSets, instAccessSets, divRegionSets, sameInstVecSets, *tmpAccess, *ii) 
           && accessSameMemoryRegion(executor, state, tmpAccess->mo->getBaseExpr(), ii->mo->getBaseExpr()) 
            && !findSameThreadFromTmpSet(tmpRWSet, ii->tid)) {
      // The current element is in the same warp with tmpAccess 
      MemoryAccess access(*ii);
      tmpRWSet.push_back(access);
      ii = rwSet.erase(ii);
    }
    else ++ii;
  }

  delete tmpAccess;
}

void HierAddressSpace::constructGlobalMemAccessSets(Executor &executor, ExecutionState &state, 
                                                    std::vector<CorrespondTid> &cTidSets, unsigned BINum) {
  // device memory
  if (!deviceMemory.readSet.empty() || !deviceMemory.writeSet.empty())
    deviceMemory.constructGlobalMemAccessSet(executor, state, cTidSets, instAccessSets, 
                                             divRegionSets, sameInstVecSets, BINum);
  // cpu memory
  if (!cpuMemory.readSet.empty() || !cpuMemory.writeSet.empty())
    cpuMemory.constructGlobalMemAccessSet(executor, state, cTidSets, instAccessSets, 
                                          divRegionSets, sameInstVecSets, BINum);
}

static void updateThreadWarpMark(std::vector<InstAccessSet> &sets, 
                                 std::vector<SameInstVec> &sameInstSet, 
                                 unsigned start, unsigned end)  {
  SameInstVec sameVec;
  for (unsigned i = start; i <= end; i++) {
    bool findSame = false;
    unsigned j = 0;
    for (; j < sameVec.size(); j++) {
      std::vector<unsigned>::iterator ti = sameVec[j].begin();
      if (isTwoInstAccessSetSame(sets[i], sets[*ti])){ 
        findSame = true;
        break;
      }
    }
    if (findSame)
      sameVec[j].push_back(i);
    else {
      std::vector<unsigned> tmpVec;
      tmpVec.push_back(i);
      sameVec.push_back(tmpVec);
    }
  }
  sameInstSet.push_back(sameVec);
}

static void dumpInstSetAndDivergenceRegion(std::vector<InstAccessSet> &accessSets, 
                                           std::vector<RefDivRegionSetVec> &divRegionSets) { 
  unsigned size = accessSets.size();
  for (unsigned i = 0; i < size; i++) {
    GKLEE_INFO << "Tid " << i << ":" << std::endl;
    unsigned idx = 0;
    for (InstAccessSet::iterator ii = accessSets[i].begin(); 
         ii != accessSets[i].end(); ii++, idx++) {
      GKLEE_INFO << "idx: " << idx << " ";
      ii->dump();
    }
    
    GKLEE_INFO << "**********" << std::endl;
    for (RefDivRegionSetVec::iterator ii = divRegionSets[i].begin();
         ii != divRegionSets[i].end(); ii++) {
      GKLEE_INFO << "Other tid: " << ii->otherTid << std::endl;
      idx = 0;
      for (DivRegionSet::iterator jj = (ii->regionSet).begin(); 
           jj != (ii->regionSet).end(); jj++, idx++) {
        if (jj->isEmpty) {
          GKLEE_INFO << "div BB idx <" << idx << "," << 0 << ">"
                    << ": (" << jj->bbStart << ", " << jj->bbEnd << ")" 
                    << " | (" << jj->startIdx << ", " << jj->endIdx << ")" 
                    << std::endl;
        } else {
          GKLEE_INFO << "div BB idx <" << idx << ","  << 1 << ">" 
                    << ": (" << jj->bbStart << ", " << jj->bbEnd << ")" 
                    << " | (" << jj->startIdx << ", " << jj->endIdx << ")" 
                    << std::endl;
        }
      }
    }
    GKLEE_INFO << "**********" << std::endl;
  }
}

void HierAddressSpace::dumpWarpsBranchDivRegionSets() {
  // Dump all threads' instructions 
  for (unsigned i = 0; i < instAccessSets.size(); i++) {
    GKLEE_INFO << "Tid " << i << ":" << std::endl;
    unsigned idx = 0;
    for (InstAccessSet::iterator ii = instAccessSets[i].begin(); 
         ii != instAccessSets[i].end(); ii++, idx++) {
      GKLEE_INFO << "idx " << idx << ":" << std::endl;
      ii->inst->dump();
    }
  }

  for (unsigned i = 0; i < warpsBranchDivRegionSets.size(); i++) {
    GKLEE_INFO << "Warp " << i << ": " << std::endl;
    for (unsigned j = 0; j < warpsBranchDivRegionSets[i].size(); j++) {
      GKLEE_INFO << "=================  Branch " << j << ": ================== " << std::endl;
      warpsBranchDivRegionSets[i][j].brInst->dump();
      GKLEE_INFO << "Post dominator: " << std::endl;
      warpsBranchDivRegionSets[i][j].postDominator->dump();
      if (warpsBranchDivRegionSets[i][j].isCondBr) {
        GKLEE_INFO << "True path: " << std::endl;
        BranchDivRegionVec &truePath = warpsBranchDivRegionSets[i][j].branchSets[0];
        std::vector<BranchDivRegion> &trueSet = truePath.branchDivRegionVec;

        for (unsigned k = 0; k < trueSet.size(); k++) {
          GKLEE_INFO << "<tid: " << trueSet[k].tid << ", l: " << trueSet[k].regionStart
                    << ", r: " << trueSet[k].regionEnd << ">" << std::endl;
        }
        GKLEE_INFO << "False path: " << std::endl;
        BranchDivRegionVec &falsePath = warpsBranchDivRegionSets[i][j].branchSets[1];
        std::vector<BranchDivRegion> &falseSet = falsePath.branchDivRegionVec;

        for (unsigned k = 0; k < falseSet.size(); k++) {
          GKLEE_INFO << "<tid: " << falseSet[k].tid << ", l: " << falseSet[k].regionStart
                    << ", r: " << falseSet[k].regionEnd << ">" << std::endl;
        }
      } else {
        std::vector<BranchDivRegionVec> &branchSets = warpsBranchDivRegionSets[i][j].branchSets;
        for (unsigned k = 0; k < branchSets.size(); k++) {
          GKLEE_INFO << "The " << k << "th path for switch instruction: " << std::endl;
          std::vector<BranchDivRegion> &pathSet = branchSets[k].branchDivRegionVec;

          for (unsigned m = 0; m < pathSet.size(); m++) {
            GKLEE_INFO << "<tid: " << pathSet[m].tid << ", l: " << pathSet[m].regionStart
                      << ", r: " << pathSet[m].regionEnd << ">" << std::endl;
          }
        }
      }
    }
  }
}

void HierAddressSpace::ensureThreadsDivergenceRegion(std::vector<CorrespondTid>& cTidSets) {
  unsigned start = 0;
  unsigned end = 0;
  unsigned tidIdx = 0;
  unsigned warpNum = 0;
  
  if (GPUConfig::verbose > 0) {
    for (unsigned i = 0; i < cTidSets.size(); i++) {
      GKLEE_INFO << "Tid " << i << ":" << std::endl;
      unsigned idx = 0;
      for (InstAccessSet::iterator ii = instAccessSets[i].begin(); 
           ii != instAccessSets[i].end(); ii++, idx++) {
        GKLEE_INFO << "idx: " << idx << " ";
        ii->dump();
      }
    }
  }
    
  // Create the warp mark set ... 
  for (std::vector<CorrespondTid>::iterator vi = cTidSets.begin(); 
       vi != cTidSets.end(); vi++, tidIdx++) {
    if (vi->warpNum == warpNum) {
      end = tidIdx;
    } else {
      updateThreadWarpMark(instAccessSets, sameInstVecSets, start, end);
      warpNum = vi->warpNum;
      // New warp starts... 
      start = tidIdx;
      end = tidIdx;
    }
  }
  updateThreadWarpMark(instAccessSets, sameInstVecSets, start, end);

  if (SimdSchedule) {
    std::vector<BranchDivRegionSet> tmpIDRS = branchDivRegionSets;
    warpsBranchDivRegionSets.push_back(tmpIDRS);
    branchDivRegionSets.clear();
  }

  unsigned bid = 0;
  for (std::vector<AddressSpace>::iterator ii = sharedMemories.begin(); 
       ii != sharedMemories.end(); ii++, bid++) {
    unsigned bStart = bid * GPUConfig::block_size;
    unsigned bEnd = bStart + GPUConfig::block_size - 1;
    // forwarding explore all threads in the block
    forwardingExploreInstSet(bStart, bEnd);
    constructDivergRegionSets(cTidSets, bStart, bEnd); 
  }

  // For debugging ...
  if (GPUConfig::verbose > 0) {
    GKLEE_INFO << "********** Divergence Region **********" << std::endl;
    dumpInstSetAndDivergenceRegion(instAccessSets, divRegionSets);
    GKLEE_INFO << "***************************************" << std::endl;
  }
}

bool HierAddressSpace::hasVolatileMissing(Executor &executor, ExecutionState &state, 
                                          std::vector<CorrespondTid> &cTidSets) {
  int i = 0;
  for (std::vector<AddressSpace>::iterator ii = sharedMemories.begin(); 
       ii != sharedMemories.end(); ii++) {
    GKLEE_INFO << "\n********** Start checking volatile missing at SharedMemory " 
              << i++ << " ********** \n";
    if (GPUConfig::verbose > 0) {
      ii->dump(true);
    }

    bool vm = ii->hasVolatileMissing(executor, state, cTidSets, instAccessSets, 
                                     divRegionSets, sameInstVecSets, vmCondComb);
    if (vm) {
      hasVM = true;
      return true;
    }
  }
  return false;
}

//******************************************************************************************
// Hierachical Memory Architecture
//********************************************************************************************

HierAddressSpace::HierAddressSpace() {
  numWDBI = 0;
  numWD = 0;
  
  raceQueryNum = 0;
  mcQueryNum = 0;
  bcQueryNum = 0;

  cpuMemory.ctype = GPUConfig::HOST;
  deviceMemory.ctype = GPUConfig::DEVICE;

  hasBC = false;
  hasNoMC = false;
  hasVM = false;

  bcCondComb = ConstantExpr::create(1, Expr::Bool);
  nonMCCondComb = ConstantExpr::create(1, Expr::Bool);
  vmCondComb = ConstantExpr::create(1, Expr::Bool);

  unsigned num_blocks = UseSymbolicConfig? GPUConfig::sym_num_blocks : 
                                           GPUConfig::num_blocks;
  unsigned num_threads = UseSymbolicConfig? GPUConfig::sym_num_threads:
                                            GPUConfig::num_threads;

  AddressSpace space;
  for (unsigned k = 0; k < num_blocks; k++) {
    sharedMemories.push_back(space);
    sharedMemories.back().ctype = GPUConfig::SHARED;
  }

  for (unsigned k = 0; k < num_threads; k++) {
    localMemories.push_back(space);  
    localMemories.back().ctype = GPUConfig::LOCAL;
  }

  InstAccessSet instSet; 
  RefDivRegionSetVec regionSet;
  BBAccessSet bbSet;
  for (unsigned k = 0; k < num_threads; k++) {
    instAccessSets.push_back(instSet);  
    divRegionSets.push_back(regionSet);
    bbAccessSets.push_back(bbSet);
  }
}

HierAddressSpace::HierAddressSpace(const HierAddressSpace &address) {
  numWDBI = address.numWDBI;
  numWD = address.numWD;   

  raceQueryNum = address.raceQueryNum;
  mcQueryNum = address.mcQueryNum;
  bcQueryNum = address.bcQueryNum;

  cpuMemory = address.cpuMemory;
  deviceMemory = address.deviceMemory;
  sharedMemories = address.sharedMemories;
  localMemories = address.localMemories;
  instAccessSets = address.instAccessSets;
  bbAccessSets = address.bbAccessSets;
  divRegionSets = address.divRegionSets;
  sameInstVecSets = address.sameInstVecSets;
  branchDivRegionSets = address.branchDivRegionSets;
  warpsBranchDivRegionSets = address.warpsBranchDivRegionSets;

  hasBC = address.hasBC;
  hasNoMC = address.hasNoMC;
  hasVM = address.hasVM;
  bcCondComb = address.bcCondComb;
  nonMCCondComb = address.nonMCCondComb;
  vmCondComb = address.vmCondComb;

  bcWDSet = address.bcWDSet;
  nomcWDSet = address.nomcWDSet;
  wdWDSet = address.wdWDSet;
}

AddressSpace& HierAddressSpace::getAddressSpace(GPUConfig::CTYPE ctype, unsigned b_t_index) {
  switch (ctype) {
  case GPUConfig::UNKNOWN :
    assert(false && "the ctype is unknown!");
  case GPUConfig::HOST :
    return cpuMemory;
  case GPUConfig::DEVICE :
  case GPUConfig::CONSTANT:
    return deviceMemory;
  case GPUConfig::SHARED :
    return sharedMemories[b_t_index];
  default:  // LOCAL
    return localMemories[b_t_index];
  }
}

void HierAddressSpace::bindObject(const MemoryObject *mo, ObjectState *os, unsigned b_t_index) {
  getAddressSpace(mo->ctype, b_t_index).bindObject(mo, os);
}

void HierAddressSpace::unbindObject(const MemoryObject *mo, unsigned b_t_index) {
  getAddressSpace(mo->ctype, b_t_index).unbindObject(mo);
}

const ObjectState* HierAddressSpace::findObject(const MemoryObject *mo, unsigned b_t_index) {
  return getAddressSpace(mo->ctype, b_t_index).findObject(mo);
}

ObjectState* HierAddressSpace::findNonConstantObject(const MemoryObject *mo, unsigned b_t_index) {
  return getAddressSpace(mo->ctype, b_t_index).findNonConstantObject(mo);
}

ObjectState* HierAddressSpace::getWriteable(const MemoryObject *mo,
					    const ObjectState *os,
					    unsigned b_t_index) {
  return getAddressSpace(mo->ctype, b_t_index).getWriteable(mo, os);
}

void HierAddressSpace::addWrite(const MemoryObject *mo, 
                                ref<Expr> &offset, ref<Expr> &val, 
                                Expr::Width width, 
                                unsigned bid, unsigned tid, 
                                llvm::Instruction *instr, unsigned seqNum, 
                                bool isAtomic, std::string fence, 
                                unsigned b_t_index, 
                                ref<Expr> accessExpr) {
  if (mo->ctype != GPUConfig::LOCAL) {
    if (!mo->is_builtin) {
      getAddressSpace(mo->ctype, b_t_index).writeSet.
        push_back(MemoryAccess(mo, offset, width, bid, tid, 
                               instr, seqNum, fence, 
                               isAtomic, true, 
                               accessExpr, val));
    }
  }
}

void HierAddressSpace::addRead(const MemoryObject* mo, 
                               ref<Expr> &offset, ref<Expr> &val, 
                               Expr::Width width, unsigned bid, unsigned tid, 
                               llvm::Instruction *instr, unsigned seqNum, 
                               bool isAtomic, std::string fence, 
                               unsigned b_t_index, ref<Expr> accessExpr) {
  if (mo->ctype != GPUConfig::LOCAL) {
    if (!mo->is_builtin) {
      getAddressSpace(mo->ctype, b_t_index).readSet.
        push_back(MemoryAccess(mo, offset, width, bid, tid, 
                               instr, seqNum, fence, 
                               isAtomic, false, 
                               accessExpr, val));
    }
  }
}

void HierAddressSpace::insertInst(bool is_GPU_mode, unsigned bid, unsigned tid, 
                                  llvm::Instruction *instr, bool isBr, 
                                  unsigned &seqNum) {
  if (!is_GPU_mode) return; 

  seqNum = instAccessSets[tid].size();
  instAccessSets[tid].push_back(InstAccess(bid, tid, instr, isBr));
}

//******************************************************************************************

bool HierAddressSpace::resolveOne(const ref<ConstantExpr> &addr, 
				  ObjectPair &result, 
				  GPUConfig::CTYPE ctype,
				  unsigned b_t_index) {
  return getAddressSpace(ctype, b_t_index).resolveOne(addr, result);
}

bool HierAddressSpace::resolveOne(ExecutionState &state,
				  TimingSolver *solver,
				  ref<Expr> address,
				  ObjectPair &result,
				  bool &success,
				  GPUConfig::CTYPE ctype,
				  unsigned b_t_index) {
  ExecutorUtil::copyOutConstraintUnderSymbolic(state);
  bool res = getAddressSpace(ctype, b_t_index).resolveOne(state, solver, address, result, success);
  ExecutorUtil::copyBackConstraintUnderSymbolic(state);
  return res;
}

bool HierAddressSpace::resolve(ExecutionState &state,
			       TimingSolver *solver, 
			       ref<Expr> p, 
			       ResolutionList &rl, 
			       unsigned maxResolutions,
			       double timeout,
			       GPUConfig::CTYPE ctype,
			       unsigned b_t_index) {
  ExecutorUtil::copyOutConstraintUnderSymbolic(state);
  bool res = getAddressSpace(ctype, b_t_index).resolve(state, solver, p, rl, maxResolutions, timeout);
  ExecutorUtil::copyBackConstraintUnderSymbolic(state);
  return res;
}

// These two are pretty big hack so we can sort of pass memory back
// and forth to externals. They work by abusing the concrete cache
// store inside of the object states, which allows them to
// transparently avoid screwing up symbolics (if the byte is symbolic
// then its concrete cache byte isn't being used) but is just a hack.

void HierAddressSpace::copyOutConcretes(unsigned tid, unsigned bid) {
  cpuMemory.copyOutConcretes();

  return;   // external function calls only on CPU?!

  deviceMemory.copyOutConcretes();
  sharedMemories[bid].copyOutConcretes();
  localMemories[tid].copyOutConcretes();
}

bool HierAddressSpace::copyInConcretes(unsigned tid, unsigned bid) {
  return cpuMemory.copyInConcretes();

  return 
    cpuMemory.copyInConcretes() &&
    deviceMemory.copyInConcretes() &&
    sharedMemories[bid].copyInConcretes() &&
    localMemories[tid].copyInConcretes();
}


//******************************************************************************************
// Other functions
//******************************************************************************************

void HierAddressSpace::clearAccessSet(char mask) {
  if (mask & 0x8)
    cpuMemory.clearAccessSet();
  if (mask & 0x4)
    deviceMemory.clearAccessSet();
  if (mask & 0x2) {
    for (unsigned k = 0; k < sharedMemories.size(); k++)
      sharedMemories[k].clearAccessSet();  
  }
  if (mask & 0x1) {
    for (unsigned k = 0; k < localMemories.size(); k++)
      localMemories[k].clearAccessSet();  
  }
}

void HierAddressSpace::clearInstAccessSet(bool clearAll) {
  for (unsigned i = 0; i<instAccessSets.size(); i++) {
    if (clearAll)
      instAccessSets[i].clear();
    bbAccessSets[i].clear();
    divRegionSets[i].clear();
  }
  sameInstVecSets.clear();
  // clear warps div region sets ... 
  for (unsigned i = 0; i < warpsBranchDivRegionSets.size(); i++)
    warpsBranchDivRegionSets[i].clear();
}

void HierAddressSpace::clearGlobalAccessSet() {
  deviceMemory.clearGlobalMemoryAccessSets();
  cpuMemory.clearGlobalMemoryAccessSets();
}

void HierAddressSpace::clearWarpDefectSet() {
  bcWDSet.clear();  
  nomcWDSet.clear();  
  wdWDSet.clear();
}

void AddressSpace::clearGlobalMemoryAccessSets() {
  for (unsigned i = 0; i < MemAccessSets.size(); i++)
    MemAccessSets[i].clear();
  for (unsigned i = 0; i < MemAccessSetsPureCS.size(); i++)
    MemAccessSetsPureCS[i].clear();
}

void AddressSpace::clearSymGlobalMemoryAccessSets() {
  for (unsigned i = 0; i < symGlobalReadSets.size(); i++)
    symGlobalReadSets[i].clear();
  for (unsigned i = 0; i < symGlobalWriteSets.size(); i++)
    symGlobalWriteSets[i].clear();
}

void AddressSpace::dump(bool rwset_only) {
  if (!rwset_only) {
    for (MemoryMap::iterator it = objects.begin(), ie = objects.end(); 
	 it != ie; ++it) {
      const MemoryObject *mo = it->first;
      const ObjectState *os = it->second;
      if (mo->name != "unnamed") 
	GKLEE_INFO << mo->name << " (" << mo->address << ") := ";
      else
	GKLEE_INFO << mo->address << " := ";
      for (unsigned k = 0; k < os->size/4; k++) {
	os->read(k * 4, 32)->print(std::cout);
	std::cout << " ";
      }
      std::cout << std::endl;
      // os->read(0, os->size * 8)->dump();
    }
  }

  if (!readSet.empty()) {
    GKLEE_INFO << "Read Set: \n";
    for (MemoryAccessVec::const_iterator ii = readSet.begin(); ii != readSet.end(); ii++)
      ii->dump();
    std::cout << std::endl;
  }
  if (!writeSet.empty()) {
    GKLEE_INFO << "Write Set: \n";
    for (MemoryAccessVec::const_iterator ii = writeSet.begin(); ii != writeSet.end(); ii++)
      ii->dump();
    std::cout << std::endl;
  }
}


void HierAddressSpace::dump(char mask) {
  if (mask & 0x8) {
    GKLEE_INFO << "------------------CPU Memory------------------\n";
    cpuMemory.dump();
  }
  if (mask & 0x4) {
    GKLEE_INFO << "------------------Device Memory------------------ \n";
    deviceMemory.dump();
  }
  if (mask & 0x2) {
    GKLEE_INFO << "------------------Shared Memories------------------ \n";
    for (unsigned k = 0; k < sharedMemories.size(); k++) {
      GKLEE_INFO << "---------------Shared Memory " << k << ": \n";
      sharedMemories[k].dump();  
    }
  }
  if (mask & 0x1) {
    GKLEE_INFO << "------------------Local Memories------------------ \n";
    for (unsigned k = 0; k < localMemories.size(); k++) {
      GKLEE_INFO << "---------------Thread " << k << ": \n";
      localMemories[k].dump();
    }
  }
}

void HierAddressSpace::dumpInstAccessSet() {
  GKLEE_INFO << "numWDBI: " << numWDBI << " ,numWD: " << numWD << std::endl;
  for (unsigned i = 0; i < instAccessSets.size(); i++) {
    GKLEE_INFO << "\n--------------Thread " << i << "--------------- \n";
    for (InstAccessSet::const_iterator ii = instAccessSets[i].begin(); 
         ii != instAccessSets[i].end(); ii++) {
      ii->dump();
    }
  }
} 

/***/
