//===-- AddressSpace.cpp --------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AddressSpace.h"
#include "CoreStats.h"
#include "Memory.h"
#include "TimingSolver.h"

#include "klee/Expr.h"
#include "klee/TimerStatIncrementer.h"
#include "llvm/Instructions.h"
#include "llvm/BasicBlock.h"
#include "klee/ExecutionState.h"

#include <iostream>

using namespace klee;

///

void AddressSpace::bindObject(const MemoryObject *mo, ObjectState *os) {
  assert(os->copyOnWriteOwner==0 && "object already has owner");
  assert(mo->ctype == ctype && "unmatched ctypes");
  os->copyOnWriteOwner = cowKey;
  objects = objects.replace(std::make_pair(mo, os));
}

void AddressSpace::unbindObject(const MemoryObject *mo) {
  assert(mo->ctype == ctype && "unmatched ctypes");
  objects = objects.remove(mo);
}

const ObjectState *AddressSpace::findObject(const MemoryObject *mo) const {
  assert(mo->ctype == ctype && "unmatched ctypes");
  const MemoryMap::value_type *res = objects.lookup(mo);
  
  return res ? res->second : 0;
}
  
ObjectState *AddressSpace::findNonConstantObject(const MemoryObject *mo) const {
  assert(mo->ctype == ctype && "unmatched ctypes");
  const MemoryMap::value_type *res = objects.lookup(mo);
  
  return res ? res->second : 0;
}

ObjectState *AddressSpace::getWriteable(const MemoryObject *mo,
                                        const ObjectState *os) {
  assert(!os->readOnly);

  if (cowKey==os->copyOnWriteOwner) {
    return const_cast<ObjectState*>(os);
  } else {
    ObjectState *n = new ObjectState(*os);
    n->copyOnWriteOwner = cowKey;
    objects = objects.replace(std::make_pair(mo, n));
    return n;    
  }
}

/// 

bool AddressSpace::resolveOne(const klee::ref<ConstantExpr> &addr, 
                              ObjectPair &result) {

  uint64_t address = addr->getZExtValue();
  MemoryObject hack(address);

  /*std::cout << "address = " << address << std::endl;
  std::cout << "Memory type: " << CUDAUtil::getCTypeStr(ctype) << std::endl; 
  for (MemoryMap::iterator oi = objects.begin(); oi != objects.end(); ++oi) {
    std::cout << "addr: " << oi->first->address << ", size: " << oi->first->size << std::endl;
  }
  std::cout << std::endl;*/

  if (const MemoryMap::value_type *res = objects.lookup_previous(&hack)) {
    const MemoryObject *mo = res->first;
    if ((mo->size==0 && address==mo->address) ||
        (address - mo->address < mo->size)) {
      result = *res;
      return true;
    }
  }

  return false;
}

bool AddressSpace::resolveOne(ExecutionState &state,
                              TimingSolver *solver,
                              klee::ref<Expr> address,
                              ObjectPair &result,
                              bool &success) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(address)) {
    success = resolveOne(CE, result);
    return true;
  } else {
    TimerStatIncrementer timer(stats::resolveTime);

    // try cheap search, will succeed for any inbounds pointer
    //state.dumpStateConstraint(); 
    //std::cout << "address expr: " << std::endl;
    //address->dump();

    klee::ref<ConstantExpr> cex;
    if (!solver->getValue(state, address, cex))
      return false;
    uint64_t example = cex->getZExtValue();
    MemoryObject hack(example);
    const MemoryMap::value_type *res = objects.lookup_previous(&hack);
    
    if (res) {
      const MemoryObject *mo = res->first;
      if (example - mo->address < mo->size) {
        result = *res;
        success = true;
        return true;
      }
    }

    // didn't work, now we have to search
       
    MemoryMap::iterator oi = objects.upper_bound(&hack);
    MemoryMap::iterator begin = objects.begin();
    MemoryMap::iterator end = objects.end();
      
    MemoryMap::iterator start = oi;
    while (oi!=begin) {
      --oi;
      const MemoryObject *mo = oi->first;
        
      bool mayBeTrue;
      if (!solver->mayBeTrue(state, 
                             mo->getBoundsCheckPointer(address), mayBeTrue))
        return false;
      if (mayBeTrue) {
        result = *oi;
        success = true;
        return true;
      } else {
        bool mustBeTrue;
        if (!solver->mustBeTrue(state, 
                                UgeExpr::create(address, mo->getBaseExpr()),
                                mustBeTrue))
          return false;
        if (mustBeTrue)
          break;
      }
    }

    // search forwards
    for (oi=start; oi!=end; ++oi) {
      const MemoryObject *mo = oi->first;

      bool mustBeTrue;
      if (!solver->mustBeTrue(state, 
                              UltExpr::create(address, mo->getBaseExpr()),
                              mustBeTrue))
        return false;
      if (mustBeTrue) {
        break;
      } else {
        bool mayBeTrue;

        if (!solver->mayBeTrue(state, 
                               mo->getBoundsCheckPointer(address),
                               mayBeTrue))
          return false;
        if (mayBeTrue) {
          result = *oi;
          success = true;
          return true;
        }
      }
    }

    success = false;
    return true;
  }
}

bool AddressSpace::resolve(ExecutionState &state,
                           TimingSolver *solver, 
                           klee::ref<Expr> p, 
                           ResolutionList &rl, 
                           unsigned maxResolutions,
                           double timeout) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(p)) {
    ObjectPair res;
    if (resolveOne(CE, res))
      rl.push_back(res);
    return false;
  } else {
    TimerStatIncrementer timer(stats::resolveTime);
    uint64_t timeout_us = (uint64_t) (timeout*1000000.);

    // XXX in general this isn't exactly what we want... for
    // a multiple resolution case (or for example, a \in {b,c,0})
    // we want to find the first object, find a cex assuming
    // not the first, find a cex assuming not the second...
    // etc.
    
    // XXX how do we smartly amortize the cost of checking to
    // see if we need to keep searching up/down, in bad cases?
    // maybe we don't care?
    
    // XXX we really just need a smart place to start (although
    // if its a known solution then the code below is guaranteed
    // to hit the fast path with exactly 2 queries). we could also
    // just get this by inspection of the expr.
    
    klee::ref<ConstantExpr> cex;
    if (!solver->getValue(state, p, cex))
      return true;
    uint64_t example = cex->getZExtValue();
    MemoryObject hack(example);
    
    MemoryMap::iterator oi = objects.upper_bound(&hack);
    MemoryMap::iterator begin = objects.begin();
    MemoryMap::iterator end = objects.end();
      
    MemoryMap::iterator start = oi;
      
    // XXX in the common case we can save one query if we ask
    // mustBeTrue before mayBeTrue for the first result. easy
    // to add I just want to have a nice symbolic test case first.
      
    // search backwards, start with one minus because this
    // is the object that p *should* be within, which means we
    // get write off the end with 4 queries (XXX can be better,
    // no?)
    while (oi!=begin) {
      --oi;
      const MemoryObject *mo = oi->first;
      if (timeout_us && timeout_us < timer.check())
        return true;

      // XXX I think there is some query wasteage here?
      klee::ref<Expr> inBounds = mo->getBoundsCheckPointer(p);
      bool mayBeTrue;
      if (!solver->mayBeTrue(state, inBounds, mayBeTrue))
        return true;
      if (mayBeTrue) {
        rl.push_back(*oi);
        
        // fast path check
        unsigned size = rl.size();
        if (size==1) {
          bool mustBeTrue;
          if (!solver->mustBeTrue(state, inBounds, mustBeTrue))
            return true;
          if (mustBeTrue)
            return false;
        } else if (size==maxResolutions) {
          return true;
        }
      }
        
      bool mustBeTrue;
      if (!solver->mustBeTrue(state, 
                              UgeExpr::create(p, mo->getBaseExpr()),
                              mustBeTrue))
        return true;
      if (mustBeTrue)
        break;
    }
    // search forwards
    for (oi=start; oi!=end; ++oi) {
      const MemoryObject *mo = oi->first;
      if (timeout_us && timeout_us < timer.check())
        return true;

      bool mustBeTrue;
      if (!solver->mustBeTrue(state, 
                              UltExpr::create(p, mo->getBaseExpr()),
                              mustBeTrue))
        return true;
      if (mustBeTrue)
        break;
      
      // XXX I think there is some query wasteage here?
      klee::ref<Expr> inBounds = mo->getBoundsCheckPointer(p);
      bool mayBeTrue;
      if (!solver->mayBeTrue(state, inBounds, mayBeTrue))
        return true;
      if (mayBeTrue) {
        rl.push_back(*oi);
        
        // fast path check
        unsigned size = rl.size();
        if (size==1) {
          bool mustBeTrue;
          if (!solver->mustBeTrue(state, inBounds, mustBeTrue))
            return true;
          if (mustBeTrue)
            return false;
        } else if (size==maxResolutions) {
          return true;
        }
      }
    }
  }

  return false;
}

// These two are pretty big hack so we can sort of pass memory back
// and forth to externals. They work by abusing the concrete cache
// store inside of the object states, which allows them to
// transparently avoid screwing up symbolics (if the byte is symbolic
// then its concrete cache byte isn't being used) but is just a hack.

void AddressSpace::copyOutConcretes() {
  for (MemoryMap::iterator it = objects.begin(), ie = objects.end(); 
       it != ie; ++it) {
    const MemoryObject *mo = it->first;

    if (!mo->isUserSpecified) {
      ObjectState *os = it->second;
      uint8_t *address = (uint8_t*) (unsigned long) mo->address;

      if (!os->readOnly)
        memcpy(address, os->concreteStore, mo->size);
    }
  }
}

bool AddressSpace::copyInConcretes() {
  for (MemoryMap::iterator it = objects.begin(), ie = objects.end(); 
       it != ie; ++it) {
    const MemoryObject *mo = it->first;

    if (!mo->isUserSpecified) {
      const ObjectState *os = it->second;
      uint8_t *address = (uint8_t*) (unsigned long) mo->address;

      if (memcmp(address, os->concreteStore, mo->size)!=0) {
        if (os->readOnly) {
          return false;
        } else {
          ObjectState *wos = getWriteable(mo, os);
          memcpy(wos->concreteStore, address, mo->size);
        }
      }
    }
  }

  return true;
}

static bool isTwoBBIdentical(std::string funcName1, std::string funcName2, 
                             llvm::BasicBlock *bb1, llvm::BasicBlock *bb2) {
  return funcName1.compare(funcName2) == 0 && bb1 == bb2;
}

static bool isInstBrOrSwitch(llvm::Instruction *inst) {
  return inst->getOpcode() == llvm::Instruction::Br 
          || inst->getOpcode() == llvm::Instruction::Switch; 
}

// To find the basic block graph ...
void HierAddressSpace::forwardingExploreInstSet(unsigned start, unsigned end) {
  for (unsigned i = start; i <= end; i++) {
    if (instAccessSets[i].size() == 0) continue;
    // go through the instruction set one by one (forward)...
    unsigned idx = 0;
    InstAccessSet::iterator ii = instAccessSets[i].begin();
    llvm::BasicBlock *curBB = ii->inst->getParent();
    std::string curFuncName = curBB->getParent()->getName().str();
    bbAccessSets[i].push_back(BasicBlockAccess(curFuncName, curBB, idx, idx)); 
    bool newBlock = isInstBrOrSwitch(ii->inst);
    unsigned size = 0;
    ii++;
    idx++;
    for (; ii != instAccessSets[i].end(); ii++, idx++) {
      llvm::BasicBlock *bb = ii->inst->getParent(); 
      std::string funcName = bb->getParent()->getName().str();
      size = bbAccessSets[i].size();

      if (newBlock) {
        curBB = bb;
        curFuncName = funcName;
        bbAccessSets[i].push_back(BasicBlockAccess(curFuncName, curBB, idx, idx)); 
      } else {   
        if (isTwoBBIdentical(curFuncName, funcName, curBB, bb)) {
          bbAccessSets[i][size-1].endIdx = idx; 
        } else { 
          curBB = bb;
          curFuncName = funcName;
          bbAccessSets[i].push_back(BasicBlockAccess(curFuncName, curBB, idx, idx)); 
        }
      }
      // determine if the current instruction is branch 
      // or switch instruction
      newBlock = isInstBrOrSwitch(ii->inst);
    }
  }
}

static bool isTwoInstIdentical(llvm::Instruction *inst1, llvm::Instruction *inst2) {
  std::string func1Name = (inst1->getParent()->getParent()->getName()).str();
  std::string func2Name = (inst2->getParent()->getParent()->getName()).str();
  llvm::BasicBlock *bb1 = inst1->getParent();
  llvm::BasicBlock *bb2 = inst2->getParent();

  return func1Name.compare(func2Name) == 0 
           && bb1 == bb2
             && inst1->isIdenticalTo(inst2); 
}

static unsigned getBasicBlockAccessIdx(BBAccessSet &bbSet, unsigned instIdx) {
  unsigned bbIdx = 0;
  for (BBAccessSet::iterator ii = bbSet.begin(); ii != bbSet.end();
       ii++, bbIdx++) {
    if (ii->startIdx <= instIdx && instIdx <= ii->endIdx) {
      break;
    }
  } 
  return bbIdx;
} 

static bool findMergePoint(BBAccessSet &bbAccessSet, unsigned start, unsigned end, 
                           BasicBlockAccess &bbAccess, unsigned &pos) {
  bool findMP = false;
  for (unsigned i = start; i <= end; i++) {
    if (isTwoBBIdentical(bbAccessSet[i].funcName, bbAccess.funcName, 
                         bbAccessSet[i].bb, bbAccess.bb)) {
      pos = i;
      findMP = true;
      break;
    }
  }
  return findMP; 
}

static void dumpBBAccessSet(BBAccessSet &bbAccessSet) {
  for (unsigned i = 0; i < bbAccessSet.size(); i++) {
    std::cout << "<" << bbAccessSet[i].bb->getName().str() << ", ("
              << bbAccessSet[i].startIdx << ", " << bbAccessSet[i].endIdx
              << ")>" << "->";
  }
  std::cout << "end" << std::endl; 
}

static void compareTwoInstSets(InstAccessSet &instSet1, InstAccessSet &instSet2, 
                               std::vector<BBAccessSet> &bbAccessSets, 
                               std::vector<RefDivRegionSetVec> &divRegionSets) {
  // Try the set1 first. 
  unsigned tid1 = instSet1.begin()->tid;
  unsigned tid2 = instSet2.begin()->tid;

  if (GPUConfig::verbose > 0) {
    std::cout << "tid1: " << tid1 << ", tid2: " << tid2 << std::endl;
    std::cout << "bbAccessSets tid1: " << std::endl;
    dumpBBAccessSet(bbAccessSets[tid1]);
    std::cout << "bbAccessSets tid2: " << std::endl;
    dumpBBAccessSet(bbAccessSets[tid2]);
  }

  DivRegionSet set1;
  RefDivRegionSet regionSet1(tid2, set1);

  DivRegionSet set2;
  RefDivRegionSet regionSet2(tid1, set2);

  for (unsigned i = 0, j = 0; i < instSet1.size() && j < instSet2.size();) {
    if (isTwoInstIdentical(instSet1[i].inst, instSet2[j].inst)) {
      i++; j++;
    } else {
      unsigned bbIdx1 = getBasicBlockAccessIdx(bbAccessSets[tid1], i);  
      unsigned bbIdx2 = getBasicBlockAccessIdx(bbAccessSets[tid2], j);

      // From first thread's point of view ..
      std::vector <unsigned> bbVec1;
      std::vector <unsigned> bbVec2;
      
      unsigned len1 = bbAccessSets[tid1].size() - bbIdx1 - 1;
      unsigned len2 = bbAccessSets[tid2].size() - bbIdx2 - 1;
      unsigned len = (len1 > len2)? len1 : len2;

      unsigned tmpIdx1 = bbIdx1; 
      unsigned tmpIdx2 = bbIdx2;

      unsigned pos1 = 0;
      unsigned pos2 = 0;
     
      bool findFrom1 = false;
      bool findFrom2 = false;

      for (unsigned k = 0; k <= len; k++) {
        // For 1, find from 2..
        findFrom2 = findMergePoint(bbAccessSets[tid2], bbIdx2, tmpIdx2, bbAccessSets[tid1][tmpIdx1], pos2);
     
        if (findFrom2) break; 

        // For 2, find from 1..    
        findFrom1 = findMergePoint(bbAccessSets[tid1], bbIdx1, tmpIdx1, bbAccessSets[tid2][tmpIdx2], pos1);

        if (findFrom1) break;

        if (tmpIdx1 != bbAccessSets[tid1].size()-1) tmpIdx1++;
        if (tmpIdx2 != bbAccessSets[tid2].size()-1) tmpIdx2++;
      }

      unsigned minIdx1 = 0;
      unsigned minIdx2 = 0;

      if (!findFrom1 && !findFrom2) {
        minIdx1 = bbAccessSets[tid1].size()-1;
        minIdx2 = bbAccessSets[tid2].size()-1;

        regionSet1.regionSet.push_back(DivRegion(bbIdx1, minIdx1, bbAccessSets[tid1][bbIdx1].startIdx, 
                                       bbAccessSets[tid1][minIdx1].endIdx, false));
        regionSet2.regionSet.push_back(DivRegion(bbIdx2, minIdx2, bbAccessSets[tid2][bbIdx2].startIdx, 
                                       bbAccessSets[tid2][minIdx2].endIdx, false));
      } else {
        if (findFrom1) {
          minIdx1 = pos1;
          minIdx2 = tmpIdx2;
        } else {
          minIdx1 = tmpIdx1;
          minIdx2 = pos2;
        }
        if (minIdx1 == bbIdx1)
          regionSet1.regionSet.push_back(DivRegion(minIdx1-1, minIdx1, bbAccessSets[tid1][minIdx1-1].endIdx, 
                                         bbAccessSets[tid1][minIdx1].startIdx, true));
        else 
          regionSet1.regionSet.push_back(DivRegion(bbIdx1, minIdx1-1, bbAccessSets[tid1][bbIdx1].startIdx, 
                                         bbAccessSets[tid1][minIdx1-1].endIdx, false));

        if (minIdx2 == bbIdx2)
          regionSet2.regionSet.push_back(DivRegion(minIdx2-1, minIdx2, bbAccessSets[tid2][minIdx2-1].endIdx, 
                                         bbAccessSets[tid2][minIdx2].startIdx, true));
        else 
          regionSet2.regionSet.push_back(DivRegion(bbIdx2, minIdx2-1, bbAccessSets[tid2][bbIdx2].startIdx, 
                                         bbAccessSets[tid2][minIdx2-1].endIdx, false));
      }

      i = bbAccessSets[tid1][minIdx1].endIdx+1;
      j = bbAccessSets[tid2][minIdx2].endIdx+1;
    }
  }

  divRegionSets[tid1].push_back(regionSet1); 
  divRegionSets[tid2].push_back(regionSet2); 
}

static void dealwithRepresentativeTidSet(std::vector<InstAccessSet> &instAccessSets, 
                                         std::vector<RefDivRegionSetVec> &divRegionSets, 
                                         std::vector<BBAccessSet> &bbAccessSets, 
                                         SameInstVec &sameInstVec, std::vector<unsigned> respSet) {
  if (respSet.size() == 1)
    return;

  for (std::vector<unsigned>::iterator ii = respSet.begin(); 
       ii != respSet.end(); ii++) {
    std::vector<unsigned>::iterator jj = ii;
    jj++;
    for (; jj != respSet.end(); jj++) {
      compareTwoInstSets(instAccessSets[*ii], instAccessSets[*jj], 
                         bbAccessSets, divRegionSets); 
    }
  }

  for (unsigned i = 0; i<sameInstVec.size(); i++) {
    std::vector<unsigned> set = sameInstVec[i];
    unsigned sTid = respSet[i];
    for (unsigned j = 1; j < set.size(); j++) {
      for (RefDivRegionSetVec::iterator ii = divRegionSets[sTid].begin();
           ii != divRegionSets[sTid].end(); ii++) { 
        DivRegionSet region = ii->regionSet;
        divRegionSets[set[j]].push_back(RefDivRegionSet(ii->otherTid, region));
      }
    }
  }
}

// construct divergence region sets.. 
void HierAddressSpace::constructDivergRegionSets(std::vector<CorrespondTid>& cTidSets, 
                                                 unsigned start, unsigned end) {
  unsigned warpStart = cTidSets[start].warpNum;
  unsigned warpEnd = cTidSets[end].warpNum;

  for (unsigned i = warpStart; i <= warpEnd; i++) {
    std::vector<unsigned> repTids;
    for (SameInstVec::iterator ii = sameInstVecSets[i].begin();
         ii != sameInstVecSets[i].end(); ii++) {
      unsigned tid = *(ii->begin());
      repTids.push_back(tid);
    }
    // Deal with representative tid sets...
    dealwithRepresentativeTidSet(instAccessSets, divRegionSets, 
                                 bbAccessSets, sameInstVecSets[i], repTids);
  }
}

/***/
bool MemoryObjectLT::operator()(const MemoryObject *a, const MemoryObject *b) const {
  return a->address < b->address;
}

