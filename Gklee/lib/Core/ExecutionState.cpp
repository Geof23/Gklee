//===-- ExecutionState.cpp ------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "klee/ExecutionState.h"

#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"

#include "klee/Expr.h"

#include "klee/logging.h"

#include "Memory.h"
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
#include "llvm/IR/Function.h"
#else
#include "llvm/Function.h"
#endif
#include "llvm/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <map>
#include <set>
#include <stdarg.h>

#include "CUDA.h"

using namespace llvm;
using namespace klee;

namespace runtime { 
  cl::opt<bool>
  DebugLogStateMerge("debug-log-state-merge");
}

using namespace runtime;

/***/

StackFrame::StackFrame(KInstIterator _caller, KFunction *_kf)
  : caller(_caller), kf(_kf), callPathNode(0), 
    minDistToUncoveredOnReturn(0), varargs(0) {
  Gklee::Logging::enterFunc< std::string >( "new stack frame" , __PRETTY_FUNCTION__ );  
  locals = new Cell[kf->numRegisters];
  Gklee::Logging::exitFunc();
}

StackFrame::StackFrame(const StackFrame &s) 
  : caller(s.caller),
    kf(s.kf),
    callPathNode(s.callPathNode),
    allocas(s.allocas),
    minDistToUncoveredOnReturn(s.minDistToUncoveredOnReturn),
    varargs(s.varargs) {
  Gklee::Logging::enterFunc< std::string >( "new stack frame" , __PRETTY_FUNCTION__ );
  // std::cout << "Copying stackframe \n";
  locals = new Cell[s.kf->numRegisters];
  for (unsigned i=0; i<s.kf->numRegisters; i++)
    locals[i] = s.locals[i];
  Gklee::Logging::exitFunc();
}

StackFrame& StackFrame::operator=(const StackFrame& s) {
  // std::cout << "Assigning stackframe \n";
  Gklee::Logging::enterFunc< std::string >( "new stack frame" , __PRETTY_FUNCTION__ );
  if (this != &s) {
    caller = s.caller;
    kf = s.kf;
    callPathNode = s.callPathNode;
    allocas = s.allocas;
    minDistToUncoveredOnReturn = s.minDistToUncoveredOnReturn;
    varargs = s.varargs;
    locals = new Cell[s.kf->numRegisters];
    for (unsigned i=0; i<s.kf->numRegisters; i++)
      locals[i] = s.locals[i];
  }
  Gklee::Logging::exitFunc();
  return *this;
}

StackFrame::~StackFrame() { 
  Gklee::Logging::enterFunc< std::string >( "" , __PRETTY_FUNCTION__ );
  delete[] locals; 
  Gklee::Logging::exitFunc();
}

/***/

ExecutionState::ExecutionState(KFunction *kf) 
  : deviceSet(0),
    fakeState(false),
    fence(""),
    underConstrained(false),
    depth(0),
    brMeta(NA, NULL),
    tinfo(kf->instructions),
    queryCost(0.),
    weight(1),
    forkStateBINum(0),
    kernelNum(0),
    BINum(0),
    instsSinceCovNew(0),
    coveredNew(false),
    forkDisabled(false),
    ptreeNode(0), 
    maxKernelSharedSize(0)
{
  Gklee::Logging::enterFunc< std::string >( "create ExecutionState" , __PRETTY_FUNCTION__ );
  pushAllFrames(0, kf);
  Gklee::Logging::exitFunc();
}

ExecutionState::ExecutionState(const std::vector<klee::ref<Expr> > &assumptions) 
  : deviceSet(0),
    fakeState(true),
    fence(""),
    underConstrained(false),
    brMeta(NA, NULL),
    constraints(assumptions),
    queryCost(0.),
    ptreeNode(0), 
    maxKernelSharedSize(0) {
  Gklee::Logging::enterFunc( assumptions[0] , __PRETTY_FUNCTION__ );
  Gklee::Logging::exitFunc();
}

ExecutionState::~ExecutionState() {
  Gklee::Logging::enterFunc< std::string >( "" , __PRETTY_FUNCTION__ );  
  for (unsigned int i=0; i<symbolics.size(); i++)
  {
    const MemoryObject *mo = symbolics[i].first;
    assert(mo->refCount > 0);
    mo->refCount--;
    if (mo->refCount == 0)
      delete mo;
  }
  popAllFrames();
  Gklee::Logging::exitFunc();
}

ExecutionState::ExecutionState(const ExecutionState& state)
  : fnAliases(state.fnAliases),
    deviceSet(state.deviceSet),
    fakeState(state.fakeState),
    underConstrained(state.underConstrained),
    depth(state.depth),
    brMeta(state.brMeta),
    cTidSets(state.cTidSets),
    tinfo(state.tinfo),
    stacks(state.stacks),
    constraints(state.constraints),
    paraConstraints(state.paraConstraints),
    paraTreeSets(state.paraTreeSets),
    symInputVec(state.symInputVec),
    concreteTimeVec(state.concreteTimeVec),
    symTimeVec(state.symTimeVec),
    queryCost(state.queryCost),
    weight(state.weight),
    addressSpace(state.addressSpace),
    forkStateBINum(state.forkStateBINum),
    kernelNum(state.kernelNum),
    BINum(state.BINum),
    pathOS(state.pathOS),
    symPathOS(state.symPathOS),
    instsSinceCovNew(state.instsSinceCovNew),
    coveredNew(state.coveredNew),
    forkDisabled(state.forkDisabled),
    coveredLines(state.coveredLines),
    ptreeNode(state.ptreeNode),
    maxKernelSharedSize(state.maxKernelSharedSize),
    symbolics(state.symbolics),
    arrayNames(state.arrayNames),
    shadowObjects(state.shadowObjects),
    incomingBBIndex(state.incomingBBIndex)
{
  Gklee::Logging::enterFunc< std::string >( "" , __PRETTY_FUNCTION__ );  
  for (unsigned int i=0; i<symbolics.size(); i++)
    symbolics[i].first->refCount++;
  Gklee::Logging::exitFunc();
}

ExecutionState *ExecutionState::branch() {
  Gklee::Logging::enterFunc< std::string >( "" , __PRETTY_FUNCTION__ );  
  depth++;

  ExecutionState *falseState = new ExecutionState(*this);
  falseState->coveredNew = false;
  falseState->coveredLines.clear();

  weight *= .5;
  falseState->weight -= weight;
  Gklee::Logging::exitFunc();
  return falseState;
}

static void constructMemoryAccessSets(HierAddressSpace &addressSpace, bool PureCS) {
  Gklee::Logging::enterFunc( addressSpace.bcCondComb , __PRETTY_FUNCTION__ );  

  if (!UseSymbolicConfig) {
    MemoryAccessSetVec setVec;          
    addressSpace.deviceMemory.MemAccessSets.push_back(setVec); 
    addressSpace.cpuMemory.MemAccessSets.push_back(setVec);                  

    if (PureCS) {
      MemoryAccessSetVecPureCS setVecPureCS;        
      addressSpace.deviceMemory.MemAccessSetsPureCS.push_back(setVecPureCS); 
      addressSpace.cpuMemory.MemAccessSetsPureCS.push_back(setVecPureCS);
    }
  }
  Gklee::Logging::exitFunc();
}

void ExecutionState::setCorrespondTidSets() {
  unsigned curBid = 0;
  unsigned rTid = 0;
  unsigned warpNum = 0;
  unsigned tmpWarpNum = 0;
  Gklee::Logging::enterFunc< std::string >( "construct cTidSets, tinfo initThreadInfo" , __PRETTY_FUNCTION__ );
  constructMemoryAccessSets(addressSpace, true); //not used for symbolic-config
  unsigned num_threads = tinfo.get_num_threads();

  for (unsigned i = 0; i < num_threads; i++, rTid++) {
    unsigned bid = i/GPUConfig::block_size;
    if (bid != curBid) {
      curBid = bid;
      rTid = 0; 
      warpNum++;
      tmpWarpNum = 0;
      constructMemoryAccessSets(addressSpace, true);
    } else {
      if (rTid/GPUConfig::warpsize != tmpWarpNum) {
        tmpWarpNum = rTid / GPUConfig::warpsize;
        warpNum++;
        constructMemoryAccessSets(addressSpace, false);
      }
    }
    klee::ref<Expr> expr = ConstantExpr::create(1, Expr::Bool);
    cTidSets.push_back(CorrespondTid(curBid, rTid, warpNum, 
                                     false, false, false, expr));
    Gklee::Logging::outItem( std::string("bid:tid:warp;") +
			     std::to_string( curBid ) + ":" +
			     std::to_string( rTid ) + ":" +
			     std::to_string( warpNum ),
			     std::string( "new cTidSet" ) );
  }
  tinfo.setInitThreadInfo(cTidSets);
  Gklee::Logging::outItem< std::string >( "created cTidSets" , "initialize tinfo" );
  Gklee::Logging::exitFunc();
}

void ExecutionState::clearCorrespondTidSets() {
  cTidSets.clear();
}

void ExecutionState::incKernelNum() {
  kernelNum++;
}

unsigned ExecutionState::getKernelNum() {
  return kernelNum;
}

// create all the stacks; should be called only when the state is created
// a thread should never call this function
void ExecutionState::pushAllFrames(KInstIterator caller, KFunction *kf) {
  Gklee::Logging::enterFunc< std::string >( std::string( "threads: " ) +
					    std::to_string( tinfo.get_num_threads() ), 
					    __PRETTY_FUNCTION__ );  
  for (unsigned i = 0; i < tinfo.get_num_threads(); i++) {
    stack_ty stk;
    stk.push_back(StackFrame(caller, kf));
    stacks.push_back(stk);
    incomingBBIndex.push_back(0);
  }
  Gklee::Logging::exitFunc();
}

// destroy all the stacks; should be called only when the execution state is 
// released a thread should never call this function
void ExecutionState::popAllFrames() {
  Gklee::Logging::enterFunc< std::string >( std::string( "stacks: " ) +
					    std::to_string( stacks.size()), 
					    __PRETTY_FUNCTION__ );  
  unsigned i = 0;
  for (stacks_ty::iterator ii = stacks.begin(); ii != stacks.end(); ii++) {
    if (ii->size() > 0) {
      if (i == 0) { // Indicate the thread with tid 0
        StackFrame &sf = ii->back();
        for (std::vector<const MemoryObject*>::iterator it = sf.allocas.begin(), 
	     ie = sf.allocas.end(); it != ie; ++it)
	  addressSpace.unbindObject(*it);
        ii->pop_back();
      } else { // Indicate other threads except thread 0
        while (ii->size() != 0) {
          StackFrame &sf = ii->back();
          sf.allocas.clear();
          ii->pop_back();
        }
      }
    }
    i++;
  }
  Gklee::Logging::exitFunc();
}

void ExecutionState::pushFrame(KInstIterator caller, KFunction *kf) {
  Gklee::Logging::enterFunc( *kf, __PRETTY_FUNCTION__ );
  getCurStack().push_back(StackFrame(caller,kf));
  Gklee::Logging::exitFunc();
}

void ExecutionState::popFrame() {
  Gklee::Logging::enterFunc< std::string >( "" , __PRETTY_FUNCTION__ );
  StackFrame &sf = getCurStack().back();
  Gklee::Logging::outItem( *( sf.kf ), "popping (discarding) frame" );
  for (std::vector<const MemoryObject*>::iterator it = sf.allocas.begin(), 
	 ie = sf.allocas.end(); it != ie; ++it)
    addressSpace.unbindObject(*it);
  getCurStack().pop_back();
  Gklee::Logging::exitFunc();
}

void ExecutionState::addSymbolic(const MemoryObject *mo, const Array *array) {
  Gklee::Logging::enterFunc( std::string( "mo name: " ) +
			     mo->getName() + " array name: " +
			     array->name, __PRETTY_FUNCTION__ );
  mo->refCount++;
  symbolics.push_back(std::make_pair(mo, array));
  Gklee::Logging::exitFunc();
}

///

std::string ExecutionState::getFnAlias(std::string fn) {
  Gklee::Logging::enterFunc( fn , __PRETTY_FUNCTION__ );
  std::map < std::string, std::string >::iterator it = fnAliases.find(fn);
  if (it != fnAliases.end()){
    Gklee::Logging::exitFunc();
    return it->second;
  }else{
    Gklee::Logging::exitFunc();
    return "";
  }
}

void ExecutionState::addFnAlias(std::string old_fn, std::string new_fn) {
  Gklee::Logging::enterFunc( new_fn , __PRETTY_FUNCTION__ );
  fnAliases[old_fn] = new_fn;
  Gklee::Logging::exitFunc();
}

void ExecutionState::removeFnAlias(std::string fn) {
  Gklee::Logging::enterFunc( fn , __PRETTY_FUNCTION__ );
  fnAliases.erase(fn);
  Gklee::Logging::exitFunc();
}

/**/

std::ostream &klee::operator<<(std::ostream &os, const MemoryMap &mm) {
  os << "{";
  MemoryMap::iterator it = mm.begin();
  MemoryMap::iterator ie = mm.end();
  if (it!=ie) {
    os << "MO" << it->first->id << ":" << it->second;
    for (++it; it!=ie; ++it)
      os << ", MO" << it->first->id << ":" << it->second;
  }
  os << "}";
  return os;
}

bool ExecutionState::merge(const ExecutionState &b) {
  /*if (DebugLogStateMerge)
    std::cerr << "-- attempting merge of A:" 
              << this << " with B:" << &b << "--\n";
  if (pc != b.pc)
    return false;

  // XXX is it even possible for these to differ? does it matter? probably
  // implies difference in object states?
  if (symbolics!=b.symbolics)
    return false;

  {
    std::vector<StackFrame>::const_iterator itA = stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    while (itA!=stack.end() && itB!=b.stack.end()) {
      // XXX vaargs?
      if (itA->caller!=itB->caller || itA->kf!=itB->kf)
        return false;
      ++itA;
      ++itB;
    }
    if (itA!=stack.end() || itB!=b.stack.end())
      return false;
  }

  std::set< klee::ref<Expr> > aConstraints(constraints.begin(), constraints.end());
  std::set< klee::ref<Expr> > bConstraints(b.constraints.begin(), 
                                     b.constraints.end());
  std::set< klee::ref<Expr> > commonConstraints, aSuffix, bSuffix;
  std::set_intersection(aConstraints.begin(), aConstraints.end(),
                        bConstraints.begin(), bConstraints.end(),
                        std::inserter(commonConstraints, commonConstraints.begin()));
  std::set_difference(aConstraints.begin(), aConstraints.end(),
                      commonConstraints.begin(), commonConstraints.end(),
                      std::inserter(aSuffix, aSuffix.end()));
  std::set_difference(bConstraints.begin(), bConstraints.end(),
                      commonConstraints.begin(), commonConstraints.end(),
                      std::inserter(bSuffix, bSuffix.end()));
  if (DebugLogStateMerge) {
    std::cerr << "\tconstraint prefix: [";
    for (std::set< klee::ref<Expr> >::iterator it = commonConstraints.begin(), 
           ie = commonConstraints.end(); it != ie; ++it)
      std::cerr << *it << ", ";
    std::cerr << "]\n";
    std::cerr << "\tA suffix: [";
    for (std::set< klee::ref<Expr> >::iterator it = aSuffix.begin(), 
           ie = aSuffix.end(); it != ie; ++it)
      std::cerr << *it << ", ";
    std::cerr << "]\n";
    std::cerr << "\tB suffix: [";
    for (std::set< klee::ref<Expr> >::iterator it = bSuffix.begin(), 
           ie = bSuffix.end(); it != ie; ++it)
      std::cerr << *it << ", ";
    std::cerr << "]\n";
  }

  // We cannot merge if addresses would resolve differently in the
  // states. This means:
  // 
  // 1. Any objects created since the branch in either object must
  // have been free'd.
  //
  // 2. We cannot have free'd any pre-existing object in one state
  // and not the other

  if (DebugLogStateMerge) {
    std::cerr << "\tchecking object states\n";
    std::cerr << "A: " << addressSpace.objects << "\n";
    std::cerr << "B: " << b.addressSpace.objects << "\n";
  }
    
  std::set<const MemoryObject*> mutated;
  MemoryMap::iterator ai = addressSpace.objects.begin();
  MemoryMap::iterator bi = b.addressSpace.objects.begin();
  MemoryMap::iterator ae = addressSpace.objects.end();
  MemoryMap::iterator be = b.addressSpace.objects.end();
  for (; ai!=ae && bi!=be; ++ai, ++bi) {
    if (ai->first != bi->first) {
      if (DebugLogStateMerge) {
        if (ai->first < bi->first) {
          std::cerr << "\t\tB misses binding for: " << ai->first->id << "\n";
        } else {
          std::cerr << "\t\tA misses binding for: " << bi->first->id << "\n";
        }
      }
      return false;
    }
    if (ai->second != bi->second) {
      if (DebugLogStateMerge)
        std::cerr << "\t\tmutated: " << ai->first->id << "\n";
      mutated.insert(ai->first);
    }
  }
  if (ai!=ae || bi!=be) {
    if (DebugLogStateMerge)
      std::cerr << "\t\tmappings differ\n";
    return false;
  }
  
  // // merge stack

  klee::ref<Expr> inA = ConstantExpr::alloc(1, Expr::Bool);
  klee::ref<Expr> inB = ConstantExpr::alloc(1, Expr::Bool);
  for (std::set< klee::ref<Expr> >::iterator it = aSuffix.begin(), 
         ie = aSuffix.end(); it != ie; ++it)
    inA = AndExpr::create(inA, *it);
  for (std::set< klee::ref<Expr> >::iterator it = bSuffix.begin(), 
         ie = bSuffix.end(); it != ie; ++it)
    inB = AndExpr::create(inB, *it);

  // XXX should we have a preference as to which predicate to use?
  // it seems like it can make a difference, even though logically
  // they must contradict each other and so inA => !inB

  std::vector<StackFrame>::iterator itA = stack.begin();
  std::vector<StackFrame>::const_iterator itB = b.stack.begin();
  for (; itA!=stack.end(); ++itA, ++itB) {
    StackFrame &af = *itA;
    const StackFrame &bf = *itB;
    for (unsigned i=0; i<af.kf->numRegisters; i++) {
      klee::ref<Expr> &av = af.locals[i].value;
      const klee::ref<Expr> &bv = bf.locals[i].value;
      if (av.isNull() || bv.isNull()) {
        // if one is null then by implication (we are at same pc)
        // we cannot reuse this local, so just ignore
      } else {
        av = SelectExpr::create(inA, av, bv);
      }
    }
  }

  for (std::set<const MemoryObject*>::iterator it = mutated.begin(), 
         ie = mutated.end(); it != ie; ++it) {
    const MemoryObject *mo = *it;
    const ObjectState *os = addressSpace.findObject(mo);
    const ObjectState *otherOS = b.addressSpace.findObject(mo);
    assert(os && !os->readOnly && 
           "objects mutated but not writable in merging state");
    assert(otherOS);

    ObjectState *wos = addressSpace.getWriteable(mo, os);
    for (unsigned i=0; i<mo->size; i++) {
      klee::ref<Expr> av = wos->read8(i);
      klee::ref<Expr> bv = otherOS->read8(i);
      wos->write(i, SelectExpr::create(inA, av, bv));
    }
  }

  constraints = ConstraintManager();
  for (std::set< klee::ref<Expr> >::iterator it = commonConstraints.begin(), 
         ie = commonConstraints.end(); it != ie; ++it)
    constraints.addConstraint(*it);
  constraints.addConstraint(OrExpr::create(inA, inB));

  return true;
*/
  return false;
}
void ExecutionState::dumpStack(std::ostream &out) const {
  for (unsigned i = 0; i < stacks.size(); i++) {
    unsigned idx = 0;
    const KInstruction *target = tinfo.prevPCs[i];
    const stack_ty &stack = stacks[i];
    out << "\tthread " << i << "'s stack:";
    for (ExecutionState::stack_ty::const_reverse_iterator
         it = stack.rbegin(), ie = stack.rend(); it != ie; ++it) {
      const StackFrame &sf = *it;
      Function *f = sf.kf->function;
      const InstructionInfo &ii = *target->info;
      out << "\t#" << idx++ 
          << " " << std::setw(8) << std::setfill('0') << ii.assemblyLine
          << " in " << f->getName().str() << " (";
      // Yawn, we could go up and print varargs if we wanted to.
      unsigned index = 0;
      for (Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
           ai != ae; ++ai) {
        if (ai!=f->arg_begin()) out << ", ";
        
        out << ai->getName().str();
        // XXX should go through function
        klee::ref<Expr> value = sf.locals[sf.kf->getArgRegister(index++)].value;
        if (isa<ConstantExpr>(value))
          out << "=" << value;
      }
      out << ")";
      if (ii.file != "")
        out << " at " << ii.file << ":" << ii.line;
      out << "\n";
      target = sf.caller;
    }
  }
}

// CUDA stuffs

KInstIterator ExecutionState::getPC() { return tinfo.getPC(); }
void ExecutionState::setPC(KInstIterator _pc) { tinfo.setPC(_pc); }
KInstIterator ExecutionState::getPrevPC() { return tinfo.getPrevPC(); }
void ExecutionState::setPrevPC(KInstIterator _pc) { tinfo.setPrevPC(_pc); }
void ExecutionState::incPC() { tinfo.incPC(); }

ExecutionState::stack_ty& ExecutionState::getCurStack() { 
  
  return stacks[tinfo.get_cur_tid()];
}

void ExecutionState::reconfigGPU() {
  // extend the shared memories
  std::vector<AddressSpace> &sharedMems = addressSpace.sharedMemories;  
  unsigned oldsize = sharedMems.size();
  AddressSpace space; 
  unsigned k = 0;
  for (unsigned z = 0; z < GPUConfig::GridSize[2]; z++) { 
    for (unsigned y = 0; y < GPUConfig::GridSize[1]; y++) { 
      for (unsigned x = 0; x < GPUConfig::GridSize[0]; x++) {
        k = GPUConfig::GridSize[0] * (GPUConfig::GridSize[1] * z + y) + x;
        if (k >= oldsize) {
          sharedMems.push_back(space);
          sharedMems.back().ctype = GPUConfig::SHARED;
        } 
        MemoryObject* mo = tinfo.block_id_mo; 
        ObjectState* os = new ObjectState(mo);
        os->write32(0, x);
        os->write32(4, y);
        os->write32(8, z);
        sharedMems[k].bindObject(mo, os);
      }
    }
  }

  if (GPUConfig::num_blocks < oldsize) {
    for (unsigned i = GPUConfig::num_blocks; i < oldsize; i++)
      sharedMems.pop_back();
  }

  // extend the local memories
  std::vector<AddressSpace>& localMems = addressSpace.localMemories;
  oldsize = localMems.size();
  k = 0;
  for (unsigned gz = 0; gz < GPUConfig::GridSize[2]; gz++) 
    for (unsigned gy = 0; gy < GPUConfig::GridSize[1]; gy++) 
      for (unsigned gx = 0; gx < GPUConfig::GridSize[0]; gx++) {
        unsigned g = (GPUConfig::GridSize[0] * (GPUConfig::GridSize[1] * gz + gy) + gx) * GPUConfig::block_size;
        for (unsigned z = 0; z < GPUConfig::BlockSize[2]; z++) 
          for (unsigned y = 0; y < GPUConfig::BlockSize[1]; y++) 
            for (unsigned x = 0; x < GPUConfig::BlockSize[0]; x++) {
  	      k = g + GPUConfig::BlockSize[0] * (GPUConfig::BlockSize[1] * z + y) + x;
  	      if (k >= oldsize) { // add more local memories
  	        localMems.push_back(space);
	        localMems.back().ctype = GPUConfig::LOCAL;
	      }
  	      MemoryObject* mo = tinfo.thread_id_mo;
  	      // std::cerr << "tid: <" << x << "," << y << "," << z << ">";
  	      ObjectState* os = new ObjectState(mo);
  	      os->write32(0, x);
  	      os->write32(4, y);
  	      os->write32(8, z);
  	      localMems[k].bindObject(mo, os);
            }
      }

  // write the gridsize into the memory (set gridDim...)
  for (unsigned i = 0; i < 3; i++) {
    tinfo.grid_size_os->write32(i * 4, GPUConfig::GridSize[i]);
  } 

  // write the blocksize into the memory (set blockDim...)
  for (unsigned i = 0; i < 3; i++) {
    tinfo.block_size_os->write32(i * 4, GPUConfig::BlockSize[i]);
  }
  
  // extend or decrease the stacks
  if (GPUConfig::num_threads < oldsize) {
    for (unsigned i = GPUConfig::num_threads; i < oldsize; i++) {
      localMems.pop_back();
      stacks.pop_back();
      incomingBBIndex.pop_back();
    }
  } else {
    for (unsigned i = oldsize; i < GPUConfig::num_threads; i++) {
      stack_ty stk;
      stacks.push_back(stk);
      incomingBBIndex.push_back(0);
    }
  }

  // extend the instAccessSets
  unsigned iassize = addressSpace.instAccessSets.size();
  InstAccessSet instSet; 
  RefDivRegionSetVec regionSetVec;
  BBAccessSet bbSet;

  for (unsigned i = iassize; i<GPUConfig::num_threads; i++) {
    addressSpace.instAccessSets.push_back(instSet);
    addressSpace.divRegionSets.push_back(regionSetVec);
    addressSpace.bbAccessSets.push_back(bbSet);
  }  
  
  // dump host, shared and local memories layout. 
  if (GPUConfig::verbose > 0) {
    std::cout << "Please test the verbose one!" << std::endl;
    addressSpace.dump(0x11);
  }
}

void ExecutionState::reconfigGPUSymbolic() {
  // extend the shared memories
  std::vector<AddressSpace> &sharedMems = addressSpace.sharedMemories;  
  unsigned oldsize = sharedMems.size();
  AddressSpace space; 
  unsigned k;

  std::string bidName0 = "bid_arr_k" + llvm::utostr(kernelNum) + "_" + llvm::utostr(0);
  std::string bidName1 = "bid_arr_k" + llvm::utostr(kernelNum) + "_" + llvm::utostr(1);
  const Array *blockArray0 = new Array(bidName0, tinfo.block_id_mo->size);
  addSymbolic(tinfo.block_id_mo, blockArray0);
  const Array *blockArray1 = new Array(bidName1, tinfo.block_id_mo->size);
  addSymbolic(tinfo.block_id_mo, blockArray1);

  for (k = 0; k < GPUConfig::sym_num_blocks; k++) {
    if (k >= oldsize) {
      sharedMems.push_back(space);
      sharedMems.back().ctype = GPUConfig::SHARED;
    }
    MemoryObject* mo = tinfo.block_id_mo; 

    if (k == 1) { 
      ObjectState* os1 = new ObjectState(mo, blockArray1); 
      sharedMems[k].bindObject(mo, os1);
    } else {
      ObjectState* os0 = new ObjectState(mo, blockArray0);
      sharedMems[k].bindObject(mo, os0);
    }
  }

  if (GPUConfig::sym_num_blocks < oldsize) {
    for (unsigned i = GPUConfig::sym_num_blocks; i < oldsize; i++)
      sharedMems.pop_back();
  }

  // extend the local memories
  std::string tidName0 = "tid_arr_k" + llvm::utostr(kernelNum) + "_" + llvm::utostr(0);
  std::string tidName1 = "tid_arr_k" + llvm::utostr(kernelNum) + "_" + llvm::utostr(1);

  const Array *threadArray0 = new Array(tidName0, tinfo.thread_id_mo->size);
  addSymbolic(tinfo.thread_id_mo, threadArray0);
  const Array *threadArray1 = new Array(tidName1, tinfo.thread_id_mo->size);
  addSymbolic(tinfo.thread_id_mo, threadArray1);

  std::vector<AddressSpace>& localMems = addressSpace.localMemories;
  oldsize = localMems.size();
  for (k = 0; k < GPUConfig::sym_num_threads; k++) {
    if (k >= oldsize) {
      localMems.push_back(space);
      localMems.back().ctype = GPUConfig::LOCAL;
    }
    MemoryObject* mo = tinfo.thread_id_mo;

    if (k == 1) {
      ObjectState* os1 = new ObjectState(mo, threadArray1);
      localMems[k].bindObject(mo, os1);
    } else { 
      ObjectState* os0 = new ObjectState(mo, threadArray0);
      localMems[k].bindObject(mo, os0);
    }
  }

  // write the gridsize into the memory (set gridDim...)
  for (unsigned i = 0; i < 3; i++) {
    tinfo.grid_size_os->write32(i * 4, GPUConfig::GridSize[i]);
  }

  // write the blocksize into the memory (set blockDim...)
  for (unsigned i = 0; i < 3; i++) {
    tinfo.block_size_os->write32(i * 4, GPUConfig::BlockSize[i]);
  }
  
  // extend or decrease the stacks
  if (GPUConfig::sym_num_threads < oldsize) {
    for (unsigned i = GPUConfig::sym_num_threads; i < oldsize; i++) {
      localMems.pop_back();
      stacks.pop_back();
      incomingBBIndex.pop_back();
    }
  } else {
    for (unsigned i = oldsize; i < GPUConfig::sym_num_threads; i++) {
      stack_ty stk;
      stacks.push_back(stk);
      incomingBBIndex.push_back(0);
    }
  }

  // dump host, shared and local memories layout. 
  if (GPUConfig::verbose > 0) {
    std::cout << "Please test the verbose one!" << std::endl;
    addressSpace.dump(0x11);
  }
}

void ExecutionState::constructUnboundedBlockEncodedConstraint(unsigned cur_bid) {
  ObjectState *os = addressSpace.cpuMemory.findNonConstantObject(tinfo.sym_gdim_mo);

  klee::ref<Expr> gdimx = os->read(0, Expr::Int32);
  klee::ref<Expr> gdimy = os->read(4, Expr::Int32);  
  klee::ref<Expr> gdimz = os->read(8, Expr::Int32);  

  klee::ref<Expr> gdimxConstr = AndExpr::create(UgtExpr::create(gdimx, ConstantExpr::create(0, Expr::Int32)), 
                                          UleExpr::create(gdimx, ConstantExpr::create(GPUConfig::SymMaxGridSize[0], Expr::Int32)));
  klee::ref<Expr> gdimyConstr = AndExpr::create(UgtExpr::create(gdimy, ConstantExpr::create(0, Expr::Int32)), 
                                          UleExpr::create(gdimy, ConstantExpr::create(GPUConfig::SymMaxGridSize[1], Expr::Int32)));
  klee::ref<Expr> gdimzConstr = AndExpr::create(UgtExpr::create(gdimz, ConstantExpr::create(0, Expr::Int32)), 
                                          UleExpr::create(gdimz, ConstantExpr::create(GPUConfig::SymMaxGridSize[2], Expr::Int32)));

  klee::ref<Expr> gdimConstraint = AndExpr::create(AndExpr::create(gdimxConstr, gdimyConstr),
                                             gdimzConstr); 
  addConstraint(gdimConstraint);

  // construct thread constraint based on the symbolic config 
  ObjectState *bos = addressSpace.localMemories[cur_bid].findNonConstantObject(tinfo.thread_id_mo); 
  klee::ref<Expr> bidx = bos->read(0, Expr::Int32);
  klee::ref<Expr> bidy = bos->read(4, Expr::Int32);
  klee::ref<Expr> bidz = bos->read(8, Expr::Int32);

  klee::ref<Expr> bidxConstr = AndExpr::create(UltExpr::create(bidx, gdimx), 
                                         UgeExpr::create(bidx, ConstantExpr::create(0, Expr::Int32))); 
  klee::ref<Expr> bidyConstr = AndExpr::create(UltExpr::create(bidy, gdimy), 
                                         UgeExpr::create(bidy, ConstantExpr::create(0, Expr::Int32))); 
  klee::ref<Expr> bidzConstr = AndExpr::create(UltExpr::create(bidz, gdimz), 
                                         UgeExpr::create(bidz, ConstantExpr::create(0, Expr::Int32))); 
  klee::ref<Expr> totalExpr = AndExpr::create(AndExpr::create(bidxConstr, bidyConstr), bidzConstr);
  addConstraint(totalExpr);
}

void ExecutionState::constructUnboundedThreadEncodedConstraint(unsigned cur_tid) {
  ObjectState *os = addressSpace.cpuMemory.findNonConstantObject(tinfo.sym_bdim_mo);

  klee::ref<Expr> bdimx = os->read(0, Expr::Int32);
  klee::ref<Expr> bdimy = os->read(4, Expr::Int32);  
  klee::ref<Expr> bdimz = os->read(8, Expr::Int32);  

  klee::ref<Expr> bdimxConstr = AndExpr::create(UgtExpr::create(bdimx, ConstantExpr::create(0, Expr::Int32)), 
                                          UleExpr::create(bdimx, ConstantExpr::create(GPUConfig::SymMaxBlockSize[0], Expr::Int32)));
  klee::ref<Expr> bdimyConstr = AndExpr::create(UgtExpr::create(bdimy, ConstantExpr::create(0, Expr::Int32)), 
                                          UleExpr::create(bdimy, ConstantExpr::create(GPUConfig::SymMaxBlockSize[1], Expr::Int32)));
  klee::ref<Expr> bdimzConstr = AndExpr::create(UgtExpr::create(bdimz, ConstantExpr::create(0, Expr::Int32)), 
                                          UleExpr::create(bdimz, ConstantExpr::create(GPUConfig::SymMaxBlockSize[2], Expr::Int32)));

  klee::ref<Expr> bdimConstraint = AndExpr::create(AndExpr::create(bdimxConstr, bdimyConstr),
                                             bdimzConstr); 
  addConstraint(bdimConstraint);

  // construct thread constraint based on the symbolic config 
  ObjectState *tos = addressSpace.localMemories[cur_tid].findNonConstantObject(tinfo.thread_id_mo); 
  klee::ref<Expr> tidx = tos->read(0, Expr::Int32);
  klee::ref<Expr> tidy = tos->read(4, Expr::Int32);
  klee::ref<Expr> tidz = tos->read(8, Expr::Int32);

  klee::ref<Expr> tidxConstr = AndExpr::create(UltExpr::create(tidx, bdimx), 
                                         UgeExpr::create(tidx, ConstantExpr::create(0, Expr::Int32))); 
  klee::ref<Expr> tidyConstr = AndExpr::create(UltExpr::create(tidy, bdimy), 
                                         UgeExpr::create(tidy, ConstantExpr::create(0, Expr::Int32))); 
  klee::ref<Expr> tidzConstr = AndExpr::create(UltExpr::create(tidz, bdimz), 
                                         UgeExpr::create(tidz, ConstantExpr::create(0, Expr::Int32))); 
  klee::ref<Expr> totalExpr = AndExpr::create(AndExpr::create(tidxConstr, tidyConstr), tidzConstr);
  addConstraint(totalExpr);
}

// construct the block-level encoded constraint ...
void ExecutionState::constructBlockEncodedConstraint(klee::ref<Expr> &constraint, unsigned cur_bid) {
  ObjectState *os = addressSpace.findNonConstantObject(tinfo.block_id_mo, cur_bid);
  // bid x ...
  klee::ref<Expr> bidx = os->read(0, Expr::Int32);
  klee::ref<Expr> xcond = AndExpr::create(SltExpr::create(bidx, ConstantExpr::create(GPUConfig::GridSize[0], Expr::Int32)), 
                                    SgeExpr::create(bidx, ConstantExpr::create(0, Expr::Int32)));
  // bid y ...
  klee::ref<Expr> bidy = os->read(4, Expr::Int32);
  klee::ref<Expr> ycond = AndExpr::create(SltExpr::create(bidy, ConstantExpr::create(GPUConfig::GridSize[1], Expr::Int32)), 
                                    SgeExpr::create(bidy, ConstantExpr::create(0, Expr::Int32)));
  // bid z ...
  klee::ref<Expr> bidz = os->read(8, Expr::Int32);
  klee::ref<Expr> zcond = AndExpr::create(SltExpr::create(bidz, ConstantExpr::create(GPUConfig::GridSize[2], Expr::Int32)), 
                                    SgeExpr::create(bidz, ConstantExpr::create(0, Expr::Int32)));
  // regardless of number of grid dimensions
  constraint = AndExpr::create(AndExpr::create(xcond, ycond), zcond);
}

// construct the thread-level encoded constraint ...
void ExecutionState::constructThreadEncodedConstraint(klee::ref<Expr> &constraint, unsigned cur_tid) {
  ObjectState *os = addressSpace.findNonConstantObject(tinfo.thread_id_mo, cur_tid);
  // tid x ...
  klee::ref<Expr> tidx = os->read(0, Expr::Int32);
  klee::ref<Expr> xcond = AndExpr::create(SltExpr::create(tidx, ConstantExpr::create(GPUConfig::BlockSize[0], Expr::Int32)), 
                                    SgeExpr::create(tidx, ConstantExpr::create(0, Expr::Int32)));
  // tid y ...
  klee::ref<Expr> tidy = os->read(4, Expr::Int32);
  klee::ref<Expr> ycond = AndExpr::create(SltExpr::create(tidy, ConstantExpr::create(GPUConfig::BlockSize[1], Expr::Int32)), 
                                    SgeExpr::create(tidy, ConstantExpr::create(0, Expr::Int32)));
  // tid z ...
  klee::ref<Expr> tidz = os->read(8, Expr::Int32);
  klee::ref<Expr> zcond = AndExpr::create(SltExpr::create(tidz, ConstantExpr::create(GPUConfig::BlockSize[2], Expr::Int32)), 
                                    SgeExpr::create(tidz, ConstantExpr::create(0, Expr::Int32))); 
  // regardless of number of block dimensions
  constraint = AndExpr::create(AndExpr::create(xcond, ycond), zcond);
}

static bool isTwoInstIdentical(llvm::Instruction *inst1, llvm::Instruction *inst2) {
  std::string func1Name = inst1->getParent()->getParent()->getName().str();
  std::string func2Name = inst2->getParent()->getParent()->getName().str();
  llvm::BasicBlock *bb1 = inst1->getParent();
  llvm::BasicBlock *bb2 = inst2->getParent();

  return func1Name.compare(func2Name) == 0 
           && bb1 == bb2
             && inst1->isIdenticalTo(inst2); 
}

static bool checkLastBranchDivRegionSet(std::vector<BranchDivRegionSet> &branchDivRegionSets, 
                                        llvm::Instruction *brInst, unsigned &brNum) {
  bool findBr = false;   
  unsigned size = branchDivRegionSets.size();

  if (isTwoInstIdentical(branchDivRegionSets[size-1].brInst, brInst)
      && !branchDivRegionSets[size-1].explored) {
    brNum = size-1;
    findBr = true;
  }
 
  return findBr;
}

static bool existTidInNonSyncSet(std::vector< std::vector<unsigned> > &nonSyncSets, 
                                 unsigned tid) {
  for (unsigned i = 0; i < nonSyncSets.size(); i++) {
    for (unsigned j = 0; j < nonSyncSets[i].size(); j++) {
      if (nonSyncSets[i][j] == tid) return true;  
    }
  }
  return false;
}

static void updateCurrentTidRegion(std::vector<CorrespondTid> &cTidSets, 
                                   std::vector<BranchDivRegionSet> &branchDivRegionSets,
                                   llvm::Instruction *curInst,
                                   unsigned tid, unsigned instSeqNum) {
  for (unsigned i = 0; i < branchDivRegionSets.size(); i++) {
    if (branchDivRegionSets[i].allSync) continue;

    bool tidFound = false; 
    std::vector<BranchDivRegionVec> &branchSets = branchDivRegionSets[i].branchSets;  
    for (unsigned j = 0; j < branchSets.size(); j++) {
      std::vector<BranchDivRegion> &branchDivRegionVec = branchSets[j].branchDivRegionVec;
      for (unsigned k = 0; k < branchDivRegionVec.size(); k++) {
        if (branchDivRegionVec[k].tid == tid) {
          branchDivRegionVec[k].regionEnd = instSeqNum;
          tidFound = true;
          break;
        }
      }
      if (tidFound) break;
    }
  }
}

static void identifyDivRegionsForBranch(BranchDivRegionSet &branchDivRegionSet, bool isCondBr, 
                                        llvm::Instruction *curInst, unsigned tid, unsigned instSeqNum) {
  bool branchFound = false;
  llvm::BasicBlock* bb = curInst->getParent();
  std::vector< std::vector<unsigned> > &nonSyncSets = branchDivRegionSet.nonSyncSets;
  std::vector<BranchDivRegionVec> &branchSets = branchDivRegionSet.branchSets;
  for (unsigned i = 0; i < branchSets.size(); i++) {
    if (branchSets[i].whichBB == bb) {
      nonSyncSets[i].push_back(tid);
      branchSets[i].branchDivRegionVec.push_back(BranchDivRegion(tid, instSeqNum, instSeqNum));
      branchFound = true;
      break;
    }
  }
  assert(branchFound && "The branch side not found!\n");
}

BasicBlock *ExecutionState::findNearestCommonPostDominator(llvm::PostDominatorTree *postDominator, 
                                                           llvm::Instruction *inst, bool isCondBr) {
  BasicBlock *postDomBB = NULL;
  llvm::Function *fn = inst->getParent()->getParent();
  BasicBlock *BB1 = NULL;
  BasicBlock *BB2 = NULL;
  if (isCondBr) {
    BranchInst *bi = cast<BranchInst>(inst);
    BB1 = bi->getSuccessor(0);
    BB2 = bi->getSuccessor(1);
  } else {
    SwitchInst *si = cast<SwitchInst>(inst);
    assert(si->getNumSuccessors() >= 2 && "Number of successors smaller than 2!\n");
    // pick two successors
    BB1 = si->getSuccessor(0);
    BB2 = si->getSuccessor(1);
  }
  postDominator->runOnFunction(*fn);
  postDomBB = postDominator->findNearestCommonDominator(BB1, BB2);

  if (postDomBB && GPUConfig::verbose > 0) {
    std::cout << "The branch or switch instruction: " << std::endl;
    inst->dump();
    std::cout << "The nearest common post-dominator for branch inst: " << std::endl;
    postDomBB->dump();    
  }
  return postDomBB;
}

static bool allThreadsExploreTheSameSide(BranchDivRegionSet &branchDivRegionSet, 
                                         unsigned sTid, unsigned eTid) {
  std::vector< std::vector<unsigned> > &nonSyncSets = branchDivRegionSet.nonSyncSets;
  unsigned num = 0; 
  unsigned which = 0;
  unsigned total = eTid - sTid + 1;

  for (unsigned i = 0; i < nonSyncSets.size(); i++) {
    if (nonSyncSets[i].size() > 0) {
      num++;
      which = i;
    }
  } 

  if (num == 1 && nonSyncSets[which].size() == total)
    return true;
  else 
    return false;
}

void ExecutionState::addBranchDivRegionSet(llvm::PostDominatorTree *postDominator, 
                                           llvm::Instruction *inst, bool isCondBr, unsigned instSeqNum) {
  std::vector<BranchDivRegionSet> &branchDivRegionSets = addressSpace.branchDivRegionSets;
  unsigned tid = tinfo.get_cur_tid();
  unsigned size = branchDivRegionSets.size();
  unsigned brNum = 0;

  if (size == 0 || !checkLastBranchDivRegionSet(branchDivRegionSets, inst, brNum)) {
    BasicBlock *postDomBB = findNearestCommonPostDominator(postDominator, inst, isCondBr); 
    branchDivRegionSets.push_back(BranchDivRegionSet(inst, postDomBB, isCondBr, false, false));
    brNum = size;
    size++; 
    // conditional branch
    std::vector<unsigned> nonSyncSet;
    if (isCondBr) {
      BranchInst *bi = cast<BranchInst>(inst);
      llvm::BasicBlock* trueBB = bi->getSuccessor(0);
      llvm::BasicBlock* falseBB = bi->getSuccessor(1);
      // first denotes the true path
      branchDivRegionSets[brNum].branchSets.push_back(BranchDivRegionVec(trueBB)); 
      branchDivRegionSets[brNum].nonSyncSets.push_back(nonSyncSet);
      // second denotes the false path 
      branchDivRegionSets[brNum].branchSets.push_back(BranchDivRegionVec(falseBB)); 
      branchDivRegionSets[brNum].nonSyncSets.push_back(nonSyncSet);
    } else {
      // switch branch
      SwitchInst *si = cast<SwitchInst>(inst);
      for (unsigned i = 0; i < si->getNumSuccessors(); i++) {
        branchDivRegionSets[brNum].branchSets.push_back(BranchDivRegionVec(si->getSuccessor(i)));
        branchDivRegionSets[brNum].nonSyncSets.push_back(nonSyncSet);
      }
    }
  }

  KInstIterator ki = getPC();
  identifyDivRegionsForBranch(branchDivRegionSets[brNum], isCondBr, ki->inst, tid, instSeqNum);
  if (cTidSets[tid].inBranch)
    updateCurrentTidRegion(cTidSets, branchDivRegionSets, ki->inst, tid, instSeqNum);
  cTidSets[tid].inBranch = true;
  
  if (!tinfo.warpInBranch) {
    bool sameSide = allThreadsExploreTheSameSide(branchDivRegionSets[size-1], 
                                                 tinfo.get_cur_warp_start_tid(), 
                                                 tinfo.get_cur_warp_end_tid());
    if (sameSide) {
      // All threads in the warp go to the same side of branch 
      branchDivRegionSets.pop_back(); 
      for (unsigned i = tinfo.get_cur_warp_start_tid(); i <= tinfo.get_cur_warp_end_tid(); i++) 
        cTidSets[i].inBranch = false;
    } else {
      bool allInBranch = true;
      for (unsigned i = tinfo.get_cur_warp_start_tid(); i <= tinfo.get_cur_warp_end_tid(); i++) {
        if (!cTidSets[i].inBranch) {
          allInBranch = false;
          break;
        }
      }
      tinfo.warpInBranch = allInBranch;
    }
  }
}

static bool removeThreadFromNonSyncSet(std::vector< std::vector<unsigned> > &nonSyncSets, unsigned tid) {
  bool findAndErase = false;
  for (unsigned i = 0; i < nonSyncSets.size(); i++) {
    std::vector<unsigned>::iterator vi = nonSyncSets[i].begin();
    for (;vi != nonSyncSets[i].end(); vi++) {
      if (*vi == tid) {
        nonSyncSets[i].erase(vi);
        findAndErase = true;
        break;
      }
    }
    if (findAndErase) break;
  }
  
  bool allSync = true;
  for (unsigned i = 0; i < nonSyncSets.size(); i++) {
    if (!nonSyncSets[i].empty()) {
      allSync = false;
      break;
    }
  }
  return allSync; 
}

static bool isTwoBBIdentical(BasicBlock *postBB, BasicBlock *curBB) {
  return curBB->getParent()->getName().compare(postBB->getParent()->getName()) == 0
            && curBB == postBB; 
}

static bool encounterImplicitSyncBarrier(std::vector<BranchDivRegionSet> &branchDivRegionSets, 
                                         std::vector<CorrespondTid> &cTidSets, 
                                         unsigned tid, BasicBlock *curBB) {
  unsigned size = branchDivRegionSets.size();
  bool encounter = false;
  for (int i = size-1; i >= 0; i--) {
    if (!branchDivRegionSets[i].allSync && 
         existTidInNonSyncSet(branchDivRegionSets[i].nonSyncSets, tid)) {
      BasicBlock *postDomBB = branchDivRegionSets[i].postDominator;
      if (postDomBB && isTwoBBIdentical(postDomBB, curBB)) { 
        bool allSync = removeThreadFromNonSyncSet(branchDivRegionSets[i].nonSyncSets, tid);
        branchDivRegionSets[i].allSync = allSync;
        cTidSets[tid].syncEncounter = true;
        encounter = true;
      }
    }
  }
  return encounter;
}

void ExecutionState::updateBranchDivRegionSet(llvm::Instruction *curInst, unsigned instSeqNum) {
  unsigned cur_tid = tinfo.get_cur_tid();
  std::vector<BranchDivRegionSet> &branchDivRegionSets = addressSpace.branchDivRegionSets;
  updateCurrentTidRegion(cTidSets, branchDivRegionSets, curInst, cur_tid, instSeqNum);

  if (!cTidSets[cur_tid].barrierEncounter) {
    llvm::BasicBlock *curBB = curInst->getParent();
    bool encounter = encounterImplicitSyncBarrier(branchDivRegionSets, cTidSets, cur_tid, curBB); 
    if (encounter) {
      // some threads encounter implicit barriers, 
      // and some states need to be upgraded ... 
      updateStateAfterEncounterBarrier();
    }
  }
}

void ExecutionState::encounterSyncthreadsBarrier(unsigned cur_tid) {
  if (!UseSymbolicConfig) {
    bool encounter = false;
    cTidSets[cur_tid].syncEncounter = true;
    cTidSets[cur_tid].barrierEncounter = true;
    std::vector<BranchDivRegionSet> &branchDivRegionSets = addressSpace.branchDivRegionSets;
    unsigned size = branchDivRegionSets.size();
    for (int i = size-1; i >= 0; i--) {
      if (!branchDivRegionSets[i].allSync 
           && existTidInNonSyncSet(branchDivRegionSets[i].nonSyncSets, cur_tid)) {
        bool allSync = removeThreadFromNonSyncSet(branchDivRegionSets[i].nonSyncSets, cur_tid);
        branchDivRegionSets[i].allSync = allSync;
        encounter = true;
      }
    }
    if (encounter) updateStateAfterEncounterBarrier();
  } else {
    cTidSets[cur_tid].syncEncounter = true; 
    cTidSets[cur_tid].barrierEncounter = true;
    // create the inherit cond from parametric flow tree 
    ParaTreeNode *current = getCurrentParaTree().getCurrentNode(); 
    if (current) {
      klee::ref<Expr> expr = getCurrentParaTree().getCurrentNodeTDCExpr(); 
      cTidSets[cur_tid].inheritExpr = constraints.simplifyExpr(expr);
    }
    //std::cout << "cur_tid: " << cur_tid << std::endl;
    //cTidSets[cur_tid].inheritExpr->dump();

    if (!tinfo.warpInBranch) {
      for (unsigned i = 0; i < tinfo.symExecuteSet.size(); i++) {
         unsigned tid = tinfo.symExecuteSet[i];
         cTidSets[tid].syncEncounter = true;
         cTidSets[tid].barrierEncounter = true;
      }
    }
    updateStateAfterEncounterBarrier();
  }
}

// Indicate that threads encounter the explicit or implicit barrier.
void ExecutionState::updateStateAfterEncounterBarrier() {
  if (!UseSymbolicConfig) {
    tinfo.updateStateAfterBarriers(cTidSets, addressSpace.branchDivRegionSets);
  } else {
    unsigned cur_tid = tinfo.get_cur_tid();
    getCurrentParaTree().encounterExplicitBarrier(cTidSets, cur_tid);
  }  
}

void ExecutionState::moveToNextWarpAfterExplicitBarrier(bool moveToNextBI) {
  std::vector<BranchDivRegionSet> &branchDivRegionSets = addressSpace.branchDivRegionSets;

  for (unsigned i = 0; i < branchDivRegionSets.size(); i++) {
    if (!branchDivRegionSets[i].allSync) {
      branchDivRegionSets[i].allSync = true;
      std::vector< std::vector<unsigned> > &nonSyncSets = branchDivRegionSets[i].nonSyncSets;
      for (unsigned j = 0; j < nonSyncSets.size(); j++)
        nonSyncSets[j].clear();
    }
  }

  if (!moveToNextBI) {
    std::vector<BranchDivRegionSet> tmpIDRS = branchDivRegionSets;
    addressSpace.warpsBranchDivRegionSets.push_back(tmpIDRS);
    branchDivRegionSets.clear();
  } else {
    if (GPUConfig::verbose > 0)
      addressSpace.dumpWarpsBranchDivRegionSets();
    addressSpace.warpsBranchDivRegionSets.clear();
  }
}

void ExecutionState::restoreCorrespondTidSets() {
  if (!UseSymbolicConfig) {
    for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
      cTidSets[i].syncEncounter = false;
      cTidSets[i].barrierEncounter = false;
      cTidSets[i].inBranch = false;
    }
  } else {
    for (unsigned i = 0; i < cTidSets.size(); i++) {
      if (cTidSets[i].slotUsed) {
        cTidSets[i].syncEncounter = false;
        cTidSets[i].barrierEncounter = false;
        cTidSets[i].inBranch = false;
      } else break;
    }
  }
}

bool ExecutionState::allThreadsEncounterBarrier() {
  for (unsigned i = 0; i < GPUConfig::num_threads; i++) {
    if (!cTidSets[i].barrierEncounter)
      return false;
  }
  return true;
}

bool ExecutionState::allSymbolicThreadsEncounterBarrier() {
  for (unsigned i = 0; i < cTidSets.size(); i++) {
    if (cTidSets[i].slotUsed) {
      if (i != 1 && !cTidSets[i].barrierEncounter) {
        return false;
      }
    }
    else break;
  }
  return true;
}

void ExecutionState::copyAddressSpaceObjects(unsigned src, unsigned dst) {
  AddressSpace &srcSpace = addressSpace.localMemories[src];
  AddressSpace &dstSpace = addressSpace.localMemories[dst];
  for (MemoryMap::iterator oi = srcSpace.objects.begin(); 
       oi != srcSpace.objects.end(); ++oi) {
    if (!oi->first->is_builtin) {
      ObjectState *tmpOS = new ObjectState(*(oi->second));
      dstSpace.bindObject(oi->first, tmpOS);
    }
  }
} 

void ExecutionState::synchronizeBranchStacks(ParaTreeNode *current) {
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  unsigned sTid = configVec[0].sym_tid;
  for (unsigned i = 1; i < configVec.size(); i++) {
    unsigned tid = configVec[i].sym_tid;
    stacks[tid] = stacks[sTid];
    copyAddressSpaceObjects(sTid, tid);
  }
}

void ExecutionState::symEncounterPostDominator(llvm::Instruction *inst) {
  ParaTree &paraTree = getCurrentParaTree();
  llvm::BasicBlock *curBB = inst->getParent();
  ParaTreeNode *tmp = paraTree.getCurrentNode();
  klee::ref<Expr> cond = ConstantExpr::create(1, Expr::Bool);

  while (tmp != NULL) {
    unsigned which = tmp->whichSuccessor;
    std::vector<ParaConfig> &configVec = tmp->successorConfigVec;
    if (tmp->brInst && !configVec[which].postDomEncounter) {
      // check postDom is equal to curBB
      BasicBlock *postDomBB = tmp->postDom;

      if (postDomBB && isTwoBBIdentical(postDomBB, curBB)) {
        configVec[which].postDomEncounter = true;
        //GKLEE_INFO << "Threads in flow " << configVec[which].sym_tid 
        //           << " encounters the post dominator on the side "
        //           << tmp->whichSuccessor << "!" << std::endl;
      }
    }
    tmp = tmp->parent;
  }
}

ParaTreeSet& ExecutionState::getCurrentParaTreeSet() {
  return paraTreeSets[paraTreeSets.size()-1];
}

ParaTreeVec& ExecutionState::getCurrentParaTreeVec() {
  ParaTreeSet &paraTreeSet = paraTreeSets[paraTreeSets.size()-1];
  unsigned size = paraTreeSet.size();
  return paraTreeSet[size-1];
}

ParaTree& ExecutionState::getCurrentParaTree() {
  ParaTreeSet &paraTreeSet = paraTreeSets[paraTreeSets.size()-1];
  unsigned size = paraTreeSet.size();
  ParaTreeVec &paraTreeVec = paraTreeSet[size-1];
  unsigned curTree = tinfo.symParaTreeVec[0];
  return (curTree == 0)? paraTreeVec[curTree] : paraTreeVec[curTree-1];
}

klee::ref<Expr> ExecutionState::getTDCCondition(bool ignoreCur) {
  klee::ref<Expr> expr;
  if (!tinfo.is_GPU_mode) {
    expr = klee::ConstantExpr::create(1, Expr::Bool);
  } else {
    ParaTree &paraTree = getCurrentParaTree();
    ParaTreeNode *current = paraTree.getCurrentNode();

    if (current != NULL) {
      unsigned which = current->whichSuccessor;
      std::vector<ParaConfig> &configVec = current->successorConfigVec;
      if (current->symBrType == TDC) 
        expr = AndExpr::create(current->inheritCond, configVec[which].cond);
      else {
        if (ignoreCur)
          expr = current->inheritCond;
        else 
          expr = AndExpr::create(current->inheritCond, configVec[which].cond);
      }
    } else {
      unsigned tid = tinfo.get_cur_tid();
      expr = cTidSets[tid].inheritExpr;
    }
  }
  expr = constraints.simplifyExpr(expr);

  return expr;
}

void ExecutionState::dumpStateConstraint() {
  ConstraintManager &constr = constraints;
  unsigned i = 0;
  std::cout << "dump state constraint: " << std::endl;
  for (ConstraintManager::constraint_iterator ci = constr.begin(); 
       ci != constr.end(); ci++, i++) {
    std::cout << "constraint " << i << ": " << std::endl;
    (*ci)->dump();  
  }
}
