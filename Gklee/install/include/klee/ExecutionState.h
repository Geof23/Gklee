//===-- ExecutionState.h ----------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_EXECUTIONSTATE_H
#define KLEE_EXECUTIONSTATE_H

#include "klee/Constraints.h"
#include "klee/Expr.h"
#include "klee/Internal/ADT/TreeStream.h"

// FIXME: We do not want to be exposing these? :(
#include "../../lib/Core/AddressSpace.h"
#include "klee/Internal/Module/KInstIterator.h"
#include "../../lib/Core/CUDA.h"

#include <map>
#include <set>
#include <vector>

// ********************************************

namespace klee {
  class Array;
  class CallPathNode;
  class Cell;
  class KFunction;
  class KInstruction;
  class MemoryObject;
  class PTreeNode;
  class InstructionInfo;

  class ThreadInfo;

std::ostream &operator<<(std::ostream &os, const MemoryMap &mm);

struct StackFrame {
  KInstIterator caller;
  KFunction *kf;
  CallPathNode *callPathNode;

  std::vector<const MemoryObject*> allocas;
  Cell *locals;

  /// Minimum distance to an uncovered instruction once the function
  /// returns. This is not a good place for this but is used to
  /// quickly compute the context sensitive minimum distance to an
  /// uncovered instruction. This value is updated by the StatsTracker
  /// periodically.
  unsigned minDistToUncoveredOnReturn;

  // For vararg functions: arguments not passed via parameter are
  // stored (packed tightly) in a local (alloca) memory object. This
  // is setup to match the way the front-end generates vaarg code (it
  // does not pass vaarg through as expected). VACopy is lowered inside
  // of intrinsic lowering.
  MemoryObject *varargs;

  StackFrame(KInstIterator caller, KFunction *kf);
  StackFrame(const StackFrame &s);
  
  StackFrame& operator=(const StackFrame& s);

  ~StackFrame();
};



class ExecutionState {
public:
  typedef std::vector<StackFrame> stack_ty;
  typedef std::vector<stack_ty> stacks_ty;

private:
  // unsupported, use copy constructor
  ExecutionState &operator=(const ExecutionState&); 
  std::map< std::string, std::string > fnAliases;

public:
  bool fakeState;
  // Are we currently underconstrained?  Hack: value is size to make fake
  // objects.
  unsigned underConstrained;
  unsigned depth;
  
  // // pc - pointer to current instruction stream
  // KInstIterator pc, prevPC;

  ThreadInfo tinfo;   // the PCs of the threads
  stacks_ty stacks;    // the stacks of the threads

  ConstraintManager constraints;
  mutable double queryCost;
  double weight;

  // The address space is now a hierarchical memory model 
  // AddressSpace addressSpace;
  HierAddressSpace addressSpace;

  TreeOStream pathOS, symPathOS;
  unsigned instsSinceCovNew;
  bool coveredNew;

  /// Disables forking, set by user code.
  bool forkDisabled;

  std::map<const std::string*, std::set<unsigned> > coveredLines;
  PTreeNode *ptreeNode;

  /// ordered list of symbolics: used to generate test cases. 
  //
  // FIXME: Move to a shared list structure (not critical).
  std::vector< std::pair<const MemoryObject*, const Array*> > symbolics;

  // Used by the checkpoint/rollback methods for fake objects.
  // FIXME: not freeing things on branch deletion.
  MemoryMap shadowObjects;

  unsigned incomingBBIndex;

  std::string getFnAlias(std::string fn);
  void addFnAlias(std::string old_fn, std::string new_fn);
  void removeFnAlias(std::string fn);
  
private:
  ExecutionState() : fakeState(false), underConstrained(0), ptreeNode(0) {};

public:
  ExecutionState(KFunction *kf);

  // XXX total hack, just used to make a state so solver can
  // use on structure
  ExecutionState(const std::vector<ref<Expr> > &assumptions);

  ~ExecutionState();
  
  ExecutionState *branch();

  // for a single thread
  void pushFrame(KInstIterator caller, KFunction *kf);
  void popFrame();

  // for all threads
  void pushAllFrames(KInstIterator caller, KFunction *kf);
  void popAllFrames();

  void addSymbolic(const MemoryObject *mo, const Array *array) { 
    symbolics.push_back(std::make_pair(mo, array));
  }
  void addConstraint(ref<Expr> e) { 
    constraints.addConstraint(e); 
  }

  // used in "Searcher.cpp"
  bool merge(const ExecutionState &b);

  // for CUDA
  KInstIterator getPC();
  void setPC(KInstIterator _pc);
  KInstIterator getPrevPC();
  void setPrevPC(KInstIterator _pc);
  void incPC();
  stack_ty& getCurStack();

  // reconfigurate the GPU 
  void reconfigGPU();

};

}

#endif
