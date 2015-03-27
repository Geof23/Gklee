#ifndef KLEE_PATHREDUCTION_H
#define KLEE_PATHREDUCTION_H

#include "klee/Expr.h"
#include "klee/ExecutionState.h"
#include "llvm/Instructions.h"
#include "BCCoverage.h"


// ****************************************************************************************
// path reduction
// ****************************************************************************************

namespace klee {

// Heuristics for path reduction
struct Heuristics {
  typedef std::set< klee::ref<ReadExpr> > ReadSet;
  ReadSet readset;
  ReadSet cur_readset;  

  bool use_dep;  // activate the dependency analysis
  unsigned expected_tid;
  unsigned expected_cond;
  unsigned cur_cond;
  unsigned val;
  
  Heuristics() : use_dep(true), expected_tid(0), expected_cond(1), cur_cond(1), val(0) { };
};


// path reduction information
class PRInfo : public Heuristics {
  enum MODE {NONE, PER_BI, PER_THREAD};
  MODE mode;

public:
  PRInfo() : mode(NONE) { Heuristics(); }
  void init(char c, unsigned val, bool _use_dep = true);
  void init(std::string& op, bool _use_dep = true);

  static std::string toModeStr(char c);
  bool symFullyExplore(ExecutionState &current, BICovInfo &covInfo);
  bool fullyExplore(ExecutionState &current, BICovInfo &covInfo);
};

} // end namespace

#endif
