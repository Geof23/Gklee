//===-- Constraints.h -------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_CONSTRAINTS_H
#define KLEE_CONSTRAINTS_H

#include "klee/Expr.h"
#include <map>

// FIXME: Currently we use ConstraintManager for two things: to pass
// sets of constraints around, and to optimize constraints. We should
// move the first usage into a separate data structure
// (ConstraintSet?) which ConstraintManager could embed if it likes.
namespace klee {

class ExprVisitor;
  
class ConstraintManager {
public:
  typedef std::vector< klee::ref<Expr> > constraints_ty;
  typedef constraints_ty::iterator iterator;
  typedef constraints_ty::const_iterator const_iterator;

  ConstraintManager() {}

  // create from constraints with no optimization
  explicit
  ConstraintManager(const std::vector< klee::ref<Expr> > &_constraints) :
    constraints(_constraints) {}

  ConstraintManager(const ConstraintManager &cs) : constraints(cs.constraints) {}

  typedef std::vector< klee::ref<Expr> >::const_iterator constraint_iterator;

  // given a constraint which is known to be valid, attempt to 
  // simplify the existing constraint set
  void simplifyForValidConstraint(klee::ref<Expr> e);

  klee::ref<Expr> simplifyExpr(klee::ref<Expr> e) const;

  klee::ref<Expr> updateExprThroughReplacement(klee::ref<Expr> e, std::map< klee::ref<Expr>, klee::ref<Expr> > equalities) const; 

  void addConstraint(klee::ref<Expr> e);

  bool empty() const {
    return constraints.empty();
  }
  void clear() {
    constraints.clear();
  }
  klee::ref<Expr> back() const {
    return constraints.back();
  }
  constraint_iterator begin() const {
    return constraints.begin();
  }
  constraint_iterator end() const {
    return constraints.end();
  }
  std::vector< klee::ref<Expr> >& getConstraints() {
    return constraints;
  }
  size_t size() const {
    return constraints.size();
  }

  bool operator==(const ConstraintManager &other) const {
    return constraints == other.constraints;
  }

  void dump() const {
    for (constraint_iterator cit=constraints.begin(); 
                             cit!=constraints.end(); ++cit) {
      (*cit)->dump(); 
    }
  } 

private:

  std::vector< klee::ref<Expr> > constraints;

  // returns true iff the constraints were modified
  bool rewriteConstraints(ExprVisitor &visitor);

  void addConstraintInternal(klee::ref<Expr> e);
};

}

#endif /* KLEE_CONSTRAINTS_H */
