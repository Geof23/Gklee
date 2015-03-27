//===-- TimingSolver.h ------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_TIMINGSOLVER_H
#define KLEE_TIMINGSOLVER_H

#include "klee/Expr.h"
#include "klee/Solver.h"

#include <vector>

namespace klee {
  class ExecutionState;
  class Solver;
  class STPSolver;

  /// TimingSolver - A simple class which wraps a solver and handles
  /// tracking the statistics that we care about.
  class TimingSolver {
  public:
    Solver *solver;
    STPSolver *stpSolver;
    bool simplifyExprs;

  public:
    /// TimingSolver - Construct a new timing solver.
    ///
    /// \param _simplifyExprs - Whether expressions should be
    /// simplified (via the constraint manager interface) prior to
    /// querying.
    TimingSolver(Solver *_solver, STPSolver *_stpSolver, 
                 bool _simplifyExprs = true) 
      : solver(_solver), stpSolver(_stpSolver), simplifyExprs(_simplifyExprs) {}
    ~TimingSolver() {
      delete solver;
    }

    void setTimeout(double t) {
      stpSolver->setTimeout(t);
    }

    bool evaluate(const ExecutionState&, klee::ref<Expr>, Solver::Validity &result);

    bool mustBeTrue(const ExecutionState&, klee::ref<Expr>, bool &result);

    bool mustBeFalse(const ExecutionState&, klee::ref<Expr>, bool &result);

    bool mayBeTrue(const ExecutionState&, klee::ref<Expr>, bool &result);

    bool mayBeFalse(const ExecutionState&, klee::ref<Expr>, bool &result);

    bool getValue(const ExecutionState &, klee::ref<Expr> expr, 
			  klee::ref<ConstantExpr> &result);

    bool getInitialValues(const ExecutionState&, 
				  const std::vector<const Array*> &objects,
				  std::vector< std::vector<unsigned char> > &result);

    std::pair< klee::ref<Expr>, klee::ref<Expr> > 
    getRange(const ExecutionState&, klee::ref<Expr> query);

  };

}

#endif
