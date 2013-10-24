#ifndef KLEE_STRSOLVER_H
#define KLEE_STRSOLVER_H

#include "../Core/TimingSolver.h"
// #include "StrConstraint.h"
// #include <iostream>

/*********************************************************/

namespace klee {


class StringSolver : public TimingSolver {

public:
    
  StringSolver(Solver *_solver, STPSolver *_stpSolver, 
	       bool _simplifyExprs = true) 
    : TimingSolver(_solver, _stpSolver, _simplifyExprs) {};

  // evaluate an expression to be True or False
  // if fail then return Unkown
  bool evaluate(const ExecutionState&, ref<Expr>, Solver::Validity &result);

  // find the value for a satisfiable assignment
  // the main function of the solver
  bool getInitialValues(const ExecutionState&, 
			const std::vector<const Array*> &objects,
			std::vector< std::vector<unsigned char> > &result);

  // obtain a possible value of an expression 
  bool getValue(const ExecutionState& state, ref<Expr> expr, 
		ref<ConstantExpr> &result);

  bool mustBeTrue(const ExecutionState&, ref<Expr>, bool &result);
  
  bool mustBeFalse(const ExecutionState&, ref<Expr>, bool &result);
  
  bool mayBeTrue(const ExecutionState&, ref<Expr>, bool &result);
  
  bool mayBeFalse(const ExecutionState&, ref<Expr>, bool &result);

public:

  // similar to those in StringHandler.h

  void printAssignments(const std::vector<const Array*>& objects, 
			std::vector< std::vector<unsigned char> >& values);

  bool solveExpr(ref<Expr> expr, std::vector<const Array*>& objects, 
		 std::vector< std::vector<unsigned char> >& values);


  bool solveExpr(ConstraintManager &constr, ref<Expr> expr, 
		 std::vector<const Array*>& objects, 
		 std::vector< std::vector<unsigned char> >& values);
  

  bool solveExpr(ConstraintManager &constr, ref<Expr> expr) {
    std::vector<const Array*> objects;
    std::vector< std::vector<unsigned char> > values;
    bool b = solveExpr(constr, expr, objects, values);
    printAssignments(objects, values);
    return b;
  }

  bool solveExpr(ref<Expr> expr) {
    std::vector<const Array*> objects;
    std::vector< std::vector<unsigned char> > values;
    bool b = solveExpr(expr, objects, values);
    printAssignments(objects, values);
    return b;
  }


};  // class StringSolver

 
/***********************************************************************/

} // end namespace klee

#endif
