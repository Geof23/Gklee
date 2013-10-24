#ifndef KLEE_STRCONSTRAINT_H
#define KLEE_STRCONSTRAINT_H

#include <map>
#include "../Core/TimingSolver.h"
#include "klee/Constraints.h"
#include <iostream>


/*********************************************************/

namespace klee {


  // the class to manager the string constraints and their solving

class StrConstraint {

public:

  typedef std::vector < ref<Expr> > ConstrsType;

  // another solution; not used now
/*   typedef std::map <Expr*, const Array*> ExprSymVarMap; */
/*   typedef std::map <const Array*, unsigned> SymVarValueMap; */

  ConstraintManager constrs;        // cotaining the string expressions.
  ConstraintManager len_constrs;    // containing the length expressions 
                                    // or the expressions converted from string expressions.
  ConstraintManager assgn_constrs;  // containing the values (of the symbolic values) returned 
                                    // by the solver.

  /*  the work flow is as follws:
                                      convert                solve                  set lenss
   string constrs (w. symbolic lens) ---------> len_constrs ----------> assignments ------->
                                      convert                solver   
   string constrs (w. concrete lens) ---------> val_constrs ----------> assignments

   */


  std::set <const Array*> symlens;   // the symbolic variables (e.g. lengths of var-len strings)

/*   ExprSymVarMap expr2vars; */
/*   SymVarValueMap var2values; */

  STPSolver *solver;

  bool involveStr;        // Do the constraints contain strings?

public:

  StrConstraint() {
    // solver = new STPSolver(false, true); 
  }

  StrConstraint(const ConstraintManager& cm, STPSolver* solver) {
    // solver = new STPSolver(false, true); 
    this->solver = solver;
    constrs = cm;
//     for (ConstraintManager::constraint_iterator ii = cm.begin(); 
// 	 ii != cm.end(); ii++) {
//       constrs.push_back(*ii);
//     }
  }

  ~StrConstraint() {
    // delete solver; 
  }

      
  void addConstraint(ref<Expr> c) { constrs.addConstraint(c); }
  void addLenConstraint(ref<Expr> c) { len_constrs.addConstraint(c); }

  void addSymLen(const Array* c) {
    for (std::set<const Array*>::const_iterator ii = symlens.begin();
	 ii != symlens.end(); ii++) { 
      if ((*ii)->name == c->name)  // avoid adding duplicate variables
	return;
    }
    symlens.insert(c);
  }

  void dump() {
    std::cout << "\n ********  The contents of the current string constraint. ******* \n  ";
    std::cout << "\n           String Constraints: \n";
    for (ConstraintManager::const_iterator ii = constrs.begin(); 
	 ii != constrs.end(); ii++) {
     (*ii)->dump();
    }

    std::cout << "\n           Length/Resolved Constraints: \n";
    for (ConstraintManager::const_iterator ii = len_constrs.begin(); 
	 ii != len_constrs.end(); ii++) {
      (*ii)->dump();
    }

    std::cout << "\n           Assignment Constraints: \n";
    for (ConstraintManager::const_iterator ii = assgn_constrs.begin(); 
	 ii != assgn_constrs.end(); ii++) {
      (*ii)->dump();
    }
  }


  // use a known assignment to set the lengths of var-len strings
  // the values of other symbolic variables are also set.
  bool concretizeLengths(std::vector<const Array*>& objects, 
			 std::vector< std::vector<unsigned char> >& values);

  // obtain the symbolic variables in a constraint
  void getSymArrays(std::vector<const Array*>& objects);

  // consult the solver to get the (unique) value of an expression;
  // this value is calculated based on the known assignment.
  bool getValue(ref<Expr> expr, ref<ConstantExpr> &result) {
     return solver->getValue(Query(assgn_constrs, expr), result);
  }

/*   unsigned get_value(ref<Expr> expr) { */
/*     return var2values[expr2vars[expr.get()]]; */
/*   } */

/*   unsigned get_value(Expr* &expr) { */
/*     return var2values[expr2vars[expr]]; */
/*   } */

  // build constraints on the lengths of string expressions
  static ref<Expr> makeLengthConstraint(StrConstraint& constr, ref<Expr> expr) {
    return resolveStrExpr(constr, expr, Expr::MakeStrLen);
  }

  // build constraints on the contents (values) of string expressions
  static ref<Expr> makeStrConstraint(StrConstraint& constr, ref<Expr> expr) {
    return resolveStrExpr(constr, expr, Expr::MakeStrExpr);
  }

  // build string constraints
  static ref<Expr> resolveStrExpr(StrConstraint& constr, ref<Expr> expr, 
				  Expr::TravPurpose pur) {
    constr.len_constrs.clear();     // clear previous results
    constr.involveStr = false;      // recalculate this flag

    for (ConstraintManager::const_iterator ii = constr.constrs.begin(); 
	 ii != constr.constrs.end(); ii++) {
      ref<Expr> res = (*ii)->resolveStrExpr(constr, pur);
/*       std::cout << "\n res = "; */
/*       res->dump(); */
      constr.addLenConstraint(res);
    }

    return expr->resolveStrExpr(constr, pur);
  }

};

 
/***********************************************************************/

} // end namespace klee

#endif
