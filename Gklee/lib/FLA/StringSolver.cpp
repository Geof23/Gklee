
#include "StringSolver.h"
#include "klee/ExecutionState.h"
#include "klee/Solver.h"
#include "klee/Statistics.h"

#include "../Core/CoreStats.h"

#include "llvm/System/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "iostream"

#include "StrExpr.h"
#include "StringHandler.h"
#include "StrConstraint.h"

using namespace klee;
using namespace llvm;


/*******************************************************************
 Implementation of the declared methods
 ******************************************************************/


/*******************************************************************
 String Constraint
 ******************************************************************/

bool StrConstraint::concretizeLengths(std::vector<const Array*>& objects, 
				      std::vector< std::vector<unsigned char> >& values) {
  assgn_constrs.clear();

  for (unsigned i = 0; i < objects.size(); i++) {
    const Array* arr = objects[i];
    if (arr->kind == Array::LenVarKind) {
      LengthVar* var = (LengthVar*) arr;
      var->setArrayLength(values[i][0]);
    }
    
    klee::ref<Expr> e1 = StrExpr::createValueRead(arr, arr->size * 8);

    klee::ref<Expr> e2 = ConstantExpr::create(values[i][0], Expr::Int8);
    for (unsigned j = 1; j < values[i].size(); j++) {
      e2 = ConcatExpr::create(e2, ConstantExpr::create(values[i][j], Expr::Int8));
    }

    assgn_constrs.addConstraint(EqExpr::create(e1, e2));
  }
  
  return true;
}


void StrConstraint::getSymArrays(std::vector<const Array*>& objects) {
  for (std::set<const Array*>::const_iterator ii = symlens.begin();
       ii != symlens.end(); ii++) { 
    if ((*ii)->kind == Array::LenVarKind) {
      LengthVar* var = (LengthVar*) (*ii);
      objects.push_back(var->getLinkedArray());
    }
  }
}



/*******************************************************************
 String Solver
 ******************************************************************/


void StringSolver:: printAssignments( const std::vector<const Array*>& objects, 
				      std::vector< std::vector<unsigned char> >& values ) {
  printf("The assignments: \n");
  for (unsigned i=0; i < objects.size(); i++) {
    printf("object: %s \n", objects[i]->name.c_str());
    printf("value: ");
    for (unsigned j=0; j < values[i].size(); j++) {
      printf("%d ", values[i][j]);
    }
    printf("\n");
  }
}


bool StringSolver::solveExpr(klee::ref<Expr> expr, std::vector<const Array*>& objects, 
			     std::vector< std::vector<unsigned char> >& values) {
  ConstraintManager constraints;
  constraints.addConstraint(expr);
  Query qr(constraints, expr);
  return solver->getInitialObjectsValues(qr, objects, values);
}


bool StringSolver::solveExpr(ConstraintManager &constr, klee::ref<Expr> expr, 
			     std::vector<const Array*>& objects, 
			     std::vector< std::vector<unsigned char> >& values) {
  Query qr(constr, expr);
  return solver->getInitialObjectsValues(qr, objects, values);
}


bool StringSolver::evaluate(const ExecutionState& state, klee::ref<Expr> expr,
			    Solver::Validity &result) {

  // Fast path, to avoid timer and OS overhead.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(expr)) {
    result = CE->isTrue() ? Solver::True : Solver::False;
    return true;
  }

#ifdef FLA_DEBUG
  KLEE_INFO2 << "StringSolver: evaluate \n";
#endif

  StrConstraint constr(state.constraints, stpSolver);
  klee::ref<Expr> len_exp = StrConstraint::makeLengthConstraint(constr, expr);

#ifdef FLA_DEBUG
  KLEE_INFO2 << "the length constraint: \n ";
  len_exp->dump();
  constr.dump();
#endif

//   std::vector<const Array*> objects;
//   for (std::set<const Array*>::iterator ii = constr.symlens.begin(); 
//        ii !=  constr.symlens.end(); ii++) {
//     objects.push_back(*ii);
//   }

  bool success = solver->evaluate(Query(constr.len_constrs, len_exp), result);

  switch (result) {
    case Solver::True:
#ifdef FLA_DEBUG
      KLEE_INFO2 << "The length constraint is evaluated to be True. \n";
#endif
      // result = Solver::Unknown;   // still need to consider the contents
      break;
    case Solver::False:
#ifdef FLA_DEBUG
      KLEE_INFO2 << "The length constraint is evaluated to be False. \n";
#endif
      break;
    default:
#ifdef FLA_DEBUG
      KLEE_INFO2 << "The length constraint is evaluated to be Unknown. \n";
#endif
      break;
  }

  return success;


  // the following is for resolving and evaluating the string expression
  // skip at this moment to reduce solving time

  /*
  std::vector< std::vector<unsigned char> > values;
  bool b = solveExpr(constr.len_constrs, len_exp, objects, values);
  
  if (b) {
    KLEE_INFO2 << "The length constraint is satisfiable. \n";
    printAssignments(objects, values);
  }
  else {
    KLEE_INFO2 << "The length constraint is NOT satisfiable! \n";
    return false;
  }

  // set the lengths
  constr.concretizeLengths(objects, values);
  
  KLEE_INFO2 << "the resolved expression: \n ";
  // solve the string expression
  klee::ref<Expr> str_exp = StrConstraint::makeStrConstraint(constr, expr);
    
  objects.clear();
  values.clear();
  constr.getSymArrays(objects);
  
//   for (ConstraintManager::const_iterator ii = constr.assgn_constrs.begin(); 
//        ii != constr.assgn_constrs.end(); ii++) {
//     constr.len_constrs.addConstraint(*ii);
//   }

  KLEE_INFO2 << "the constraint of the string expression: \n ";
  str_exp->dump();
  constr.dump();
  
  bool success = solveExpr(constr.len_constrs, str_exp, objects, values);
  if (success) {
    KLEE_INFO << "The string expression is satisfiable. \n";
    printAssignments(objects, values);
  }
  else {
    KLEE_INFO << "The string expression is NOT satisfiable! \n";
    return false;
  }

  return success;
  */

//     sys::TimeValue now(0,0),user(0,0),delta(0,0),sys(0,0);
//     sys::Process::GetTimeUsage(now,user,sys);

//     if (simplifyExprs)
//       expr = constrs.simplifyExpr(expr);

//     bool success = solver->evaluate(Query(constrs, expr), result);

//     sys::Process::GetTimeUsage(delta,user,sys);
//     delta -= now;
//     stats::solverTime += delta.usec();
//     state.queryCost += delta.usec()/1000000.;

//     return success;

}


bool 
StringSolver::getInitialValues(const ExecutionState& state, 
                               const std::vector<const Array*>
			       & _objects,
                               std::vector< std::vector<unsigned char> >
			       & _result) {
//   std::cout << "getInitialValues \n";

//   for (unsigned i = 0; i < _objects.size(); i++) {
//     std::printf("object: %s \n", _objects[i]->name.c_str());
//   }
  
  if (_objects.empty())
    return true;
  
#ifdef FLA_DEBUG
  KLEE_INFO2 << "StringSolver: getInitialValues \n";
#endif
  
  klee::ref<Expr> expr = ConstantExpr::create(1, Expr::Bool);

  StrConstraint constr(state.constraints, stpSolver);
  klee::ref<Expr> len_exp = StrConstraint::makeLengthConstraint(constr, expr);

#ifdef FLA_DEBUG
  KLEE_INFO2 << "the length constraint: \n ";
  len_exp->dump();
  constr.dump();
#endif
  
  std::vector<const Array*> objects;
  for (std::set<const Array*>::iterator ii = constr.symlens.begin(); 
       ii !=  constr.symlens.end(); ii++) {
    objects.push_back(*ii);
  }

  std::vector< std::vector<unsigned char> > values;
  bool b = solveExpr(constr.len_constrs, len_exp, objects, values);
  
  if (b) {
#ifdef FLA_DEBUG
    KLEE_INFO2 << "The length constraint is satisfiable. \n";
    printAssignments(objects, values);
#endif
  }
  else {
#ifdef FLA_DEBUG
    KLEE_INFO2 << "The length constraint is NOT satisfiable! \n";
#endif
    return false;
  }

  if (!constr.involveStr)   // no string operations are involved
    return b;

  // set the lengths
  constr.concretizeLengths(objects, values);

  // build the string expression
  klee::ref<Expr> str_exp = StrConstraint::makeStrConstraint(constr, expr);

#ifdef FLA_DEBUG
  KLEE_INFO2 << "the constraint of the string expression: \n ";
  str_exp->dump();
  constr.dump();
#endif

  // solve the string expression
  objects.clear();
  values.clear();
  constr.getSymArrays(objects);
  
//   for (ConstraintManager::const_iterator ii = constr.assgn_constrs.begin(); 
//        ii != constr.assgn_constrs.end(); ii++) {
//     constr.len_constrs.addConstraint(*ii);
//   }
    
  bool success = solver->getInitialValues(Query(constr.len_constrs,
						ConstantExpr::alloc(0, Expr::Bool)), 
					  _objects, _result);

  if (success) {
#ifdef FLA_DEBUG
    KLEE_INFO << "The string expression is satisfiable. \n";
    printAssignments(_objects, _result);
#endif
  }
  else {
#ifdef FLA_DEBUG
    KLEE_INFO << "The string expression is NOT satisfiable! \n";
#endif
    return false;
  }
  
  return success;

//   sys::TimeValue now(0,0),user(0,0),delta(0,0),sys(0,0);
//   sys::Process::GetTimeUsage(now,user,sys);

//   bool success = solver->getInitialValues(Query(state.constraints,
//                                                 ConstantExpr::alloc(0, Expr::Bool)), 
//                                           objects, result);
  
//   sys::Process::GetTimeUsage(delta,user,sys);
//   delta -= now;
//   stats::solverTime += delta.usec();
//   state.queryCost += delta.usec()/1000000.;
  
//   return success;
}


bool StringSolver::getValue(const ExecutionState& state, klee::ref<Expr> expr, 
                            klee::ref<ConstantExpr> &result) {

  // Fast path, to avoid timer and OS overhead.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(expr)) {
    result = CE;
    return true;
  }

#ifdef FLA_DEBUG
  std::cout << "StringSolver:: getValue \n";
#endif

  StrConstraint constr(state.constraints, stpSolver);
  klee::ref<Expr> len_exp = StrConstraint::makeLengthConstraint(constr, expr);
  
  std::vector<const Array*> objects;
  for (std::set<const Array*>::iterator ii = constr.symlens.begin(); 
       ii !=  constr.symlens.end(); ii++) {
    objects.push_back(*ii);
  }

  std::vector< std::vector<unsigned char> > values;
  bool b = solveExpr(constr.len_constrs, len_exp, objects, values);
  
  if (b) {
#ifdef FLA_DEBUG
    KLEE_INFO2 << "The length constraint is satisfiable. \n";
    printAssignments(objects, values);
#endif
  }
  else {
#ifdef FLA_DEBUG
    KLEE_INFO2 << "The length constraint is NOT satisfiable! \n";
#endif
    return true;
  }

  if (!constr.involveStr)   // no string operations are involved
    return b;

  // set the lengths
  constr.concretizeLengths(objects, values);

  // build the string expression
  klee::ref<Expr> str_exp = StrConstraint::makeStrConstraint(constr, expr);

  // solve the string expression
  objects.clear();
  values.clear();
  constr.getSymArrays(objects);
  
  b = solver->getValue(Query(constr.len_constrs, expr), result);
  return b;


  return TimingSolver::getValue(state, expr, result);
  
//   sys::TimeValue now(0,0),user(0,0),delta(0,0),sys(0,0);
//   sys::Process::GetTimeUsage(now,user,sys);

//   if (simplifyExprs) {
//     expr = state.constraints.simplifyExpr(expr);
//   }

//   bool success = solver->getValue(Query(state.constraints, expr), result);

//   sys::Process::GetTimeUsage(delta,user,sys);
//   delta -= now;
//   stats::solverTime += delta.usec();
//   state.queryCost += delta.usec()/1000000.;

//   return success;
}


bool StringSolver::mustBeTrue(const ExecutionState& state, klee::ref<Expr> expr, 
                              bool &result) {
  std::cout << "StringSover:: mustBeTrue \n";
  return TimingSolver::mustBeTrue(state, expr, result);
}


bool StringSolver::mustBeFalse(const ExecutionState& state, klee::ref<Expr> expr, 
                              bool &result) {
  std::cout << "StringSover:: mustBeFalse \n";  
  return TimingSolver::mustBeFalse(state, expr, result);
}

bool StringSolver::mayBeTrue(const ExecutionState& state, klee::ref<Expr> expr, 
                              bool &result) {  
  return TimingSolver::mayBeTrue(state, expr, result);
}

bool StringSolver::mayBeFalse(const ExecutionState& state, klee::ref<Expr> expr, 
                              bool &result) {
  std::cout << "StringSover:: mayBeFalse \n";  
  return TimingSolver::mayBeFalse(state, expr, result);
}
