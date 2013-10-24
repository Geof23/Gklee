#ifndef KLEE_STRHANDLER_H
#define KLEE_STRHANDLER_H

#include "../Core/Executor.h"


/*********************************************************/


namespace klee {


class StringHandler {

  Executor* executor;

  typedef Executor::ExactResolutionList ExactResolutionList;

  // function pointer
  typedef bool (StringHandler::*Handler) (ExecutionState &state,
					  KInstruction *target,
					  std::vector<ref<Expr> >
					  &arguments);
  
  // map a function name to the handler
  typedef std::map<std::string, StringHandler::Handler> FunctionMap;

public:

  bool enabled;
  
  FunctionMap functionMap;

  StringHandler(Executor* _executor) : executor(_executor) {
    enabled = true;

#define addFunc(name, handler) functionMap[name] = &StringHandler::handler

    addFunc("FLA_klee_make_symbolic_length", handleMakeSymbolicLength);
    addFunc("_ZNKSs6lengthEv", handleStringLength);
    addFunc("_ZSteqIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_", handleStringEqual);
    addFunc("_ZStneIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_", handleStringNotEqual);
    addFunc("_ZNKSs12find_last_ofEcj", handleStringFindLastOf);
    addFunc("_ZNKSs6substrEjj", handleStringSubStr);
    addFunc("_ZNKSs4findERKSsj", handleStringFind);
    addFunc("_ZNKSs7compareEjjRKSsjj", handleStringCompare);
    addFunc("_ZNSsD1Ev", handleStringDeallocate);
    addFunc("_ZNSsaSERKSs", handleStringAssign);

#undef addFunc

  };

public:

  // invoke the handler according to the function names
  bool dispatchStringFunction(ExecutionState &state,
			      KInstruction *ki,
			      llvm::Function *f,
			      std::vector< ref<Expr> > &arguments);

  // print out the assignment (object -> value) if a solution is found
  static void printAssignments( std::vector<const Array*>& objects, 
				std::vector< std::vector<unsigned char> >& values );
  
private:


  // call the solver to check the expression in a constraint environment

  bool solveExpr(ref<Expr> expr, std::vector<const Array*>& objects, 
		 std::vector< std::vector<unsigned char> >& values);


  bool solveExpr(std::vector< ref<Expr> > &constr, ref<Expr> expr, 
		 std::vector<const Array*>& objects, 
		 std::vector< std::vector<unsigned char> >& values);


  bool solveExpr(std::vector< ref<Expr> > &constr, ref<Expr> expr) {
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


  // enforce the virtual machine to write the value into the memory

  void executeMemoryOperation(ExecutionState &state,
			      bool isWrite,
			      ref<Expr> address,
			      ref<Expr> value /* undef if read */,
			      KInstruction *target /* undef if write */);

private:

  // read a concrete string from the memory
  std::string readStringAtAddress(ExecutionState &state, 
				  ref<Expr> addressExpr);

  // read a concrete integer from the memory
  ref<Expr> readIntAtAddress(ExecutionState &state, 
			     ref<Expr> addressExpr, Expr::Width width = Expr::Int32);

  // read a string expression (either concrete or symbolic) from the memory
  ref<Expr> obtainStr(ExecutionState &state,
		      ref<Expr> addr);

  // FLA's implementation of the "makeSymbolic" function
  // The length can be within a range rather than be fixed
  bool handleMakeSymbolicLength(ExecutionState &state,
				       KInstruction *target,
				       std::vector<ref<Expr> > &arguments);

  // build a "length" expression
  bool handleStringLength(ExecutionState &state,
			  KInstruction *target,
			  std::vector<ref<Expr> > &arguments);

  // build an "equal" expression
  bool handleStringEqual(ExecutionState &state,
			 KInstruction *target,
			 std::vector<ref<Expr> > &arguments);

  // build a "not equal" expression
  // to be added
  bool handleStringNotEqual(ExecutionState &state,
			    KInstruction *target,
			    std::vector<ref<Expr> > &arguments);

  // build a "FindLastOf" expression
  bool handleStringFindLastOf(ExecutionState &state,
			      KInstruction *target,
			      std::vector<ref<Expr> > &arguments);

  // build a "Substr" expression
  bool handleStringSubStr(ExecutionState &state,
			  KInstruction *target,
			  std::vector<ref<Expr> > &arguments);

  // build a "Find" expression
  bool handleStringFind(ExecutionState &state,
			KInstruction *target,
			std::vector<ref<Expr> > &arguments);

  // build a "Compare" expression
  bool handleStringCompare(ExecutionState &state,
			   KInstruction *target,
			   std::vector<ref<Expr> > &arguments);

  // deallocate a string
  bool handleStringDeallocate(ExecutionState &state,
			      KInstruction *target,
			      std::vector<ref<Expr> > &arguments);

  // handle string assignment
  bool handleStringAssign(ExecutionState &state,
			  KInstruction *target,
			  std::vector<ref<Expr> > &arguments);
  
};  // end class

 
/***********************************************************************/

} // end namespace klee

#endif
