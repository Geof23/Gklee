
#include "StrExpr.h"
#include "StringHandler.h"

// #include "Executor.h" 
#include "../Core/Memory.h"
#include "StringSolver.h"
#include "llvm/Function.h"
#include "llvm/Support/raw_ostream.h"

namespace klee {


/*******************************************************************
 Implementation of the declared methods
 Auxiliary functions
 ******************************************************************/

 void StringHandler::
   printAssignments( std::vector<const Array*>& objects, 
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


 bool StringHandler::solveExpr(ref<Expr> expr, std::vector<const Array*>& objects, 
			       std::vector< std::vector<unsigned char> >& values) {

   ConstraintManager constraints;
   constraints.addConstraint(expr);
   Query qr(constraints, expr);
   
   bool success = executor->solver->stpSolver->
     getInitialObjectsValues(qr, objects, values);
   
   return success;
 }
 

bool StringHandler::solveExpr( std::vector< ref<Expr> > &constr, ref<Expr> expr, 
			       std::vector<const Array*>& objects, 
			       std::vector< std::vector<unsigned char> >& values) {

    ConstraintManager constraints;
    for (std::vector< ref<Expr> >::iterator ii = constr.begin(); ii != constr.end(); ii++) {
/*       ref<Expr> e = constraints.simplifyExpr(*ii); */
/*       if (e->getKind() == Expr::Constant) { */
/* 	ref<ConstantExpr> e1 = cast<ConstantExpr>(e); */
/* 	if (e1->isTrue()) */
/* 	  continue; */
/* 	else if (e1->isFalse()) */
/* 	  return false;      */
/*       } */
      if (!(constraints.addConstraint(*ii)))
	return false;
    }

/*     expr = constraints.simplifyExpr(expr); */
/*     if (expr->getKind() == Expr::Constant) { */
/*       ref<ConstantExpr> e1 = cast<ConstantExpr>(expr); */
/*       if (e1->isTrue()) */
/* 	; */
/*       else if (e1->isFalse()) */
/* 	return false; */
/*       else */
/* 	constraints.addConstraint(expr); */
/*     } */
/*     else { */
/*       expr->dump(); */
/*       constraints.addConstraint(expr); */
/*     } */

    if (!(constraints.addConstraint(expr)))
      return false;

    // Query qr(constraints, ConstantExpr::alloc(0, Expr::Bool));
    Query qr(constraints, expr);
   
    bool success = executor->solver->stpSolver->
      getInitialObjectsValues(qr, objects, values);

/*     KLEE_INFO << "satisfiable = " << success << "\n"; */

/*     if (success) { */
/*       KLEE_INFO << "The length constraint is satisfiable! \n"; */
/*       printf("#symbolic values = %d\n", values.size()); */
/*       for (unsigned i = 0; i <values[0].size(); i++) */
/* 	printf("val = %d\n", values[0][i]); */
/*     } */
/*     else  */
/*       KLEE_INFO << "The length constraint is NOT satisfiable! \n"; */

    return success;
  }


ref<Expr> StringHandler::obtainStr(ExecutionState &state,
				   ref<Expr> addr) {

  if (isa<StrExpr>(addr)) {  // a string expression binded locally
#ifdef FLA_DEBUG
    printf("Found a direct string expression:");
    addr->dump();
#endif
    return addr;
  }

  ObjectPair op;
  bool success = state.addressSpace.resolveOne(cast<ConstantExpr>(addr), op);
  if (success) {
    const ObjectState *os = op.second;
    const Array* arr = os->getUpdates().root;
    if (arr != NULL && isa<VarLenArray>(arr)) {   // var-len array
      return new StrDataExpr((VarLenArray*) arr);
    }
    else {   // string with concrete values
      // here we need to go into the structure of the string class
      // and read the chars out

      ref<Expr> offset = op.first->getOffsetExpr(addr);

      // read the 32-bit pointer pointing to the string object
      ref<Expr> data_addr = os->read(offset, Expr::Int32);
      
      // if the string is a (symbolic) string expression
      if (isa<StrExpr>(data_addr)) {
	// printf("Found a reference to a string expression:");
	return data_addr;
      }
 
      // otherwise the string has concrete value (i.e. chars)

      // read the length
      ref<Expr> len_addr = 
	AddExpr::create(offset,
			ConstantExpr::create(8, Expr::Int32));
      ref<Expr> len_exp = os->read(len_addr, Expr::Int32);
      unsigned len = cast<ConstantExpr>(len_exp)->getZExtValue(32);
      // printf("len = %d\n", len);
       
/*       // read the 32-bit pointer pointing to the string object */
/*       ref<Expr> carr_addr = os->read(offset, Expr::Int32); */

      // now read the chars
      success = state.addressSpace.resolveOne(cast<ConstantExpr>(data_addr), op);
      if (success) {
	// const MemoryObject *ms = op.first;
	const ObjectState *os = op.second;
	ref<Expr> offset = op.first->getOffsetExpr(data_addr);
	 
	char* buf = new char[len+1];
	for (unsigned i = 0; i < len; i++) {
	  ref<Expr> cur = os->read8(i);
	  cur = executor->toUnique(state, cur);
	  assert(isa<ConstantExpr>(cur) && 
		 "hit symbolic char while reading concrete string");
	  buf[i] = cast<ConstantExpr>(cur)->getZExtValue(8);
	  // printf("c = %c\n", buf[i]);
	}
	 
	buf[len] = '\0';
	return new StrDataExpr(buf);
      }

      else {  // the reference points to unknown data, return it anyway 
	printf("Fail to obtain a string data expression from the memory!\n");
	return data_addr;
      }
    }
  }
  assert(0 && "Fail to obtain a string data expression from the memory!\n");
 }


// reads a unsigned integer from memory
ref<Expr> 
StringHandler::readIntAtAddress(ExecutionState &state, 
				ref<Expr> addressExpr,
				Expr::Width width) {
  ObjectPair op;
  addressExpr = executor->toUnique(state, addressExpr);
  ref<ConstantExpr> address = cast<ConstantExpr>(addressExpr);
/*   std::cout << "The address: "; */
/*   address->dump(); */
  if (!state.addressSpace.resolveOne(address, op))
    assert(0 && "XXX out of bounds / multiple resolution unhandled");
  bool res;
  assert(executor->solver->mustBeTrue(state, 
				      EqExpr::create(address, 
						     op.first->getBaseExpr()),
				      res) &&
         res &&
         "XXX interior pointer unhandled");
/*   const MemoryObject *mo = op.first; */
  const ObjectState *os = op.second;

  ref<Expr> val = os->read(ConstantExpr::create(0, Expr::Int32), 
			   width);
  return val;
 }

// reads a concrete string from memory
std::string 
StringHandler::readStringAtAddress(ExecutionState &state, 
				   ref<Expr> addressExpr) {
  ObjectPair op;
  addressExpr = executor->toUnique(state, addressExpr);
  ref<ConstantExpr> address = cast<ConstantExpr>(addressExpr);
  if (!state.addressSpace.resolveOne(address, op))
    assert(0 && "XXX out of bounds / multiple resolution unhandled");
  bool res;
  assert(executor->solver->mustBeTrue(state, 
				      EqExpr::create(address, 
						     op.first->getBaseExpr()),
				      res) &&
         res &&
         "XXX interior pointer unhandled");
  const MemoryObject *mo = op.first;
  const ObjectState *os = op.second;

  char *buf = new char[mo->size];

  unsigned i;
  for (i = 0; i < mo->size - 1; i++) {
    ref<Expr> cur = os->read8(i);
    cur = executor->toUnique(state, cur);
    assert(isa<ConstantExpr>(cur) && 
           "hit symbolic char while reading concrete string");
    buf[i] = cast<ConstantExpr>(cur)->getZExtValue(8);
  }
  buf[i] = 0;
  
  std::string result(buf);
  delete[] buf;
  return result;
}


// Access (e.g. read or write) the memory in a more advanced manner
 void StringHandler::
   executeMemoryOperation(ExecutionState &state,
			  bool isWrite,
			  ref<Expr> address,
			  ref<Expr> value /* undef if read */,
			  KInstruction *target /* undef if write */) {
   Expr::Width type = (isWrite ? value->getWidth() : 
		       Expr::getWidthForLLVMType(target->inst->getType()));
   unsigned bytes = Expr::getMinBytesForWidth(type);

/*     if (executor->SimplifySymIndices) { */
/*       if (!isa<ConstantExpr>(address)) */
/* 	address = state.constraints.simplifyExpr(address); */
/*       if (isWrite && !isa<ConstantExpr>(value)) */
/* 	value = state.constraints.simplifyExpr(value); */
/*     } */

   // fast path: single in-bounds resolution
   ObjectPair op;
   bool success;
   if (!state.addressSpace.resolveOne(state, executor->solver, address, op, success)) {
     address = executor->toConstant(state, address, "resolveOne failure");
     success = state.addressSpace.resolveOne(cast<ConstantExpr>(address), op);
   }

   if (success) {
     const MemoryObject *mo = op.first;
           
     ref<Expr> offset = mo->getOffsetExpr(address);
      
     bool inBounds;
     bool success = 
       executor->solver->mustBeTrue(state, 
				    mo->getBoundsCheckOffset(offset, bytes),
				    inBounds);
     if (!success) {
       state.setPC(state.getPrevPC());
       assert(0 && "query timed out");
       return;
     }

     if (inBounds) {
       const ObjectState *os = op.second;
       if (isWrite) {
	 if (os->readOnly) {
	   assert( 0 && "memory error: object read only");
	 } else {
	   ObjectState *wos = state.addressSpace.getWriteable(mo, os);
	   wos->write(offset, value);
	 }          
       } else {
	 ref<Expr> result = os->read(offset, type);  	  
	 executor->bindLocal(target, state, result);
       }
       
       return;
     }
   } 
   
 }
 


/*******************************************************************
 Auxiliary Functions
 ******************************************************************/

void registerExpr(ExecutionState &state, 
		  ref<StrExpr> str_exp,
		  ref<Expr> res, 
		  const Array *array) { 
  // record the relation
  ref<Expr> alias_exp = StrAliasExpr::create(res, str_exp);
  state.addConstraint(alias_exp);

#ifdef FLA_DEBUG
  std::cout << "The string expression: \n";
  alias_exp->dump();
#endif  

  // link the alias
  str_exp->array_alias = (Array*) array;
  str_exp->alias = res.get();
}


/*******************************************************************
 Implementation of the declared methods
 Handling functions of string operations
 ******************************************************************/

bool StringHandler::dispatchStringFunction(ExecutionState &state,
					   KInstruction *ki,
					   llvm::Function *f,
					   std::vector< ref<Expr> > &arguments) {

  // the string handler/solver is enabled?
  if (!enabled)
    return false;
  
  FunctionMap::iterator ii = functionMap.find(f->getName());
  if (ii != functionMap.end()) {

#ifdef FLA_DEBUG
    KLEE_INFO2 << "Found: " << f->getName() << "\n";
#else  
    if (f->getName() != "_ZNSsD1Ev")
      KLEE_INFO2 << "Found: " << f->getName() << "\n";
#endif

    StringHandler::Handler h = ii->second;
    return (this->*h)(state, ki, arguments);
  }

//   else if (f->getName() == "printf") {
//     KLEE_INFO << "Found: " << f->getName() << "\n";
    
//     ref<Expr> v = arguments[1];
//     if (arguments.size() > 1 && v->isSymbolic()) {
//       /*     v = toUnique(state, arguments[1]); */
//       /*     if (ConstantExpr *ce = dyn_cast<ConstantExpr>(v)) { */
//       /*       // non-symbolic variable */
//       /*       return false; */
//       /*     } */
//       v->dump();
//       KLEE_INFO << "Finish printing! \n";
//       return true;
//     }
//     return false;
//   }

  else if (f->getName() == "FLA_klee_make_ite") {
    KLEE_INFO << "Found: " << f->getName() << "\n";
    
    ref<Expr> cond = ZExtExpr::create(arguments[0], Expr::Bool);
    ref<Expr> ite_exp = SelectExpr::create(cond, arguments[1], arguments[2]);
    executor->bindLocal(ki, state, ite_exp);
    
    return true;
  }

  else if (f->getName() == "FLA_klee_print_expr") {
    KLEE_INFO << "Found: " << f->getName() << "\n";
    arguments[0]->dump();

    return true;
  }

  else if (f->getName() == "FLA_klee_get_max_value") {
    KLEE_INFO << "Found: " << f->getName() << "\n";
    ref<Expr> max = (executor->solver->getRange(state, arguments[0])).second;
    // executor->bindLocal(ki, state, cast<ConstantExpr>(max));
    executor->bindLocal(ki, state, max);
    return true;
  }
  
  return false;
 }


bool StringHandler::handleMakeSymbolicLength(ExecutionState &state,
					     KInstruction *target,
					     std::vector<ref<Expr> > &arguments) {
  std::string name;

  if (arguments.size() == 3) {
    name = "unnamed";
  } else {
    // FIXME: Should be a user.err, not an assert.
    assert(arguments.size()==4 &&
           "invalid number of arguments to klee_make_symbolic_length");
    name = readStringAtAddress(state, arguments[3]);
  }

  ExactResolutionList rl;
  executor->resolveExact(state, arguments[0], rl, "make_symbolic_length");
  
  for (ExactResolutionList::iterator it = rl.begin(),
         ie = rl.end(); it != ie; ++it) {
    const MemoryObject *mo = it->first.first;
    mo->setName(name);
    
    const ObjectState *old = it->first.second;
    ExecutionState *s = it->second;
    
    if (old->readOnly) {
      executor->terminateStateOnError(*s,
				      "cannot make readonly object symbolic",
				      "user.err");
      return true;
    }
    
    // read the min and max value

    ref<Expr> min_exp = arguments[1];
    ref<Expr> max_exp = arguments[2];

    unsigned min = cast<ConstantExpr>(min_exp)->getAPValue().getZExtValue();
    // std::cerr << "min = " << min << std::endl;
    unsigned max = cast<ConstantExpr>(max_exp)->getAPValue().getZExtValue();
    // std::cerr << "max = " << max << std::endl;

    unsigned max_len = 255;
    if (max > max_len) {
      std::cout << "Truncate the string size to be " << max_len << std::endl;
      max = max_len;
      min = min > max_len ? max_len : min; 
    }

    // executeMemoryOperation(state, true, base, value, 0);
    // old->write(offset, value);

    static unsigned str_id = 0;
    VarLenArray *array = new VarLenArray("str" + llvm::utostr(++str_id),
					 min, max);

    executor->bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);

    std::cout << "KLEE++: make symbolic " << name << 
      " with minLen = " << min << " and maxLen = " << max << std::endl;

  } //end for

  return true;
}


 bool StringHandler::handleStringLength(ExecutionState &state,
					KInstruction *target,
					std::vector<ref<Expr> > &arguments) {

   ref<StrDataExpr> str = cast<StrDataExpr>(obtainStr(state, arguments[0]));

   ref<Expr> str_len = new StrLenExpr(str);
#ifdef FLA_DEBUG
   std::cout << "The string expression: \n";
   str_len->dump();
#endif

   // write the result to the state
   executor->bindLocal(target, state, str_len);
   return true;

 }


 bool StringHandler::handleStringFindLastOf(ExecutionState &state,
					    KInstruction *target,
					    std::vector<ref<Expr> > &arguments) {

   // the first argument is a single string (i.e. not a string expression)
   ref<Expr> str1 = obtainStr(state, arguments[0]);
   if (!(isa<StrDataExpr>(str1))) {
     printf("Fail to obtain a string!");
     return false;
   }

   // the second argument is a character
   ref<Expr> str2 = arguments[1];
   
   ref<StrFindLastOfExpr> str_exp = new StrFindLastOfExpr(str1, str2);
   
   // create an alias for the position 
   const Array *array = new Array("#l" + llvm::utostr(++StrExpr::len_id), 1);
   ref<Expr> pos =  StrExpr::createLengthExpr(array);

   // record the relation
   registerExpr(state, str_exp, pos, array);

   // write the result to the state
   executor->bindLocal(target, state, pos);

   return true;
 }
 

 bool StringHandler::handleStringSubStr(ExecutionState &state,
					KInstruction *target,
					std::vector<ref<Expr> > &arguments) {

   ref<Expr> str = obtainStr(state, arguments[1]);
   if (!(isa<StrDataExpr>(str))) {
     printf("Fail to obtain a string!");
     return false;
   }
   ref<StrDataExpr> str_data = cast<StrDataExpr>(str);

   ref<Expr> pos = arguments[2];
   ref<Expr> len = arguments[3];
   ref<StrSubStrExpr> str_exp = new StrSubStrExpr(str, pos, len);

   // create an alias for the position 
   VarLenArray *array = 
     new VarLenArray("#s" + llvm::utostr(++StrExpr::str_id), 
		     str_data->array->min, 
		     str_data->array->max, true);
   ref<StrDataExpr> new_str = new StrDataExpr(array);

   // record the relation
   registerExpr(state, str_exp, new_str, array);
   
   // the location of the destination string 
   ref<Expr> dst = arguments[0];
   executeMemoryOperation(state, true, dst, new_str, 0);

   return true;
 }


 bool StringHandler::handleStringFind(ExecutionState &state,
				      KInstruction *target,
				      std::vector<ref<Expr> > &arguments) {

   ref<Expr> str = obtainStr(state, arguments[0]);
   if (!(isa<StrDataExpr>(str))) {
     printf("Find: Fail to obtain a string expression!");
     return false;
   }

   ref<Expr> tofind = obtainStr(state, arguments[1]);
   if (!(isa<StrDataExpr>(tofind))) {
     printf("Find: Fail to obtain a string expression!");
     return false;
   }
   
   ref<StrFindExpr> str_exp = new StrFindExpr(str, tofind);

   // create an alias for the position 
   const Array *array = new Array("#l" + llvm::utostr(++StrExpr::len_id), 1);
   ref<Expr> pos = StrExpr::createLengthExpr(array);

   // record the relation
   registerExpr(state, str_exp, pos, array);

   // write the result to the state
   executor->bindLocal(target, state, pos);

   return true;
 }


 bool StringHandler::handleStringEqual(ExecutionState &state,
				       KInstruction *target,
				       std::vector<ref<Expr> > &arguments) {
#ifdef FLA_DEBUG
   KLEE_INFO2 << "handleStringEqual!";
#endif

   ref<Expr> str1 = obtainStr(state, arguments[0]);
   if (!(isa<StrDataExpr>(str1))) {
     printf("StringEqual: Fail to obtain a string!");
     return false;
   }

   ref<Expr> str2 = obtainStr(state, arguments[1]);
   if (!(isa<StrDataExpr>(str2))) {
     printf("StringEqual: Fail to obtain a string!");
     return false;
   }

   ref<StrEqExpr> str_exp = new StrEqExpr(str1, str2);
   
   // create an alias for the position 
   const Array *array = new Array("#l" + llvm::utostr(++StrExpr::len_id), 1);
   ref<Expr> res =  StrExpr::createValueRead(array, Expr::Int8);

   // record the relation
   registerExpr(state, str_exp, res, array);

   // write the result to the state
   executor->bindLocal(target, state, res);

   return true;
 }



 bool StringHandler::handleStringCompare(ExecutionState &state,
					 KInstruction *target,
					 std::vector<ref<Expr> > &arguments) {
   ref<Expr> str = obtainStr(state, arguments[0]);
   if (!(isa<StrDataExpr>(str))) {
     printf("StringCompare: Fail to obtain a string!");
     return false;
   }
   ref<Expr> pos = arguments[1];
   ref<Expr> len = arguments[2];

   ref<Expr> tocompare = obtainStr(state, arguments[3]);
   if (!(isa<StrDataExpr>(tocompare))) {
     printf("StringCompare: Fail to obtain a string!");
     return false;
   }

   ref<StrCompareExpr> str_exp = new StrCompareExpr(str, pos, len, tocompare);
   
   // create an alias for the position 
   const Array *array = new Array("#l" + llvm::utostr(++StrExpr::len_id), 1);
   ref<Expr> res = StrExpr::createLengthExpr(array);

//    const Array *array = new Array("#l" + llvm::utostr(++StrExpr::len_id), 4);
//    ref<Expr> res = StrExpr::createValueRead(array, Expr::Int32);

   // record the relation
   registerExpr(state, str_exp, res, array);

   // write the result to the state
   executor->bindLocal(target, state, res);

   return true;
 }


  bool StringHandler::handleStringDeallocate(ExecutionState &state,
					     KInstruction *target,
					     std::vector<ref<Expr> > &arguments) {

    ref<Expr> str = obtainStr(state, arguments[0]);
    if (!(isa<StrDataExpr>(str))) {
      printf("StringDeallocate: Fail to obtain a string!");
      return false;
    }

    ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);
    if (str_exp->isSymbolic()) {
      return true;
    }
    else // call the library implementation to deallocate the memory
      return false;
  }


  bool StringHandler::handleStringAssign(ExecutionState &state,
					 KInstruction *target,
					 std::vector<ref<Expr> > &arguments) {
    ref<Expr> str = obtainStr(state, arguments[1]);
    if (!(isa<StrDataExpr>(str))) {
      printf("StringAssign: Fail to obtain a string!");
      return false;
    }

   // the location of the destination string 
   ref<Expr> dst = arguments[0];
   executeMemoryOperation(state, true, dst, str, 0);
   return true;
  }


  // to be implemented
  bool StringHandler::handleStringNotEqual(ExecutionState &state,
					   KInstruction *target,
					   std::vector<ref<Expr> > &arguments) {
    return true;
  }


/*******************************************************************
 Other functions
 ******************************************************************/

 
/***********************************************************************/

} // end namespace klee
