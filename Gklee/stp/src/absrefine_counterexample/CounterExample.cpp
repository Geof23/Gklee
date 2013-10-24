// -*- c++ -*-
/********************************************************************
 * AUTHORS: Vijay Ganesh
 *
 * BEGIN DATE: November, 2005
 *
 * LICENSE: Please view LICENSE file in the home dir of this Program
 ********************************************************************/

#include "../sat/sat.h"
#include "AbsRefine_CounterExample.h"
#include "../printer/printers.h"
#include "../to-sat/AIG/ToSATAIG.h"

const bool debug_counterexample =  false;

namespace BEEV
{

  /*FUNCTION: constructs counterexample from MINISAT counterexample
   * step1 : iterate through MINISAT counterexample and assemble the
   * bits for each AST term. Store it in a map from ASTNode to vector
   * of bools (bits).
   *
   * step2: Iterate over the map from ASTNodes->Vector-of-Bools and
   * populate the CounterExampleMap data structure (ASTNode -> BVConst)
   */
  void
  AbsRefine_CounterExample::ConstructCounterExample(MINISAT::Solver& newS,
      ToSATBase::ASTNodeToSATVar& satVarToSymbol)
  {
    //iterate over MINISAT counterexample and construct a map from AST
    //terms to vector of bools. We need this iteration step because
    //MINISAT might return the various bits of a term out of
    //order. Therfore, we need to collect all the bits and assemble
    //them properly

    if (!newS.okay())
      return;
    if (!bm->UserFlags.construct_counterexample_flag)
      return;

    assert(CounterExampleMap.size() == 0);

    CopySolverMap_To_CounterExample();

    for (ToSATBase::ASTNodeToSATVar::const_iterator it = satVarToSymbol.begin(); it
        != satVarToSymbol.end(); it++)
      {
        const ASTNode& symbol = it->first;
        const vector<unsigned>& v = it->second;

        for (int index = 0; index < v.size(); index++)
          {
            const unsigned sat_variable_index = v[index];

            if (sat_variable_index == ~((unsigned) 0)) // not sent to the sat solver.
              continue;

            if (newS.model[sat_variable_index] == MINISAT::l_Undef)
              continue;

            //assemble the counterexample here
            if (symbol.GetType() == BITVECTOR_TYPE)
              {
                const unsigned int symbolWidth = symbol.GetValueWidth();

                //'v' is the map from bit-index to bit-value
                HASHMAP<unsigned, bool> * v;
                if (_ASTNode_to_BitvectorMap.find(symbol)
                    == _ASTNode_to_BitvectorMap.end())
                  {
                    _ASTNode_to_BitvectorMap[symbol] = new HASHMAP<unsigned,
                        bool> (symbolWidth);
                  }

                //v holds the map from bit-index to bit-value
                v = _ASTNode_to_BitvectorMap[symbol];

                //Collect the bits of 'symbol' and store in v. Store
                //in reverse order.
                if (newS.model[sat_variable_index] == MINISAT::l_True)
                  (*v)[(symbolWidth - 1) - index] = true;
                else
                  (*v)[(symbolWidth - 1) - index] = false;
              }
            else
              {
                assert (symbol.GetType() == BOOLEAN_TYPE);
                const char * zz = symbol.GetName();
                //if the variables are not cnf variables then add
                //them to the counterexample
                if (0 != strncmp("cnf", zz, 3) && 0
                    != strcmp("*TrueDummy*", zz))
                  {
                    if (newS.model[sat_variable_index] == MINISAT::l_True)
                      CounterExampleMap[symbol] = ASTTrue;
                    else if (newS.model[sat_variable_index] == MINISAT::l_False)
                      CounterExampleMap[symbol] = ASTFalse;
                    else
                      FatalError("never heres.");
                  }
              }
          }
      }

    //iterate over the ASTNode_to_Bitvector data-struct and construct
    //the the aggregate value of the bitvector, and populate the
    //CounterExampleMap datastructure
    for (ASTtoBitvectorMap::iterator it = _ASTNode_to_BitvectorMap.begin(),
           itend = _ASTNode_to_BitvectorMap.end(); it != itend; it++)
      {
        ASTNode var = it->first;
        //debugging
        //cerr << var;
        if (SYMBOL != var.GetKind())
          {
            FatalError("ConstructCounterExample:"
              "error while constructing counterexample: "
                       "not a variable: ", var);
          }
        //construct the bitvector value
        HASHMAP<unsigned, bool> * w = it->second;
        ASTNode value = BoolVectoBVConst(w, var.GetValueWidth());
        //debugging
        //cerr << value;

        //populate the counterexample datastructure. add only scalars
        //variables which were declared in the input and newly
        //introduced variables for array reads
        CounterExampleMap[var] = value;
      }

    //In this loop, we compute the value of each array read, the
    //corresponding ITE against the counterexample generated above.
    for (ASTNodeMap::const_iterator it =
        ArrayTransform->ArrayRead_IteMap()->begin(), itend =
        ArrayTransform->ArrayRead_IteMap()->end(); it != itend; it++)
      {
        //the array read
        ASTNode arrayread = it->first;
        ASTNode value_ite = ArrayTransform->ArrayRead_Ite(arrayread);

        //convert it to a constant array-read and store it in the
        //counter-example. First convert the index into a constant. then
        //construct the appropriate array-read and store it in the
        //counterexample
        ASTNode arrayread_index = TermToConstTermUsingModel(arrayread[1]);
        ASTNode key = bm->CreateTerm(READ, arrayread.GetValueWidth(),
                                     arrayread[0], arrayread_index);

        //Get the ITE corresponding to the array-read and convert it
        //to a constant against the model
        ASTNode value = TermToConstTermUsingModel(value_ite);
        //save the result in the counter_example
        if (!simp->CheckSubstitutionMap(key))
          CounterExampleMap[key] = value;
      }
  } //End of ConstructCounterExample


  // FUNCTION: accepts a non-constant term, and returns the
  // corresponding constant term with respect to a model.
  //
  // term READ(A,i) is treated as follows:
  //
  //1. If (the boolean variable 'ArrayReadFlag' is true && ArrayRead
  //1. has value in counterexample), then return the value of the
  //1. arrayread.
  //
  //2. If (the boolean variable 'ArrayReadFlag' is true && ArrayRead
  //2. doesn't have value in counterexample), then return the
  //2. arrayread itself (normalized such that arrayread has a constant
  //2. index)
  //
  //3. If (the boolean variable 'ArrayReadFlag' is false) && ArrayRead
  //3. has a value in the counterexample then return the value of the
  //3. arrayread.
  //
  //4. If (the boolean variable 'ArrayReadFlag' is false) && ArrayRead
  //4. doesn't have a value in the counterexample then return 0 as the
  //4. value of the arrayread.
  ASTNode
  AbsRefine_CounterExample::TermToConstTermUsingModel(const ASTNode& t,
      bool ArrayReadFlag)
  {
    bm->Begin_RemoveWrites = false;
    bm->SimplifyWrites_InPlace_Flag = false;
    //ASTNode term = SimplifyTerm(t);
    ASTNode term = t;
    Kind k = term.GetKind();

    //cerr << "Input to TermToConstTermUsingModel: " << term << endl;
    if (!is_Term_kind(k))
      {
        FatalError("TermToConstTermUsingModel: "
          "The input is not a term: ", term);
      }
    if (k == WRITE)
      {
        FatalError("TermToConstTermUsingModel: "
          "The input has wrong kind: WRITE : ", term);
      }
    if (k == SYMBOL && BOOLEAN_TYPE == term.GetType())
      {
        FatalError("TermToConstTermUsingModel: "
          "The input has wrong kind: Propositional variable : ", term);
      }

    ASTNodeMap::iterator it1;
    if ((it1 = CounterExampleMap.find(term)) != CounterExampleMap.end())
      {
        ASTNode val = it1->second;
        if (BVCONST != val.GetKind())
          {
            //CounterExampleMap has two maps rolled into
            //one. SubstitutionMap and SolverMap.
            //
            //recursion is fine here. There are two maps that are checked
            //here. One is the substitutionmap. We garuntee that the value
            //of a key in the substitutionmap is always a constant.
            //
            //in the SolverMap we garuntee that "term" does not occur in
            //the value part of the map
            if (term == val)
              {
                FatalError("TermToConstTermUsingModel: "
                           "The input term is stored as-is "
                  "in the CounterExample: Not ok: ", term);
              }
            return TermToConstTermUsingModel(val, ArrayReadFlag);
          }
        else
          {
            return val;
          }
      }

    ASTNode output;
    switch (k)
      {
      case BVCONST:
        output = term;
        break;
      case SYMBOL:
        {
          if (term.GetType() == ARRAY_TYPE)
            {
              return term;
            }

          // Has been simplified out. Can take any value.
          output = bm->CreateZeroConst(term.GetValueWidth());
          break;
        }
      case READ:
        {
          ASTNode arrName = term[0];
          ASTNode index = term[1];
          if (0 == arrName.GetIndexWidth())
            {
            FatalError("TermToConstTermUsingModel: "
                         "array has 0 index width: ", arrName);
            }

          if (WRITE == arrName.GetKind()) //READ over a WRITE
            {
            ASTNode wrtterm = Expand_ReadOverWrite_UsingModel(term,
                ArrayReadFlag);
              if (wrtterm == term)
                {
                FatalError("TermToConstTermUsingModel: "
                  "Read_Over_Write term must be expanded "
                             "into an ITE", term);
                }
            ASTNode rtterm = TermToConstTermUsingModel(wrtterm, ArrayReadFlag);
              assert(ArrayReadFlag || (BVCONST == rtterm.GetKind()));
              return rtterm;
            }
          else if (ITE == arrName.GetKind()) //READ over an ITE
            {
              // The "then" and "else" branch are arrays.
            ASTNode indexVal = TermToConstTermUsingModel(index, ArrayReadFlag);

            ASTNode condcompute = ComputeFormulaUsingModel(arrName[0]); // Get the truth value.
              unsigned int wid = arrName.GetValueWidth();
              if (ASTTrue == condcompute)
                {
                const ASTNode & result = TermToConstTermUsingModel(
                    bm->CreateTerm(READ, wid, arrName[1], indexVal),
                                              ArrayReadFlag);
                  assert(ArrayReadFlag || (BVCONST == result.GetKind()));
                  return result;
                }
              else if (ASTFalse == condcompute)
                {
                const ASTNode & result = TermToConstTermUsingModel(
                    bm->CreateTerm(READ, wid, arrName[2], indexVal),
                                              ArrayReadFlag);
                  assert(ArrayReadFlag || (BVCONST == result.GetKind()));
                  return result;
                }
              else
                {
                cerr << "TermToConstTermUsingModel: termITE: "
                    "value of conditional is wrong: " << condcompute << endl;
                FatalError(" TermToConstTermUsingModel: termITE: "
                  "cannot compute ITE conditional against model: ", term);
                }
              FatalError("bn23143 Never Here");
            }

          ASTNode modelentry;
          if (CounterExampleMap.find(index) != CounterExampleMap.end())
            {
              //index has a const value in the CounterExampleMap
              //ASTNode indexVal = CounterExampleMap[index];
            ASTNode indexVal = TermToConstTermUsingModel(
                CounterExampleMap[index], ArrayReadFlag);
            modelentry = bm->CreateTerm(READ, arrName.GetValueWidth(), arrName,
                indexVal);
            }
          else
            {
              //index does not have a const value in the
              //CounterExampleMap. compute it.
            ASTNode indexconstval = TermToConstTermUsingModel(index,
                ArrayReadFlag);
              //update model with value of the index
              //CounterExampleMap[index] = indexconstval;
            modelentry = bm->CreateTerm(READ, arrName.GetValueWidth(), arrName,
                indexconstval);
            }
          //modelentry is now an arrayread over a constant index
          BVTypeCheck(modelentry);

          //if a value exists in the CounterExampleMap then return it
          if (CounterExampleMap.find(modelentry) != CounterExampleMap.end())
            {
            output = TermToConstTermUsingModel(CounterExampleMap[modelentry],
                                          ArrayReadFlag);
            }
          else if (ArrayReadFlag)
            {
              //return the array read over a constantindex
              output = modelentry;
            }
          else
            {
                  // Has been simplified out. Can take any value.
                  output  = bm->CreateMaxConst(modelentry.GetValueWidth());
            }
          break;
        }
      case ITE:
        {
          ASTNode condcompute = ComputeFormulaUsingModel(term[0]);
          if (ASTTrue == condcompute)
            {
              output = TermToConstTermUsingModel(term[1], ArrayReadFlag);
            }
          else if (ASTFalse == condcompute)
            {
              output = TermToConstTermUsingModel(term[2], ArrayReadFlag);
            }
          else
            {
              cerr << "TermToConstTermUsingModel: termITE: "
                << "value of conditional is wrong: " << condcompute << endl;
            FatalError(" TermToConstTermUsingModel: termITE: cannot "
                         "compute ITE conditional against model: ", term);
            }
          break;
        }
      default:
        {
          const ASTVec& c = term.GetChildren();
          ASTVec o;
                   o.reserve(c.size());
        for (ASTVec::const_iterator it = c.begin(), itend = c.end(); it
            != itend; it++)
            {
              ASTNode ff = TermToConstTermUsingModel(*it, ArrayReadFlag);
              o.push_back(ff);
            }
          output = bm->CreateTerm(k, term.GetValueWidth(), o);
          //output is a CONST expression. compute its value and store it
          //in the CounterExampleMap
          ASTNode oo = simp->BVConstEvaluator(output);
          //the return value
          output = oo;
          break;
        }
      }

    assert(ArrayReadFlag || (BVCONST == output.GetKind()));

    //when this flag is false, we should compute the arrayread to a
    //constant. this constant is stored in the counter_example
    //datastructure
    if (!ArrayReadFlag)
      {
        CounterExampleMap[term] = output;
      }

    //cerr << "Output to TermToConstTermUsingModel: " << output << endl;
    return output;
  } //End of TermToConstTermUsingModel

  //Expands read-over-write by evaluating (readIndex=writeIndex) for
  //every writeindex until, either it evaluates to TRUE or all
  //(readIndex=writeIndex) evaluate to FALSE
  ASTNode
  AbsRefine_CounterExample::Expand_ReadOverWrite_UsingModel(
      const ASTNode& term, bool arrayread_flag)
  {
    if (READ != term.GetKind() || WRITE != term[0].GetKind())
      {
        FatalError("RemovesWrites: Input must be a READ over a WRITE", term);
      }

    ASTNode output;
    ASTNodeMap::iterator it1;
    if ((it1 = CounterExampleMap.find(term)) != CounterExampleMap.end())
      {
        const ASTNode& val = it1->second;
        if (BVCONST != val.GetKind())
          {
            //recursion is fine here. There are two maps that are checked
            //here. One is the substitutionmap. We garuntee that the value
            //of a key in the substitutionmap is always a constant.
            if (term == val)
              {
                FatalError("TermToConstTermUsingModel: The input term is "
                           "stored as-is "
                           "in the CounterExample: Not ok: ", term);
              }
            return TermToConstTermUsingModel(val, arrayread_flag);
          }
        else
          {
            return val;
          }
      }

    ASTNode newRead = term;
    const ASTNode readIndex = TermToConstTermUsingModel(newRead[1], false);
    //iteratively expand read-over-write, and evaluate against the
    //model at every iteration
    ASTNode write = newRead[0];
    do
      {
        ASTNode writeIndex = TermToConstTermUsingModel(write[1], false);

        if (writeIndex == readIndex)
          {
            //found the write-value. return it
            output = TermToConstTermUsingModel(write[2], false);
            CounterExampleMap[term] = output;
            return output;
          }

        write = write[0];
      }
    while (WRITE == write.GetKind());

    const unsigned int width = term.GetValueWidth();
    newRead = bm->CreateTerm(READ, width, write, readIndex);
    output = TermToConstTermUsingModel(newRead, arrayread_flag);

    //memoize
    CounterExampleMap[term] = output;
    return output;
  } //Expand_ReadOverWrite_UsingModel()

  /* FUNCTION: accepts a non-constant formula, and checks if the
   * formula is ASTTrue or ASTFalse w.r.t to a model
   */
  ASTNode
  AbsRefine_CounterExample::ComputeFormulaUsingModel(const ASTNode& form)
  {
    const ASTNode& in = form;
    const Kind k = form.GetKind();
    if (!(is_Form_kind(k) && BOOLEAN_TYPE == form.GetType()))
      {
        FatalError(" ComputeConstFormUsingModel: "
                   "The input is a non-formula: ", form);
      }

    //cerr << "Input to ComputeFormulaUsingModel:" << form << endl;
    ASTNodeMap::iterator it1;
    if ((it1 = ComputeFormulaMap.find(form)) != ComputeFormulaMap.end())
      {
        ASTNode res = it1->second;
        if (ASTTrue == res || ASTFalse == res)
          {
            return res;
          }
        else
          {
            FatalError("ComputeFormulaUsingModel: "
                       "The value of a formula must be TRUE or FALSE:", form);
          }
      }

    ASTNode output = ASTUndefined;
    switch (k)
      {
      case TRUE:
      case FALSE:
        output = form;
        break;
      case SYMBOL:
        if (BOOLEAN_TYPE != form.GetType())
        FatalError(" ComputeFormulaUsingModel: "
                     "Non-Boolean variables are not formulas", form);
        if (CounterExampleMap.find(form) != CounterExampleMap.end())
          {
            ASTNode counterexample_val = CounterExampleMap[form];
            if (!bm->VarSeenInTerm(form, counterexample_val))
              {
                output = ComputeFormulaUsingModel(counterexample_val);
              }
            else
              {
                output = counterexample_val;
              }
          }
        else
          {
            // Has been simplified out. Can take any value.
            output = ASTFalse;
          }
        break;
      case EQ:
      case BVLT:
      case BVLE:
      case BVGT:
      case BVGE:
      case BVSLT:
      case BVSLE:
      case BVSGT:
    case BVSGE:
      {
                ASTVec children;
                children.reserve(form.Degree());

                for (ASTVec::const_iterator it = form.begin(), itend = form.end(); it
            != itend; it++)
          {
                        children.push_back(TermToConstTermUsingModel(*it, false));
                }

                output = simp->BVConstEvaluator(bm->CreateNode(k, children));

                //evaluate formula to false if bvdiv execption occurs while
                //counterexample is being checked during refinement.
                if (bm->bvdiv_exception_occured
            && bm->counterexample_checking_during_refinement)
          {
                        output = ASTFalse;
                }

        }
      break;

      case NAND:
      case NOR:
      case NOT:
      case AND:
      case XOR:
      case IFF:
      case IMPLIES:
      case OR:
        {
                ASTVec children;
                children.reserve(form.Degree());

        for (ASTVec::const_iterator it = form.begin(), itend = form.end(); it
            != itend; it++)
                {
                        children.push_back( ComputeFormulaUsingModel(*it));
                }

            output = simp->BVConstEvaluator(bm->CreateNode(k, children));

            //evaluate formula to false if bvdiv execption occurs while
            //counterexample is being checked during refinement.
            if (bm->bvdiv_exception_occured
                && bm->counterexample_checking_during_refinement)
              {
                output = ASTFalse;
              }

        }
        break;

      case ITE:
      {
         ASTNode t0 = ComputeFormulaUsingModel(form[0]);
        if (ASTTrue == t0)
          output = ComputeFormulaUsingModel(form[1]);
        else if (ASTFalse == t0)
          output = ComputeFormulaUsingModel(form[2]);
        else
          FatalError("ComputeFormulaUsingModel: ITE: "
                     "something is wrong with the formula: ", form);
      }
        break;
      case PARAMBOOL:
        output = bm->NewParameterized_BooleanVar(form[0],form[1]);
        output = ComputeFormulaUsingModel(output);
        break;
      case FOR:
        //output = Check_FiniteLoop_UsingModel(form);
        output = ASTTrue;
        break;
      default:
          cerr << _kind_names[k];
      FatalError(" ComputeFormulaUsingModel: "
                   "the kind has not been implemented", ASTUndefined);
        break;
      }

    //cout << "ComputeFormulaUsingModel output is:" << output << endl;
    assert(ASTUndefined != output);
    assert(output.isConstant());
    ComputeFormulaMap[form] = output;
    return output;
  }

  void
  AbsRefine_CounterExample::CheckCounterExample(bool t)
  {
    //input is valid, no counterexample to check
    if (bm->ValidFlag)
      return;

    //t is true if SAT solver generated a counterexample, else it is false
    if (!t)
      FatalError("CheckCounterExample: "
                 "No CounterExample to check", ASTUndefined);
    const ASTVec c = bm->GetAsserts();

    if (bm->UserFlags.stats_flag)
        printf("checking counterexample\n");

    for (ASTVec::const_iterator it = c.begin(), itend = c.end(); it != itend; it++)
    {
        if (debug_counterexample)
                cerr << "checking" << *it;

      if (ASTFalse == ComputeFormulaUsingModel(*it))
        FatalError("CheckCounterExample:counterexample bogus:"
            "assert evaluates to FALSE under counterexample: "
                   "NOT OK", *it);
    }

    // The smtlib ones don't have a query defined.
    if ((bm->GetQuery() != ASTUndefined) && ASTTrue
        == ComputeFormulaUsingModel(bm->GetQuery()))
      FatalError("CheckCounterExample:counterexample bogus:"
        "query evaluates to TRUE under counterexample: "
                 "NOT OK", bm->GetQuery());
  }

  /* FUNCTION: queries the CounterExampleMap object with 'expr' and
   * returns the corresponding counterexample value.
   */
  ASTNode
  AbsRefine_CounterExample::GetCounterExample(bool t, const ASTNode& expr)
  {
    //input is valid, no counterexample to get
    if (bm->ValidFlag)
      return ASTUndefined;

    if (BOOLEAN_TYPE == expr.GetType())
      {
        return ComputeFormulaUsingModel(expr);
      }

    if (BVCONST == expr.GetKind())
      {
        return expr;
      }

    ASTNodeMap::iterator it;
    ASTNode output;
    if ((it = CounterExampleMap.find(expr)) != CounterExampleMap.end())
      output = TermToConstTermUsingModel(CounterExampleMap[expr], false);
    else
      output = bm->CreateZeroConst(expr.GetValueWidth());
    return output;
  } //End of GetCounterExample

  // FUNCTION: prints a counterexample for INVALID inputs.  iterate
  // through the CounterExampleMap data structure and print it to
  // stdout
  void
  AbsRefine_CounterExample::PrintCounterExample(bool t, std::ostream& os)
  {
    //input is valid, no counterexample to print
    if (bm->ValidFlag)
      {
        return;
      }

    //if this option is true then print the way dawson wants using a
    //different printer. do not use this printer.
    if (bm->UserFlags.print_arrayval_declaredorder_flag)
      {
        return;
      }

    //t is true if SAT solver generated a counterexample, else it is
    //false
    if (!t)
      {
        cerr << "PrintCounterExample: No CounterExample to print: " << endl;
        return;
      }

    bm->PLPrintNodeSet.clear();
    bm->NodeLetVarMap.clear();
    bm->NodeLetVarVec.clear();
    bm->NodeLetVarMap1.clear();

    // Take a copy of the counterexample map, 'cause TermToConstTermUsingModel
    // changes it. Which breaks the iterator otherwise.
    const ASTNodeMap c(CounterExampleMap);

    //os << "\nCOUNTEREXAMPLE: \n" << endl;
    ASTNodeMap::const_iterator it = c.begin();
    ASTNodeMap::const_iterator itend = c.end();
    for (; it != itend; it++)
      {
        const ASTNode& f = it->first;
        const ASTNode& se = it->second;

        if (ARRAY_TYPE == se.GetType())
          {
            FatalError("TermToConstTermUsingModel: "
                       "entry in counterexample is an arraytype. bogus:", se);
          }

        //skip over introduced variables
        if (f.GetKind() == SYMBOL && (ArrayTransform->FoundIntroducedSymbolSet(
            f)))
          {
            continue;
          }
        if (f.GetKind() == SYMBOL || (f.GetKind() == READ && f[0].GetKind()
            == SYMBOL && f[1].GetKind() == BVCONST))
          {
            os << "ASSERT( ";
            printer::PL_Print1(os,f,0,false);
            if(BOOLEAN_TYPE == f.GetType())
              {
                os << "<=>";
              }
            else
              {
                os << " = ";
              }
            if (BITVECTOR_TYPE == se.GetType())
              {
                ASTNode rhs = TermToConstTermUsingModel(se, false);
                                assert(rhs.isConstant());
                                printer::PL_Print1(os,rhs,0,false);
              }
            else
              {
                printer::PL_Print1(os,se,0,false);
              }
            os << " );" << endl;
          }
      }
    os.flush();
    //os << "\nEND OF COUNTEREXAMPLE" << endl;
  } //End of PrintCounterExample


  /* iterate through the CounterExampleMap data structure and print it
   * to stdout. this function prints only the declared array variables
   * IN the ORDER in which they were declared. It also assumes that
   * the variables are of the form 'varname_number'. otherwise it will
   * not print anything. This function was specifically written for
   * Dawson Engler's group (bug finding research group at Stanford)
   */
  void
  AbsRefine_CounterExample::PrintCounterExample_InOrder(bool t)
  {
    //global command-line option to print counterexample. we do not
    //want both counterexample printers to print at the sametime.
    // FIXME: This should always print the counterexample.  If you want
    // to turn it off, check the switch at the point of call.
    if (bm->UserFlags.print_counterexample_flag)
      return;

    //input is valid, no counterexample to print
    if (bm->ValidFlag)
      return;

    //print if the commandline option is '-q'. allows printing the
    //counterexample in order.
    if (!bm->UserFlags.print_arrayval_declaredorder_flag)
      return;

    //t is true if SAT solver generated a counterexample, else it is
    //false
    if (!t)
      {
        cerr << "PrintCounterExample: No CounterExample to print: " << endl;
        return;
      }

    //vector to store the integer values
    std::vector<int> out_int;
    cout << "% ";
    for (ASTVec::iterator it = bm->ListOfDeclaredVars.begin(), itend =
        bm->ListOfDeclaredVars.end(); it != itend; it++)
      {
        if (ARRAY_TYPE == it->GetType())
          {
            //get the name of the variable
            const char * c = it->GetName();
            std::string ss(c);
            if (!(0 == strncmp(ss.c_str(), "ini_", 4)))
              continue;
            reverse(ss.begin(), ss.end());

            //cout << "debugging: " << ss;
            size_t pos = ss.find('_', 0);
            if (!((0 < pos) && (pos < ss.size())))
              continue;

            //get the associated length
            std::string sss = ss.substr(0, pos);
            reverse(sss.begin(), sss.end());
            int n = atoi(sss.c_str());

            it->PL_Print(cout, 2);
            for (int j = 0; j < n; j++)
              {
                ASTNode index = bm->CreateBVConst(it->GetIndexWidth(), j);
                ASTNode readexpr = bm->CreateTerm(READ, it->GetValueWidth(),
                    *it, index);
                ASTNode val = GetCounterExample(t, readexpr);
                //cout << "ASSERT( ";
                //cout << " = ";
                out_int.push_back(val.GetUnsignedConst());
                //cout << "\n";
              }
          }
      }
    cout << endl;
    for (unsigned int jj = 0; jj < out_int.size(); jj++)
      cout << out_int[jj] << endl;
    cout << endl;
  } //End of PrintCounterExample_InOrder

  // Prints Satisfying assignment directly, for debugging.
  void
  AbsRefine_CounterExample::PrintSATModel(MINISAT::Solver& newS,
      ToSATBase::ASTNodeToSATVar& m)
  {
    if (!newS.okay())
      FatalError("PrintSATModel: NO COUNTEREXAMPLE TO PRINT", ASTUndefined);
    if (!(bm->UserFlags.stats_flag && bm->UserFlags.print_nodes_flag))
      return;

    cout << "Satisfying assignment: " << endl;
    for (ToSATBase::ASTNodeToSATVar::const_iterator it = m.begin(); it != m.end(); it++)
    {
      ASTNode symbol = it->first;
      vector<unsigned> v = it->second;

      for (int i =0 ; i < v.size();i++)
        {
        if (v[i] == ~((unsigned)0)) // nb. special value.
          continue;

        if (newS.model[v[i]] == MINISAT::l_True)
          {
            it->first.nodeprint(cout);
            cout << " {" << i << "}"  << endl;
          }
        else if (newS.model[v[i]] == MINISAT::l_False)
          {
          cout << "NOT ";
          it->first.nodeprint(cout);
          cout << " {" << i << "}"  << endl;
          }
        }
    }
  } //end of PrintSATModel()

  //FUNCTION: this function accepts a boolvector and returns a BVConst
  ASTNode
  AbsRefine_CounterExample::BoolVectoBVConst(HASHMAP<unsigned, bool> * w,
      unsigned int l)
  {
    if (l < (unsigned)w->size())
      FatalError("BoolVectorBVConst : "
        "length of bitvector does not match HASHMAP size:", ASTUndefined, l);

    CBV cc = CONSTANTBV::BitVector_Create(l,true);
    for (unsigned int jj = 0; jj < l; jj++)
      {
        if ((*w)[jj] == true)
          CONSTANTBV::BitVector_Bit_On(cc,l-1-jj);
        else
                CONSTANTBV::BitVector_Bit_Off(cc,l-1-jj);
      }

    return bm->CreateBVConst(cc,l);
  } //end of BoolVectoBVConst()

  void
  AbsRefine_CounterExample::CopySolverMap_To_CounterExample(void)
  {

    if (!simp->Return_SolverMap()->empty())
      {
        CounterExampleMap.insert(simp->Return_SolverMap()->begin(),
                                 simp->Return_SolverMap()->end());
      }
  }

  SOLVER_RETURN_TYPE
  AbsRefine_CounterExample::CallSAT_ResultCheck(MINISAT::Solver& SatSolver,
      const ASTNode& modified_input, const ASTNode& original_input, ToSATBase* tosat)
  {

    bool sat = tosat->CallSAT(SatSolver, modified_input);

    if (!sat)
      {
        //PrintOutput(true);
        return SOLVER_VALID;
      }
    else if (SatSolver.okay())
      {
        bm->GetRunTimes()->start(RunTimes::CounterExampleGeneration);
        CounterExampleMap.clear();

        ToSAT::ASTNodeToSATVar satVarToSymbol =
            tosat->SATVar_to_SymbolIndexMap();
        ConstructCounterExample(SatSolver, satVarToSymbol);
        if (bm->UserFlags.stats_flag && bm->UserFlags.print_nodes_flag)
          {
            ToSAT::ASTNodeToSATVar m = tosat->SATVar_to_SymbolIndexMap();
            PrintSATModel(SatSolver, m);
          }
        //check if the counterexample is good or not
        ComputeFormulaMap.clear();
        if (bm->counterexample_checking_during_refinement)
          bm->bvdiv_exception_occured = false;
        ASTNode orig_result = ComputeFormulaUsingModel(original_input);
        if (!(ASTTrue == orig_result || ASTFalse == orig_result))
          FatalError("TopLevelSat: Original input must compute to "
                     "true or false against model");

        bm->GetRunTimes()->stop(RunTimes::CounterExampleGeneration);

        // if the counterexample is indeed a good one, then return
        // invalid
        if (ASTTrue == orig_result)
          {
            if (bm->UserFlags.check_counterexample_flag)
                                CheckCounterExample(SatSolver.okay());

            if (bm->UserFlags.stats_flag
                || bm->UserFlags.print_counterexample_flag)
              {
            PrintCounterExample(SatSolver.okay());
            PrintCounterExample_InOrder(SatSolver.okay());
              }
            return SOLVER_INVALID;
          }
        // counterexample is bogus: flag it
        else
          {
            if (bm->UserFlags.stats_flag && bm->UserFlags.print_nodes_flag)
              {
                cout << "Supposedly bogus one: \n";
                PrintCounterExample(true);
              }

            return SOLVER_UNDECIDED;
          }
      }
    else
      {
        //Control should never reach here
        //PrintOutput(true);
        return SOLVER_ERROR;
      }
  } //end of CALLSAT_ResultCheck()     
}
;
