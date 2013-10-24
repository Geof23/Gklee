// -*- c++ -*-
/********************************************************************
 * AUTHORS: Vijay Ganesh, Trevor Hansen
 *
 * BEGIN DATE: November, 2005
 *
 * LICENSE: Please view LICENSE file in the home dir of this Program
 ********************************************************************/

#ifndef TOSAT_H
#define TOSAT_H

#include "ToCNF.h"

#include "../AST/AST.h"
#include "../sat/sat.h"
#include "../STPManager/STPManager.h"
#include "ToSATBase.h"

namespace BEEV
{
  class ToSAT :public ToSATBase
  {

  private:
    /****************************************************************
     * Private Typedefs and Data                                    *
     ****************************************************************/

    // MAP: This is a map from ASTNodes to MINISAT::Vars.
    //
    // The map is populated while ASTclauses are read from the AST
    // ClauseList returned by CNF converter. For every new boolean
    // variable in ASTClause a new MINISAT::Var is created (these vars
    // typedefs for ints)
    typedef HASHMAP<
    ASTNode, 
    MINISAT::Var, 
    ASTNode::ASTNodeHasher, 
    ASTNode::ASTNodeEqual> ASTtoSATMap;
    ASTtoSATMap _ASTNode_to_SATVar_Map;

    // MAP: This is a map from  ASTNodes to MINISAT::Vars for SYMBOLS>
    //
    // Reverse map used in building counterexamples. MINISAT returns a
    // model in terms of MINISAT Vars, and this map helps us convert
    // it to a model over ASTNode variables.
    ASTNodeToSATVar SATVar_to_SymbolIndex;

    int CNFFileNameCounter;
    int benchFileNameCounter;

    /****************************************************************
     * Private Member Functions                                     *
     ****************************************************************/

    //looksup a MINISAT var from the minisat-var memo-table. if none
    //exists, then creates one.  Treat the result as const.
    MINISAT::Var LookupOrCreateSATVar(MINISAT::Solver& S, 
                                      const ASTNode& n);


    //Iteratively goes through the Clause Buckets, and calls
      //toSATandSolve()
    bool CallSAT_On_ClauseBuckets(MINISAT::Solver& SatSolver,
                                    ClauseBuckets * cb
                                    , CNFMgr*& cm);


      // Converts the clause to SAT and calls SAT solver
      bool toSATandSolve(MINISAT::Solver& S,
                         ClauseList& cll,
                         bool final,
                         CNFMgr*& cm,
  		       bool add_xor_clauses=false,
  		       bool enable_clausal_abstraction=false);


  public:
    /****************************************************************
     * Public Member Functions                                      *
     ****************************************************************/
    
    // Constructor
    ToSAT(STPMgr * bm) :
      ToSATBase(bm)
    {
      CNFFileNameCounter = 0;
      benchFileNameCounter = 0;

    }

    // Bitblasts, CNF conversion and calls toSATandSolve()
    bool CallSAT(MINISAT::Solver& SatSolver,
                 const ASTNode& input);

    ASTNodeToSATVar& SATVar_to_SymbolIndexMap()
    {
      return SATVar_to_SymbolIndex;
    }

    void ClearAllTables(void)
    {
      _ASTNode_to_SATVar_Map.clear();
      SATVar_to_SymbolIndex.clear();
    }


    ~ToSAT()
    {
       ClearAllTables();
    }
  }; //end of class ToSAT
}; //end of namespace

#endif
