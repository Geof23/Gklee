// -*- c++ -*-
/********************************************************************
 * AUTHORS: Vijay Ganesh
 *
 * BEGIN DATE: November, 2005
 *
 * LICENSE: Please view LICENSE file in the home dir of this Program
 ********************************************************************/

#ifndef SIMPLIFIER_H
#define SIMPLIFIER_H

#include "../AST/AST.h"
#include "../STPManager/STPManager.h"
#include "../AST/NodeFactory/SimplifyingNodeFactory.h"
#include "SubstitutionMap.h"

namespace BEEV
{
  ASTNode NonMemberBVConstEvaluator(const ASTNode& t);

  class Simplifier
  {
    friend class counterexample;
  private:

    /****************************************************************
     * Private Data and TypeDefs                                    *
     ****************************************************************/

    // Handy defs
    ASTNode ASTTrue, ASTFalse, ASTUndefined;

    // Memo table for simplifcation. Key is unsimplified node, and
    // value is simplified node.
    ASTNodeMap * SimplifyMap;
    ASTNodeMap * SimplifyNegMap;
    ASTNodeSet AlwaysTrueFormMap;
    ASTNodeMap MultInverseMap;

    // For ArrayWrite Abstraction: map from read-over-write term to
    // newname.
    ASTNodeMap * ReadOverWrite_NewName_Map;
      
    // For ArrayWrite Refinement: Map new arraynames to
    // Read-Over-Write terms
    ASTNodeMap NewName_ReadOverWrite_Map;

    //Ptr to STP Manager
    STPMgr * _bm;

    NodeFactory * nf;

    SubstitutionMap substitutionMap;

    void checkIfInSimplifyMap(const ASTNode& n, ASTNodeSet visited);
  public:
      
    /****************************************************************
     * Public Member Functions                                      *
     ****************************************************************/      
    Simplifier(STPMgr * bm) : _bm(bm),
    substitutionMap(this,bm)
    {
      SimplifyMap    = new ASTNodeMap(INITIAL_TABLE_SIZE);
      SimplifyNegMap = new ASTNodeMap(INITIAL_TABLE_SIZE);
      ReadOverWrite_NewName_Map = new ASTNodeMap();

      ASTTrue  = bm->CreateNode(TRUE);
      ASTFalse = bm->CreateNode(FALSE);
      ASTUndefined = bm->CreateNode(UNDEFINED);

      nf = new SimplifyingNodeFactory(*bm->hashingNodeFactory,*bm);
    }
      
    ~Simplifier()
    {
      delete SimplifyMap;
      delete SimplifyNegMap;
      delete ReadOverWrite_NewName_Map;
      delete nf;
    }

    /****************************************************************
     * Functions to check and update various Maps                   *
     ****************************************************************/      
      
    //Check the map passed to SimplifyTerm
    bool CheckMap(ASTNodeMap* VarConstMap, 
                  const ASTNode& key, ASTNode& output);

      
    //functions for checking and updating simplifcation map
    bool CheckSimplifyMap(const ASTNode& key, 
                          ASTNode& output, 
                          bool pushNeg, ASTNodeMap* VarConstMap=NULL);
    void UpdateSimplifyMap(const ASTNode& key, 
                           const ASTNode& value, 
                           bool pushNeg, ASTNodeMap* VarConstMap=NULL);
    bool CheckAlwaysTrueFormMap(const ASTNode& key);
    void UpdateAlwaysTrueFormMap(const ASTNode& val);
    bool CheckMultInverseMap(const ASTNode& key, ASTNode& output);
    void UpdateMultInverseMap(const ASTNode& key, const ASTNode& value);
      
    //Map for solved variables
    bool UpdateSolverMap(const ASTNode& e0, const ASTNode& e1);     
    ASTNode CreateSubstitutionMap(const ASTNode& a,
  		ArrayTransformer *at);

    //substitution
    bool CheckSubstitutionMap(const ASTNode& a, ASTNode& output);
    bool CheckSubstitutionMap(const ASTNode& a);
    bool UpdateSubstitutionMap(const ASTNode& e0, const ASTNode& e1);

    ASTNode applySubstitutionMap(const ASTNode& n);

    void ResetSimplifyMaps(void);

    /****************************************************************
     * Simplification functions                                     *
     ****************************************************************/      

    ASTNode SimplifyFormula_TopLevel(const ASTNode& a, 
                                     bool pushNeg,
                                     ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyTerm_TopLevel(const ASTNode& b);


    ASTNode SimplifyFormula(const ASTNode& a, 
                            bool pushNeg, 
                            ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyTerm(const ASTNode& inputterm, 
                         ASTNodeMap* VarConstMap=NULL);
      

    ASTNode SimplifyFormula_NoRemoveWrites(const ASTNode& a, 
                                           bool pushNeg, 
                                           ASTNodeMap* VarConstMap=NULL);

    void CheckSimplifyInvariant(const ASTNode& a, 
                                const ASTNode& output);


    ASTNode SimplifyAtomicFormula(const ASTNode& a, 
                                  bool pushNeg, 
                                  ASTNodeMap* VarConstMap=NULL);

    ASTNode CreateSimplifiedEQ(const ASTNode& t1, 
                               const ASTNode& t2);

    ASTNode ITEOpt_InEqs(const ASTNode& in1, 
                         ASTNodeMap* VarConstMap=NULL);

    ASTNode PullUpITE(const ASTNode& in);

    ASTNode RemoveContradictionsFromAND(const ASTNode& in);
      
    ASTNode CreateSimplifiedTermITE(const ASTNode& t1, 
                                    const ASTNode& t2, 
                                    const ASTNode& t3);

    ASTNode CreateSimplifiedFormulaITE(const ASTNode& in0, 
                                       const ASTNode& in1, 
                                       const ASTNode& in2);

    ASTNode CreateSimplifiedINEQ(const Kind k,
                                 const ASTNode& a0, 
                                 const ASTNode& a1, bool pushNeg);

    ASTNode SimplifyNotFormula(const ASTNode& a, 
                               bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyAndOrFormula(const ASTNode& a,
                                 bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyXorFormula(const ASTNode& a,
                               bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyNandFormula(const ASTNode& a,
                                bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyNorFormula(const ASTNode& a,
                               bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyImpliesFormula(const ASTNode& a,
                                   bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyIffFormula(const ASTNode& a,
                               bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyIteFormula(const ASTNode& a,
                               bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyForFormula(const ASTNode& a,
                               bool pushNeg, ASTNodeMap* VarConstMap=NULL);

    ASTNode Flatten(const ASTNode& a);

    ASTNode FlattenOneLevel(const ASTNode& a);

    ASTNode FlattenAndOr(const ASTNode& a);

    ASTNode CombineLikeTerms(const ASTNode& a);

    ASTNode LhsMinusRhs(const ASTNode& eq);

    ASTNode DistributeMultOverPlus(const ASTNode& a,
                                   bool startdistribution = false);

    ASTNode ConvertBVSXToITE(const ASTNode& a);

    ASTNode BVConstEvaluator(const ASTNode& t);

    //checks if the input constant is odd or not
    bool BVConstIsOdd(const ASTNode& c);

    //computes the multiplicatve inverse of the input
    ASTNode MultiplicativeInverse(const ASTNode& c);

    //Replaces WRITE(Arr,i,val) with ITE(j=i, val, READ(Arr,j))
    ASTNode RemoveWrites_TopLevel(const ASTNode& term);
    ASTNode RemoveWrites(const ASTNode& term);
    ASTNode SimplifyWrites_InPlace(const ASTNode& term, 
                                   ASTNodeMap* VarConstMap=NULL);

    ASTNode SimplifyArrayTerm(const ASTNode& term,ASTNodeMap* VarConstMap);

    ASTNode ReadOverWrite_To_ITE(const ASTNode& term, 
                                 ASTNodeMap* VarConstMap=NULL);

    void printCacheStatus();

    //FIXME: Get rid of this horrible function
    const ASTNodeMap * ReadOverWriteMap()
    {
      return ReadOverWrite_NewName_Map;
    } // End of ReadOverWriteMap()
      
    const ASTNodeMap * Return_SolverMap()
    {
    	return substitutionMap.Return_SolverMap();
    } // End of SolverMap()

    void ClearAllTables(void) 
    {
      SimplifyMap->clear();
      SimplifyNegMap->clear();
      ReadOverWrite_NewName_Map->clear();
      NewName_ReadOverWrite_Map.clear();
      AlwaysTrueFormMap.clear();
      MultInverseMap.clear();
      substitutionMap.clear();
    }

    // These can be cleared (to save memory) without changing the answer.
    void ClearCaches()
    {
        AlwaysTrueFormMap.clear();
        MultInverseMap.clear();
        SimplifyMap->clear();
        SimplifyNegMap->clear();
    }

  };//end of class Simplifier
}; //end of namespace
#endif
