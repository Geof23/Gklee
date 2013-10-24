%{
  /********************************************************************
   * AUTHORS:  Trevor Hansen
   *
   * BEGIN DATE: May, 2010
   *
   * This file is modified version of the STP's smtlib.y file. Please
   * see CVCL license below
   ********************************************************************/

  /********************************************************************
   * AUTHORS: Vijay Ganesh, Trevor Hansen
   *
   * BEGIN DATE: July, 2006
   *
   * This file is modified version of the CVCL's smtlib.y file. Please
   * see CVCL license below
   ********************************************************************/

  
  /********************************************************************
   *
   * \file smtlib.y
   * 
   * Author: Sergey Berezin, Clark Barrett
   * 
   * Created: Apr 30 2005
   *
   * <hr>
   * Copyright (C) 2004 by the Board of Trustees of Leland Stanford
   * Junior University and by New York University. 
   *
   * License to use, copy, modify, sell and/or distribute this software
   * and its documentation for any purpose is hereby granted without
   * royalty, subject to the terms and conditions defined in the \ref
   * LICENSE file provided with this distribution.  In particular:
   *
   * - The above copyright notice and this permission notice must appear
   * in all copies of the software and related documentation.
   *
   * - THE SOFTWARE IS PROVIDED "AS-IS", WITHOUT ANY WARRANTIES,
   * EXPRESSED OR IMPLIED.  USE IT AT YOUR OWN RISK.
   * 
   * <hr>
   ********************************************************************/
  // -*- c++ -*-

#include "ParserInterface.h"

  using namespace std; 
  using namespace BEEV;

  // Suppress the bogus warning suppression in bison (it generates
  // compile error)
#undef __GNUC_MINOR__
  
  extern char* smt2text;
  extern int smt2lineno;
  extern int smt2lex(void);

  int yyerror(const char *s) {
    cout << "syntax error: line " << smt2lineno << "\n" << s << endl;
    cout << "  token: " << smt2text << endl;
    FatalError("");
    return 1;
  }

  ASTNode querysmt2;
  ASTVec assertionsSMT2;
  vector<string> commands;
#define YYLTYPE_IS_TRIVIAL 1
#define YYMAXDEPTH 104857600
#define YYERROR_VERBOSE 1
#define YY_EXIT_FAILURE -1
#define YYPARSE_PARAM AssertsQuery
  %}

%union {  
  unsigned uintval;                  /* for numerals in types. */
  //ASTNode,ASTVec
  BEEV::ASTNode *node;
  BEEV::ASTVec *vec;
  std::string *str;
};

%start cmd

%type <node> status
%type <vec> an_formulas an_terms

%type <node> an_term  an_formula 

%token <uintval> NUMERAL_TOK
%token <str> BVCONST_DECIMAL_TOK
%token <str> BVCONST_BINARY_TOK
%token <str> BVCONST_HEXIDECIMAL_TOK

 /* We have this so we can parse :smt-lib-version 2.0 */
%token  DECIMAL_TOK

%token <node> FORMID_TOK TERMID_TOK 
%token <str> STRING_TOK


 /* set-info tokens */
%token SOURCE_TOK
%token CATEGORY_TOK
%token DIFFICULTY_TOK
%token VERSION_TOK
%token STATUS_TOK

 /* ASCII Symbols */
 /* Semicolons (comments) are ignored by the lexer */
%token UNDERSCORE_TOK
%token LPAREN_TOK
%token RPAREN_TOK

 /* Status */
%token SAT_TOK
%token UNSAT_TOK
%token UNKNOWN_TOK
  
 /*BV SPECIFIC TOKENS*/
%token BVLEFTSHIFT_1_TOK
%token BVRIGHTSHIFT_1_TOK 
%token BVARITHRIGHTSHIFT_TOK
%token BVPLUS_TOK
%token BVSUB_TOK
%token BVNOT_TOK //bvneg in CVCL
%token BVMULT_TOK
%token BVDIV_TOK
%token SBVDIV_TOK
%token BVMOD_TOK
%token SBVREM_TOK
%token SBVMOD_TOK
%token BVNEG_TOK //bvuminus in CVCL
%token BVAND_TOK
%token BVOR_TOK
%token BVXOR_TOK
%token BVNAND_TOK
%token BVNOR_TOK
%token BVXNOR_TOK
%token BVCONCAT_TOK
%token BVLT_TOK
%token BVGT_TOK
%token BVLE_TOK
%token BVGE_TOK
%token BVSLT_TOK
%token BVSGT_TOK
%token BVSLE_TOK
%token BVSGE_TOK

%token BVSX_TOK 
%token BVEXTRACT_TOK
%token BVZX_TOK
%token BVROTATE_RIGHT_TOK
%token BVROTATE_LEFT_TOK
%token BVREPEAT_TOK 
%token BVCOMP_TOK

 /* Types for QF_BV and QF_AUFBV. */
%token BITVEC_TOK
%token ARRAY_TOK
%token BOOL_TOK

/* CORE THEORY pg. 29 of the SMT-LIB2 standard 30-March-2010. */
%token TRUE_TOK; 
%token FALSE_TOK;  
%token NOT_TOK;  
%token AND_TOK;  
%token OR_TOK;  
%token XOR_TOK;  
%token ITE_TOK; 
%token EQ_TOK;
%token IMPLIES_TOK; 

 /* CORE THEORY. But not on pg 29. */
%token DISTINCT_TOK; 
%token LET_TOK; 

// COMMANDS
%token EXIT_TOK
%token CHECK_SAT_TOK
%token LOGIC_TOK
%token NOTES_TOK
%token DECLARE_FUNCTION_TOK
%token FORMULA_TOK

 /* Functions for QF_AUFBV. */
%token SELECT_TOK;
%token STORE_TOK; 

%token END 0 "end of file"

%%
cmd: commands END
{
	if(querysmt2.IsNull()) 
    {
      querysmt2 = parserInterface->CreateNode(FALSE);
    }  
        
      ((ASTVec*)AssertsQuery)->push_back(parserInterface->nf->CreateNode(AND,assertionsSMT2));
  	  ((ASTVec*)AssertsQuery)->push_back(querysmt2);
       parserInterface->letMgr.cleanupParserSymbolTable();
       YYACCEPT;
}
;


commands: cmdi commands 
| cmdi
{}
;

cmdi:
	LPAREN_TOK EXIT_TOK RPAREN_TOK
	{
		commands.push_back("exit");
	}
|	LPAREN_TOK CHECK_SAT_TOK RPAREN_TOK
	{
		commands.push_back("check-sat");
	}
|
	LPAREN_TOK LOGIC_TOK FORMID_TOK RPAREN_TOK
	{
	  if (!(0 == strcmp($3->GetName(),"QF_BV") ||
	        0 == strcmp($3->GetName(),"QF_AUFBV"))) {
	    yyerror("Wrong input logic:");
	  }
	  delete $3;
	}
|	LPAREN_TOK NOTES_TOK attribute FORMID_TOK RPAREN_TOK
	{
	delete $4;
	}
|	LPAREN_TOK NOTES_TOK attribute DECIMAL_TOK RPAREN_TOK
	{}
|	LPAREN_TOK NOTES_TOK attribute STRING_TOK RPAREN_TOK
	{
	delete $4;
	}
|	LPAREN_TOK NOTES_TOK attribute RPAREN_TOK
	{}
|   LPAREN_TOK DECLARE_FUNCTION_TOK var_decl RPAREN_TOK
    {}
|   LPAREN_TOK FORMULA_TOK an_formula RPAREN_TOK
	{
	assertionsSMT2.push_back(*$3);
	delete $3;
	}
;

status:
SAT_TOK { 
  input_status = TO_BE_SATISFIABLE; 
  $$ = NULL; 
}
| UNSAT_TOK { 
  input_status = TO_BE_UNSATISFIABLE; 
  $$ = NULL; 
  }
| UNKNOWN_TOK 
{ 
  input_status = TO_BE_UNKNOWN; 
  $$ = NULL; 
}
;

attribute:
SOURCE_TOK
{}
| CATEGORY_TOK
{}
| DIFFICULTY_TOK
{}
| VERSION_TOK
{}
| STATUS_TOK status
{} 
;

var_decl:
FORMID_TOK LPAREN_TOK RPAREN_TOK LPAREN_TOK UNDERSCORE_TOK BITVEC_TOK NUMERAL_TOK RPAREN_TOK
{
  parserInterface->letMgr._parser_symbol_table.insert(*$1);
  //Sort_symbs has the indexwidth/valuewidth. Set those fields in
  //var
  $1->SetIndexWidth(0);
  $1->SetValueWidth($7);
  delete $1;
}
| FORMID_TOK LPAREN_TOK RPAREN_TOK BOOL_TOK
{
  $1->SetIndexWidth(0);
  $1->SetValueWidth(0);
  parserInterface->letMgr._parser_symbol_table.insert(*$1);
  delete $1;
}
| FORMID_TOK LPAREN_TOK RPAREN_TOK LPAREN_TOK ARRAY_TOK LPAREN_TOK UNDERSCORE_TOK BITVEC_TOK NUMERAL_TOK RPAREN_TOK LPAREN_TOK UNDERSCORE_TOK BITVEC_TOK NUMERAL_TOK RPAREN_TOK RPAREN_TOK
{
  parserInterface->letMgr._parser_symbol_table.insert(*$1);
  unsigned int index_len = $9;
  unsigned int value_len = $14;
  if(index_len > 0) {
    $1->SetIndexWidth($9);
  }
  else {
    FatalError("Fatal Error: parsing: BITVECTORS must be of positive length: \n");
  }

  if(value_len > 0) {
    $1->SetValueWidth($14);
  }
  else {
    FatalError("Fatal Error: parsing: BITVECTORS must be of positive length: \n");
  }
  delete $1;
}
;

an_formulas:
an_formula
{
  $$ = new ASTVec;
  if ($1 != NULL) {
    $$->push_back(*$1);
    delete $1;
  }
}
|
an_formulas an_formula
{
  if ($1 != NULL && $2 != NULL) {
    $1->push_back(*$2);
    $$ = $1;
    delete $2;
  }
}
;

an_formula:   
TRUE_TOK
{
  $$ = new ASTNode(parserInterface->CreateNode(TRUE)); 
  assert(0 == $$->GetIndexWidth()); 
  assert(0 == $$->GetValueWidth());
}
| FALSE_TOK
{
  $$ = new ASTNode(parserInterface->CreateNode(FALSE)); 
  assert(0 == $$->GetIndexWidth()); 
  assert(0 == $$->GetValueWidth());
}
| FORMID_TOK
{
  $$ = new ASTNode(parserInterface->letMgr.ResolveID(*$1)); 
  delete $1;      
}
| LPAREN_TOK EQ_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(EQ,*$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK DISTINCT_TOK an_terms RPAREN_TOK
{
  using namespace BEEV;

  ASTVec terms = *$3;
  ASTVec forms;

  for(ASTVec::const_iterator it=terms.begin(),itend=terms.end();
      it!=itend; it++) {
    for(ASTVec::const_iterator it2=it+1; it2!=itend; it2++) {
      ASTNode n = (parserInterface->nf->CreateNode(NOT, parserInterface->nf->CreateNode(EQ, *it, *it2)));

          
      forms.push_back(n); 
    }
  }

  if(forms.size() == 0) 
    FatalError("empty distinct");
 
  $$ = (forms.size() == 1) ?
    new ASTNode(forms[0]) :
    new ASTNode(parserInterface->nf->CreateNode(AND, forms));

  delete $3;
}
| LPAREN_TOK DISTINCT_TOK an_formulas RPAREN_TOK
{
  using namespace BEEV;

  ASTVec terms = *$3;
  ASTVec forms;

  for(ASTVec::const_iterator it=terms.begin(),itend=terms.end();
      it!=itend; it++) {
    for(ASTVec::const_iterator it2=it+1; it2!=itend; it2++) {
      ASTNode n = (parserInterface->nf->CreateNode(NOT, parserInterface->nf->CreateNode(IFF, *it, *it2)));
      forms.push_back(n); 
    }
  }

  if(forms.size() == 0) 
    FatalError("empty distinct");
 
  $$ = (forms.size() == 1) ?
    new ASTNode(forms[0]) :
    new ASTNode(parserInterface->nf->CreateNode(AND, forms));

  delete $3;
}
| LPAREN_TOK BVSLT_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVSLT, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVSLE_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVSLE, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVSGT_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVSGT, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVSGE_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVSGE, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVLT_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVLT, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVLE_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVLE, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVGT_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVGT, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK BVGE_TOK an_term an_term RPAREN_TOK
{
  ASTNode * n = new ASTNode(parserInterface->nf->CreateNode(BVGE, *$3, *$4));
  $$ = n;
  delete $3;
  delete $4;      
}
| LPAREN_TOK an_formula RPAREN_TOK
{
  $$ = $2;
}
| LPAREN_TOK NOT_TOK an_formula RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(NOT, *$3));
  delete $3;
}
| LPAREN_TOK IMPLIES_TOK an_formula an_formula RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(IMPLIES, *$3, *$4));
  delete $3;
  delete $4;
}
| LPAREN_TOK ITE_TOK an_formula an_formula an_formula RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(ITE, *$3, *$4, *$5));
  delete $3;
  delete $4;
  delete $5;
}
| LPAREN_TOK AND_TOK an_formulas RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(AND, *$3));
  delete $3;
}
| LPAREN_TOK OR_TOK an_formulas RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(OR, *$3));
  delete $3;
}
| LPAREN_TOK XOR_TOK an_formula an_formula RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(XOR, *$3, *$4));
  delete $3;
  delete $4;
}
| LPAREN_TOK EQ_TOK an_formula an_formula RPAREN_TOK
{
  $$ = new ASTNode(parserInterface->nf->CreateNode(IFF, *$3, *$4));
  delete $3;
  delete $4;
}
| LPAREN_TOK LET_TOK LPAREN_TOK lets RPAREN_TOK an_formula RPAREN_TOK
{
  $$ = $6;
  //Cleanup the LetIDToExprMap
  parserInterface->letMgr.CleanupLetIDMap();                      
}
;

lets: let lets 
| let
{};

let: LPAREN_TOK FORMID_TOK an_formula RPAREN_TOK
{
  //set the valuewidth of the identifier
  $2->SetValueWidth($3->GetValueWidth());
  $2->SetIndexWidth($3->GetIndexWidth());
      
  //populate the hashtable from LET-var -->
  //LET-exprs and then process them:
  //
  //1. ensure that LET variables do not clash
  //1. with declared variables.
  //
  //2. Ensure that LET variables are not
  //2. defined more than once
  parserInterface->letMgr.LetExprMgr(*$2,*$3);
  
  delete $2;
  delete $3;
}
| LPAREN_TOK FORMID_TOK an_term RPAREN_TOK
{
  //set the valuewidth of the identifier
  $2->SetValueWidth($3->GetValueWidth());
  $2->SetIndexWidth($3->GetIndexWidth());
      
  //populate the hashtable from LET-var -->
  //LET-exprs and then process them:
  //
  //1. ensure that LET variables do not clash
  //1. with declared variables.
  //
  //2. Ensure that LET variables are not
  //2. defined more than once
  parserInterface->letMgr.LetExprMgr(*$2,*$3);
  
  delete $2;
  delete $3;

}
;
 
an_terms: 
an_term
{
  $$ = new ASTVec;
  if ($1 != NULL) {
    $$->push_back(*$1);
    delete $1;
  }
}
|
an_terms an_term
{
  if ($1 != NULL && $2 != NULL) {
    $1->push_back(*$2);
    $$ = $1;
    delete $2;
  }
}
;

an_term: 
TERMID_TOK
{
  $$ = new ASTNode(parserInterface->letMgr.ResolveID(*$1));
  delete $1;
}
| LPAREN_TOK an_term RPAREN_TOK
{
  $$ = $2;
}
| SELECT_TOK an_term an_term
{
  //ARRAY READ
  // valuewidth is same as array, indexwidth is 0.
  ASTNode array = *$2;
  ASTNode index = *$3;
  unsigned int width = array.GetValueWidth();
  ASTNode * n = 
    new ASTNode(parserInterface->nf->CreateTerm(READ, width, array, index));
  $$ = n;
  delete $2;
  delete $3;
}
| STORE_TOK an_term an_term an_term
{
  //ARRAY WRITE
  unsigned int width = $4->GetValueWidth();
  ASTNode array = *$2;
  ASTNode index = *$3;
  ASTNode writeval = *$4;
  ASTNode write_term = parserInterface->nf->CreateArrayTerm(WRITE,$2->GetIndexWidth(),width,array,index,writeval);
  ASTNode * n = new ASTNode(write_term);
  $$ = n;
  delete $2;
  delete $3;
  delete $4;
}
| LPAREN_TOK UNDERSCORE_TOK BVEXTRACT_TOK  NUMERAL_TOK  NUMERAL_TOK RPAREN_TOK an_term
{
  int width = $4 - $5 + 1;
  if (width < 0)
    yyerror("Negative width in extract");
      
  if((unsigned)$4 >= $7->GetValueWidth())
    yyerror("Parsing: Wrong width in BVEXTRACT\n");                      
      
  ASTNode hi  =  parserInterface->CreateBVConst(32, $4);
  ASTNode low =  parserInterface->CreateBVConst(32, $5);
  ASTNode output = parserInterface->nf->CreateTerm(BVEXTRACT, width, *$7,hi,low);
  ASTNode * n = new ASTNode(output);
  $$ = n;
  delete $7;
}
| LPAREN_TOK UNDERSCORE_TOK BVZX_TOK  NUMERAL_TOK  RPAREN_TOK an_term 
{
  if (0 != $4)
    {
      unsigned w = $6->GetValueWidth() + $4;
      ASTNode leading_zeroes = parserInterface->CreateZeroConst($4);
      ASTNode *n =  new ASTNode(parserInterface->nf->CreateTerm(BVCONCAT,w,leading_zeroes,*$6));
      $$ = n;
      delete $6;
    }
  else
    $$ = $6;
}
|  LPAREN_TOK UNDERSCORE_TOK BVSX_TOK  NUMERAL_TOK  RPAREN_TOK an_term 
{
  unsigned w = $6->GetValueWidth() + $4;
  ASTNode width = parserInterface->CreateBVConst(32,w);
  ASTNode *n =  new ASTNode(parserInterface->nf->CreateTerm(BVSX,w,*$6,width));
  $$ = n;
  delete $6;
}

|  ITE_TOK an_formula an_term an_term 
{
  const unsigned int width = $3->GetValueWidth();
  $$ = new ASTNode(parserInterface->nf->CreateArrayTerm(ITE,$4->GetIndexWidth(), width,*$2, *$3, *$4));      
  delete $2;
  delete $3;
  delete $4;
}
|  BVCONCAT_TOK an_term an_term 
{
  const unsigned int width = $2->GetValueWidth() + $3->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVCONCAT, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}
|  BVNOT_TOK an_term
{
  //this is the BVNEG (term) in the CVCL language
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVNEG, width, *$2));
  $$ = n;
  delete $2;
}
|  BVNEG_TOK an_term
{
  //this is the BVUMINUS term in CVCL langauge
  unsigned width = $2->GetValueWidth();
  ASTNode * n =  new ASTNode(parserInterface->nf->CreateTerm(BVUMINUS,width,*$2));
  $$ = n;
  delete $2;
}
|  BVAND_TOK an_term an_term 
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVAND, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}
|  BVOR_TOK an_term an_term 
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVOR, width, *$2, *$3)); 
  $$ = n;
  delete $2;
  delete $3;
}
|  BVXOR_TOK an_term an_term 
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVXOR, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}
| BVXNOR_TOK an_term an_term
{
//   (bvxnor s t) abbreviates (bvor (bvand s t) (bvand (bvnot s) (bvnot t)))
      unsigned int width = $2->GetValueWidth();
      ASTNode * n = new ASTNode(
      parserInterface->nf->CreateTerm( BVOR, width,
     parserInterface->nf->CreateTerm(BVAND, width, *$2, *$3),
     parserInterface->nf->CreateTerm(BVAND, width,
	     parserInterface->nf->CreateTerm(BVNEG, width, *$2),
     	 parserInterface->nf->CreateTerm(BVNEG, width, *$3)
     )));

      $$ = n;
      delete $2;
      delete $3;
}
|  BVCOMP_TOK an_term an_term
{
  	ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(ITE, 1, 
  	parserInterface->nf->CreateNode(EQ, *$2, *$3),
  	parserInterface->CreateOneConst(1),
  	parserInterface->CreateZeroConst(1)));
  	
      $$ = n;
      delete $2;
      delete $3;
}
|  BVSUB_TOK an_term an_term 
{
  const unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVSUB, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}
|  BVPLUS_TOK an_term an_term 
{
  const unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVPLUS, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;

}
|  BVMULT_TOK an_term an_term 
{
  const unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVMULT, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}
|      BVDIV_TOK an_term an_term  
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVDIV, width, *$2, *$3));
  $$ = n;

  delete $2;
  delete $3;
}
|      BVMOD_TOK an_term an_term
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVMOD, width, *$2, *$3));
  $$ = n;

  delete $2;
  delete $3;
}
|      SBVDIV_TOK an_term an_term
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(SBVDIV, width, *$2, *$3));
  $$ = n;

  delete $2;
  delete $3;
}
|      SBVREM_TOK an_term an_term
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(SBVREM, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}        
|      SBVMOD_TOK an_term an_term
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(SBVMOD, width, *$2, *$3));
  $$ = n;
  delete $2;
  delete $3;
}        
|  BVNAND_TOK an_term an_term 
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVNEG, width, parserInterface->nf->CreateTerm(BVAND, width, *$2, *$3)));
  $$ = n;
  delete $2;
  delete $3;
}
|  BVNOR_TOK an_term an_term 
{
  unsigned int width = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVNEG, width, parserInterface->nf->CreateTerm(BVOR, width, *$2, *$3))); 
  $$ = n;
  delete $2;
  delete $3;
}
|  BVLEFTSHIFT_1_TOK an_term an_term 
{
  // shifting left by who know how much?
  unsigned int w = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVLEFTSHIFT,w,*$2,*$3));
  $$ = n;
  delete $2;
  delete $3;
}
| BVRIGHTSHIFT_1_TOK an_term an_term 
{
  // shifting right by who know how much?
  unsigned int w = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVRIGHTSHIFT,w,*$2,*$3));
  $$ = n;
  delete $2;
  delete $3;
}
|  BVARITHRIGHTSHIFT_TOK an_term an_term
{
  // shifting arithmetic right by who know how much?
  unsigned int w = $2->GetValueWidth();
  ASTNode * n = new ASTNode(parserInterface->nf->CreateTerm(BVSRSHIFT,w,*$2,*$3));
  $$ = n;
  delete $2;
  delete $3;
}
| LPAREN_TOK UNDERSCORE_TOK BVROTATE_LEFT_TOK  NUMERAL_TOK  RPAREN_TOK an_term
{
  ASTNode *n;
  unsigned width = $6->GetValueWidth();
  unsigned rotate = $4 % width;
  if (0 == rotate)
    {
      n = $6;
    }
  else 
    {
      ASTNode high = parserInterface->CreateBVConst(32,width-1);
      ASTNode zero = parserInterface->CreateBVConst(32,0);
      ASTNode cut = parserInterface->CreateBVConst(32,width-rotate);
      ASTNode cutMinusOne = parserInterface->CreateBVConst(32,width-rotate-1);

      ASTNode top =  parserInterface->nf->CreateTerm(BVEXTRACT,rotate,*$6,high, cut);
      ASTNode bottom =  parserInterface->nf->CreateTerm(BVEXTRACT,width-rotate,*$6,cutMinusOne,zero);
      n =  new ASTNode(parserInterface->nf->CreateTerm(BVCONCAT,width,bottom,top));
      delete $6;
    }
      
  $$ = n;
}
| LPAREN_TOK UNDERSCORE_TOK BVROTATE_RIGHT_TOK  NUMERAL_TOK  RPAREN_TOK an_term 
{
  ASTNode *n;
  unsigned width = $6->GetValueWidth();
  unsigned rotate = $4 % width;
  if (0 == rotate)
    {
      n = $6;
    }
  else 
    {
      ASTNode high = parserInterface->CreateBVConst(32,width-1);
      ASTNode zero = parserInterface->CreateBVConst(32,0);
      ASTNode cut = parserInterface->CreateBVConst(32,rotate); 
      ASTNode cutMinusOne = parserInterface->CreateBVConst(32,rotate-1);

      ASTNode bottom =  parserInterface->nf->CreateTerm(BVEXTRACT,rotate,*$6,cutMinusOne, zero);
      ASTNode top =  parserInterface->nf->CreateTerm(BVEXTRACT,width-rotate,*$6,high,cut);
      n =  new ASTNode(parserInterface->nf->CreateTerm(BVCONCAT,width,bottom,top));
      delete $6;
    }
      
  $$ = n;
}
| LPAREN_TOK UNDERSCORE_TOK BVREPEAT_TOK  NUMERAL_TOK RPAREN_TOK an_term
{
	  unsigned count = $4;
	  if (count < 1)
	  	FatalError("One or more repeats please");

	  unsigned w = $6->GetValueWidth();  
      ASTNode n =  *$6;
      
      for (unsigned i =1; i < count; i++)
      {
      	  n = parserInterface->nf->CreateTerm(BVCONCAT,w*(i+1),n,*$6);
      }
      $$ = new ASTNode(n);
      delete $6;
}
| UNDERSCORE_TOK BVCONST_DECIMAL_TOK NUMERAL_TOK
{
	$$ = new ASTNode(parserInterface->CreateBVConst($2, 10, $3));
    $$->SetValueWidth($3);
    delete $2;
}
| BVCONST_HEXIDECIMAL_TOK
{
	unsigned width = $1->length()*4;
	$$ = new ASTNode(parserInterface->CreateBVConst($1, 16, width));
    $$->SetValueWidth(width);
    delete $1;
}
| BVCONST_BINARY_TOK
{
	unsigned width = $1->length();
	$$ = new ASTNode(parserInterface->CreateBVConst($1, 2, width));
    $$->SetValueWidth(width);
    delete $1;
}
;

%%
