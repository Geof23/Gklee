%{
/********************************************************************
 * AUTHORS: Vijay Ganesh
 *
 * BEGIN DATE: November, 2005
 *
 * LICENSE: Please view LICENSE file in the home dir of this Program
 ********************************************************************/

#include <iostream>
#include "parser.h"
#include "parseCVC_defs.h"
#include "ParserInterface.h"

  using namespace std;
  using namespace BEEV;  
  extern char *yytext;
  extern int cvcerror (const char *msg);
%}

%option noyywrap
%option nounput
%option noreject
%option noyymore
%option yylineno
%x	COMMENT
%x	STRING_LITERAL
LETTER	([a-zA-Z])
HEX     ([0-9a-fA-F])
BITS    ([0-1])
DIGIT	([0-9])
OPCHAR	(['?\_$])
ANYTHING ({LETTER}|{DIGIT}|{OPCHAR})
%%

[()[\]{},.;:'!#?_=]  { return yytext[0];}

[\n]             { /*Skip new line */ }
[ \t\r\f]	 { /* skip whitespace */ }
0b{BITS}+	 { cvclval.node = new BEEV::ASTNode(BEEV::ParserBM->CreateBVConst(yytext+2,  2)); return BVCONST_TOK;}
0bin{BITS}+	 { cvclval.node = new BEEV::ASTNode(BEEV::ParserBM->CreateBVConst(yytext+4,  2)); return BVCONST_TOK;}
0x{HEX}+         { cvclval.node = new BEEV::ASTNode(BEEV::ParserBM->CreateBVConst(yytext+2, 16)); return BVCONST_TOK;}
0h{HEX}+         { cvclval.node = new BEEV::ASTNode(BEEV::ParserBM->CreateBVConst(yytext+2, 16)); return BVCONST_TOK;}
0hex{HEX}+       { cvclval.node = new BEEV::ASTNode(BEEV::ParserBM->CreateBVConst(yytext+4, 16)); return BVCONST_TOK;}
{DIGIT}+	 { cvclval.uintval = strtoul(yytext, NULL, 10); return NUMERAL_TOK;}
\'b[0-1]+ { cvclval.str = strdup(yytext+2); return BIN_BASED_NUMBER;}
\'d[0-9]+ { cvclval.str = strdup(yytext+2); return DEC_BASED_NUMBER;}
\'h[0-9a-fA-F]+ { cvclval.str = strdup(yytext+2); return HEX_BASED_NUMBER;}

"%"		 { BEGIN COMMENT;}
<COMMENT>"\n"	 { BEGIN INITIAL; /* return to normal mode */}
<COMMENT>.	 { /* stay in comment mode */}

"ARRAY"		 { return ARRAY_TOK; }
"OF"		 { return OF_TOK; }
"WITH"		 { return WITH_TOK; }
"AND"		 { return AND_TOK;}
"NAND"		 { return NAND_TOK;}
"NOR"		 { return NOR_TOK;}
"NOT"		 { return NOT_TOK; }
"FOR"		 { return FOR_TOK; }
"EXCEPT"	 { return EXCEPT_TOK; }
"OR"		 { return OR_TOK; }
"/="		 { return NEQ_TOK; }
 ":="            { return ASSIGN_TOK;}
"=>"		 { return IMPLIES_TOK; }
"<=>"		 { return IFF_TOK; }
"XOR"		 { return XOR_TOK; }
"IF"		 { return IF_TOK; }
"THEN"		 { return THEN_TOK; }
"ELSE"		 { return ELSE_TOK; }
"ELSIF"		 { return ELSIF_TOK; }
"END"		 { return END_TOK; }
"ENDIF"		 { return ENDIF_TOK; }
"BV"             { return BV_TOK;}
"BITVECTOR"      { return BV_TOK;}
"BOOLEAN"        { return BOOLEAN_TOK;}
"<<"             { return BVLEFTSHIFT_TOK;}
">>"             { return BVRIGHTSHIFT_TOK;}
"BVPLUS"         { return BVPLUS_TOK;}
"BVSUB"          { return BVSUB_TOK;}
"BVUMINUS"       { return BVUMINUS_TOK;}
"BVMULT"         { return BVMULT_TOK;}
"BVDIV"          { return BVDIV_TOK;}
"BVMOD"          { return BVMOD_TOK;}
"SBVDIV"         { return SBVDIV_TOK;}
"SBVMOD"         { return SBVREM_TOK;}
"SBVREM"         { return SBVREM_TOK;}
"~"              { return BVNEG_TOK;}
"&"              { return BVAND_TOK;}
"|"              { return BVOR_TOK;}
"BVXOR"          { return BVXOR_TOK;}
"BVNAND"         { return BVNAND_TOK;}
"BVNOR"          { return BVNOR_TOK;}
"BVXNOR"         { return BVXNOR_TOK;}
"@"              { return BVCONCAT_TOK;}
"BVLT"           { return BVLT_TOK;}
"BVGT"           { return BVGT_TOK;}
"BVLE"           { return BVLE_TOK;}
"BVGE"           { return BVGE_TOK;}
"BVSLT"          { return BVSLT_TOK;}
"BVSGT"          { return BVSGT_TOK;}
"BVSLE"          { return BVSLE_TOK;}
"BVSGE"          { return BVSGE_TOK;}
"BVSX"           { return BVSX_TOK;} 
"SBVLT"          { return BVSLT_TOK;}
"SBVGT"          { return BVSGT_TOK;}
"SBVLE"          { return BVSLE_TOK;}
"SBVGE"          { return BVSGE_TOK;}
"SX"             { return BVSX_TOK;} 
"BOOLEXTRACT"    { return BOOLEXTRACT_TOK;}
"BOOLBV"         { return BOOL_TO_BV_TOK;}
"ASSERT"	 { return ASSERT_TOK; }
"QUERY"	         { return QUERY_TOK; }
"FALSE"          { return FALSELIT_TOK;}
"TRUE"           { return TRUELIT_TOK;}
"IN"             { return IN_TOK;}
"LET"            { return LET_TOK;}
"COUNTEREXAMPLE" { return COUNTEREXAMPLE_TOK;}
"COUNTERMODEL"   { return COUNTEREXAMPLE_TOK;}
"PUSH"           { return PUSH_TOK;}
"POP"            { return POP_TOK;}

(({LETTER})|(_)({ANYTHING}))({ANYTHING})*	{
  BEEV::ASTNode nptr = (BEEV::ParserBM)->CreateSymbol(yytext); 

  // Check valuesize to see if it's a prop var.  I don't like doing
  // type determination in the lexer, but it's easier than rewriting
  // the whole grammar to eliminate the term/formula distinction.  
  cvclval.node = 
    new BEEV::ASTNode(parserInterface->letMgr.ResolveID(nptr));
  //cvclval.node = new BEEV::ASTNode(nptr);
  if ((cvclval.node)->GetType() == BOOLEAN_TYPE)
    return FORMID_TOK;
  else 
    return TERMID_TOK;  
}

.                { cvcerror("Illegal input character."); }
%%
