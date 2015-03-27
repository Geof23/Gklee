#include "StrConstraint.h"
#include "llvm/Support/raw_ostream.h"

#include "iostream"
#include "StrExpr.h"

using namespace klee;
using namespace llvm;

/*******************************************************************
 Instantiate static variables
 ******************************************************************/

unsigned StrExpr::len_id = 0;
unsigned StrExpr::str_id = 0;

klee::ref<Expr> StrExpr::createMultRead(const Array *array, unsigned index, unsigned nbytes) {
  assert(nbytes > 0 && "createMultRead: invalid width");

  UpdateList ul(array, 0);
  
  klee::ref<Expr> bvals[255];
  for (unsigned i = 0; i < nbytes; i++) {
    bvals[i] = ReadExpr::create(ul, ConstantExpr::create(index + i, Int32));
  }
  return ConcatExpr::createN(nbytes, bvals);
}


klee::ref<Expr> StrExpr::createValueRead(const Array *array, Expr::Width w) {
  UpdateList ul(array, 0);

  switch (w) {
  default: assert(0 && "invalid width");
  case Expr::Bool: 
    return ZExtExpr::create(ReadExpr::create(ul, ConstantExpr::alloc(0, Expr::Int32)),
                            Expr::Bool);
  case Expr::Int8: 
    return ReadExpr::create(ul, ConstantExpr::alloc(0,Expr::Int32));
  case Expr::Int16: 
    return ConcatExpr::create(ReadExpr::create(ul, ConstantExpr::alloc(0,Expr::Int32)),
                              ReadExpr::create(ul, ConstantExpr::alloc(1,Expr::Int32)));
  case Expr::Int32: 
    return ConcatExpr::create4(ReadExpr::create(ul, ConstantExpr::alloc(0,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(1,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(2,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(3,Expr::Int32)));
  case Expr::Int64: 
    return ConcatExpr::create8(ReadExpr::create(ul, ConstantExpr::alloc(0,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(1,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(2,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(3,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(4,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(5,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(6,Expr::Int32)),
                               ReadExpr::create(ul, ConstantExpr::alloc(7,Expr::Int32)));
  }
}


klee::ref<Expr> StrExpr::createLongEq(klee::ref<Expr> e1, klee::ref<Expr> e2, unsigned size, 
				bool is_eq) {
  
  if (size <= 8) {
    klee::ref<Expr> e = is_eq ? EqExpr::create(e1, e2) : NeExpr::create(e1, e2);
    return is_eq ? EqExpr::create(e1, e2) : NeExpr::create(e1, e2);
  }
  
  klee::ref<Expr> v = ConstantExpr::create(is_eq ? 1 : 0, Expr::Bool);
  unsigned i;
  for (i = 0; i < size / 8; i++) {
    int k = i * 64;
    klee::ref<Expr> p1 = ExtractExpr::create(e1, k, 64);
    klee::ref<Expr> p2 = ExtractExpr::create(e2, k, 64);
    if (is_eq) 
      v = AndExpr::create(v,  EqExpr::create(p1, p2));
    else
      v = OrExpr::create(v, NeExpr::create(p1, p2));
  }

  int j = size % 8;
  if (j != 0) {
    klee::ref<Expr> p1 = ExtractExpr::create(e1, i * 64, j * 8);
    klee::ref<Expr> p2 = ExtractExpr::create(e2, i * 64, j * 8);
    if (is_eq) 
      v = AndExpr::create(v, EqExpr::create(p1, p2));
    else
      v = OrExpr::create(v, NeExpr::create(p1, p2));
  }
  
  return v;
}


/*******************************************************************
 Convert string expressions to usual expressions
 ******************************************************************/

/*******************    StrDataExpr    ****************/


klee::ref<Expr> StrDataExpr::makeLengthConstraint(StrConstraint &constr) {
  if (isConstant()) 
    return ConstantExpr::create(value.length(), Expr::Int32);
  else if (getMin() == getMax()) {
    return ConstantExpr::create(getMin(), Expr::Int32);
  }
  else {
    klee::ref<Expr> vlenExpr = createLengthExpr(array->lenvar);
    constr.addSymLen(array->lenvar);
    
    if (!(array->withDependentLength)) {
      klee::ref<Expr> e1 = UgeExpr::create(vlenExpr, ConstantExpr::create(getMin(), Expr::Int32));
      klee::ref<Expr> e2 = UleExpr::create(vlenExpr, ConstantExpr::create(getMax(), Expr::Int32));
      constr.addLenConstraint(e1);
      constr.addLenConstraint(e2);
    }
    
    return vlenExpr;
  }
}

klee::ref<Expr> StrDataExpr::resolveStrExpr(StrConstraint &constr, TravPurpose pur) {
  constr.involveStr = true;

  if (pur == MakeStrLen) 
    return makeLengthConstraint(constr);
  
  // convert the str expressions
  
  if (isConstant()) {
    klee::ref<Expr> arrExpr;
    // printf(" value->size = %d \n", value.size());
    for (unsigned i = 0; i < value.size(); i++) {
      if (i == 0)
	arrExpr = ConstantExpr::create(value[i], Expr::Int8);
      else
	arrExpr = ConcatExpr::create(arrExpr, ConstantExpr::create(value[i], Expr::Int8));
    }
    return arrExpr;
  }
  else if (array->size > 0) {
    klee::ref<Expr> bvals[255];  // byte values
    unsigned index = 0;
    UpdateList ul(array, 0);
    for (unsigned i = 0; i < array->size; i++) {
      bvals[index++] = ReadExpr::create(ul, ConstantExpr::create(i, Int32));
    }
    klee::ref<Expr> v = ConcatExpr::createN(array->size, bvals);
    return v;
  }
  else {  // empty string
    return klee::ref<Expr>(this);
  }
}


klee::ref<Expr> StrDataExpr::makeAliasConstraint(StrConstraint& constr, klee::ref<Expr>& e) {
  return EqExpr::create(klee::ref<Expr>(this), e);
}


/*******************    FindLastOf    *******************/


klee::ref<Expr> StrFindLastOfExpr::makeLengthConstraint(StrConstraint &constr) {
  klee::ref<Expr> len = alias;
  constr.addSymLen(array_alias);
  klee::ref<Expr> str_len = str->resolveStrExpr(constr, MakeStrLen);

  klee::ref<Expr> len_constr = 
    OrExpr::create(UltExpr::create(len, str_len),
		   EqExpr::create(len, ConstantExpr::create(npos, Int32)));
  constr.addLenConstraint(len_constr);
  return len;
}


klee::ref<Expr> StrFindLastOfExpr::resolveStrExpr(StrConstraint& constr, TravPurpose pur) {
  constr.involveStr = true;

  if (pur == MakeStrLen) 
    return makeLengthConstraint(constr);

  // convert the str expression

  assert(isa<StrDataExpr>(str) && "FindLastOf: the string element must be a single string!");
  klee::ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);

  unsigned len = str_exp->array->size;

  klee::ref<Expr> found = ConstantExpr::create(0, Bool);
  klee::ref<Expr> v = ConstantExpr::create(len - 1, Int32);
  UpdateList ul(str_exp->array, 0);
  for (unsigned i = len; i != 0; i--) {
    // found = (operator[](i-1) == c) || found;
    klee::ref<Expr> e = ReadExpr::create(ul, ConstantExpr::create(i-1, Int32));
    found = OrExpr::create(found, EqExpr::create(e, tofind));
    // v -= !found;
    v = SubExpr::create(v, ZExtExpr::create(NotExpr::create(found), Int32));
  }
  return v;
}


klee::ref<Expr> StrFindLastOfExpr::makeAliasConstraint(StrConstraint& constr, klee::ref<Expr>& e) {
#ifdef FLA_DEBUG
  printf("\n StrFindLastOf::makeAliasConstraint \n");
#endif

  klee::ref<ConstantExpr> e_val;
  if (!constr.getValue(e, e_val))
    assert(0 && "FindLastOf: the index must have a concrete value!");
  unsigned index = e_val->getZExtValue();

  klee::ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);
  unsigned len = str_exp->getSize();
  assert((index == npos || index < len) && "FindLastOf: wrong index or length");
  
  UpdateList ul(str_exp->array, 0);
  klee::ref<Expr> v;

  if (index != npos) {
    klee::ref<Expr> index_exp = ConstantExpr::create(index, Int32);
    v = EqExpr::create(tofind, ReadExpr::create(ul, index_exp));
  }
  else {
    v = ConstantExpr::create(1, Bool);
  }

  for (unsigned i = index+1; i < len; i++) {
    klee::ref<Expr> c_exp = ReadExpr::create(ul, ConstantExpr::create(i, Int32));
    v = AndExpr::create(v, NeExpr::create(c_exp, tofind));
  }

//   printf("\n v = ");
//   v->dump();
  return v;
}


/*******************    Length    *******************/


klee::ref<Expr> StrLenExpr::resolveStrExpr(StrConstraint & constr, TravPurpose pur) {
  // return the length expression
  constr.involveStr = true;
  return str->resolveStrExpr(constr, MakeStrLen);
}


/*******************    Substr    *******************/

klee::ref<Expr> StrSubStrExpr::makeLengthConstraint(StrConstraint &constr) {
  klee::ref<Expr> str_len = str->resolveStrExpr(constr, MakeStrLen);
  klee::ref<Expr> pos_exp = pos->resolveStrExpr(constr, MakeStrLen);
  klee::ref<Expr> len_exp = len->resolveStrExpr(constr, MakeStrLen);
  if (len_exp == ConstantExpr::create(npos, Int32)) {
    return SubExpr::create(str_len, pos_exp);
  }
  else {
    // no bitvector overflowing
    constr.addLenConstraint(AndExpr::create(UleExpr::create(pos_exp, str_len), 
					    UleExpr::create(len_exp, str_len)));

    klee::ref<Expr> e = UleExpr::create(AddExpr::create(pos_exp, len_exp), str_len);
    constr.addLenConstraint(e);
    return len_exp;
  }
}


klee::ref<Expr> StrSubStrExpr::resolveStrExpr(StrConstraint & constr, TravPurpose pur) {
  
  constr.involveStr = true;

  if (pur == MakeStrLen) 
    return makeLengthConstraint(constr);

  // convert the str expression

  klee::ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);
  unsigned str_len_val = str_exp->array->size;
  // printf("len = %u \n", len_val);

  klee::ref<ConstantExpr> pos_exp;
  if (!constr.getValue(pos, pos_exp))
    assert(0 && "the pos must have a concrete value!");
  unsigned pos_val = pos_exp->getZExtValue();

  klee::ref<ConstantExpr> len_exp;
  if (!constr.getValue(len, len_exp))
    assert(0 && "the pos must have a concrete value!");
  unsigned len_val = len_exp->getZExtValue();

  // printf("pos = %u \n", pos_val);
  unsigned size = len_val == npos ? str_len_val - pos_val : len_val;

  if (size > 0) {
    klee::ref<Expr> bvals[255];  // byte values

    UpdateList ul(str_exp->array, 0);
    for (unsigned i = 0; i < size; i++) {
      bvals[i] = ReadExpr::create(ul, ConstantExpr::create(pos_val + i, Int32));
    }
    klee::ref<Expr> v = ConcatExpr::createN(size, bvals);
    //   printf("v = ");
    //   v->dump();
    return v;
  }
  else {  // empty string
    return klee::ref<Expr>(alias);
  }
}


klee::ref<Expr> StrSubStrExpr::makeAliasConstraint(StrConstraint& constr, klee::ref<Expr>& e) {
#ifdef FLA_DEBUG
  printf("\n StrSubStrExpr::makeAliasConstraint \n");
#endif

  klee::ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);
  unsigned str_len_val = str_exp->array->size;
  // printf("str_len = %u \n", str_len_val);

  klee::ref<StrDataExpr> substr_exp = cast<StrDataExpr>(e);
  // printf("substr_len = %u \n", substr_exp->array->size);

  klee::ref<ConstantExpr> pos_exp;
  if (!constr.getValue(pos, pos_exp))
    assert(0 && "StrSubStrExpr: the pos must have a concrete value!");
  unsigned pos_val = pos_exp->getZExtValue();
  // printf("pos = %u \n", pos_val);

  klee::ref<ConstantExpr> len_exp;
  if (!constr.getValue(len, len_exp))
    assert(0 && "the pos must have a concrete value!");
  unsigned len_val = len_exp->getZExtValue();
  // printf("len = %u \n", len_val);

  unsigned size = len_val == npos ? str_len_val - pos_val : len_val;
  
  assert(pos_val + size <= str_len_val 
	 && "StrSubStrExpr: wrong pos or length!");

  if (size == 0)   // empty string
    return ConstantExpr::create(1, Bool);

  klee::ref<Expr> bvals[255];  // byte values
  klee::ref<Expr> sub_bvals[255];  // byte values

  UpdateList ul(str_exp->array, 0);
  UpdateList sub_ul(substr_exp->array, 0);
  for (unsigned i = 0; i < size; i++) {
    bvals[i] = ReadExpr::create(ul, ConstantExpr::create(pos_val + i, Int32));
    sub_bvals[i] = ReadExpr::create(sub_ul, 
				    ConstantExpr::create(i, Int32));
  }

  klee::ref<Expr> v = ConcatExpr::createN(size, bvals);
  klee::ref<Expr> sub_v = ConcatExpr::createN(size, sub_bvals);
  klee::ref<Expr> eq_v = createLongEq(v, sub_v, size);
  return eq_v;
}


/*******************    Find    *******************/


klee::ref<Expr> StrFindExpr::resolveStrExpr(StrConstraint& constr, TravPurpose pur) {

  constr.involveStr = true;

  if (pur == MakeStrLen) {
    klee::ref<Expr> _pos = alias;
    constr.addSymLen(array_alias);
    
    klee::ref<Expr> str_len = str->resolveStrExpr(constr, MakeStrLen);
    klee::ref<Expr> tofind_len = tofind->resolveStrExpr(constr, MakeStrLen);

    // pos == -1 \/ pos + tofind_len <= str_len
    klee::ref<Expr> not_npos = 
      AndExpr::create(UleExpr::create(_pos, str_len), // prevent bit-vector over-flowing
		      UleExpr::create(AddExpr::create(_pos, tofind_len), str_len));
    klee::ref<Expr> len_constr = 
      OrExpr::create(not_npos,
		     EqExpr::create(_pos, ConstantExpr::create(npos, Int32)));

    constr.addLenConstraint(len_constr);
    return _pos;
  }

  assert(0 && "StrFindExpr::resolveStrExpr: unimplemented!");
}


klee::ref<Expr> StrFindExpr::makeAliasConstraint(StrConstraint& constr, klee::ref<Expr>& e) {
#ifdef FLA_DEBUG
  printf("\n StrFind::makeAliasConstraint \n");
#endif

  constr.involveStr = true;

  klee::ref<ConstantExpr> e_val;
  if (!constr.getValue(e, e_val))
    assert(0 && "Find: the index must have a concrete value!");
  unsigned index = e_val->getZExtValue();
  // printf("index = %u \n", index);

  klee::ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);
  unsigned str_len = str_exp->getSize();
  // printf("str_len = %u \n", str_len);
  unsigned tofind_len = cast<StrDataExpr>(tofind)->getSize();
  // printf("tofind_len = %u \n", tofind_len);

  assert((index == npos || index + tofind_len <= str_len) 
	 && "Find: wrong index or length");
  
  if (tofind_len == 0) { // empty string
    return ConstantExpr::create(1, Bool);
  }

  klee::ref<Expr> tofind_exp = tofind->resolveStrExpr(constr, MakeStrExpr);
  klee::ref<Expr> v;

  // determined by the contents of the strings
  bool b = index == npos;
  if (b) {
    v = ConstantExpr::create(1, Bool);
    if (str_len < tofind_len)    // no need to check since the length relation is unsatisfiable
      return v;
  }
  else {
    v = createLongEq(tofind_exp, 
		     createMultRead(str_exp->array, index, tofind_len), tofind_len);
  }

  // forall j < index: str[j, j + tofind_len] != tofind
  for (unsigned i = 0; i < (b ? str_len - tofind_len + 1 : index); i++) {
    klee::ref<Expr> s_exp = createMultRead(str_exp->array, i, tofind_len);
    v = AndExpr::create(v, createLongEq(tofind_exp, s_exp, tofind_len, false));
  }
  return v;
}


/*******************    Compare    *******************/


klee::ref<Expr> StrCompareExpr::resolveStrExpr(StrConstraint &constr, TravPurpose pur) {
  constr.involveStr = true;

  if (pur == MakeStrLen) {
    klee::ref<Expr> _res = alias;
    constr.addSymLen(array_alias);
    
    klee::ref<Expr> str_len = str->resolveStrExpr(constr, MakeStrLen);
    klee::ref<Expr> tocompare_len = tocompare->resolveStrExpr(constr, MakeStrLen);
    // (res == 0 ==> (tocompare_len == len)) &&
    // (pos + len <= str_len) 
    constr.addLenConstraint(UleExpr::create(AddExpr::create(pos, len), str_len));
    klee::ref<Expr> len_constr = 
      OrExpr::create(NeExpr::create(_res, ConstantExpr::create(0, Int32)), 
		     EqExpr::create(tocompare_len, len));
    constr.addLenConstraint(len_constr);

    return _res;
  }

  return ConstantExpr::create(1, Int32);
}


klee::ref<Expr> StrCompareExpr::makeAliasConstraint(StrConstraint& constr, klee::ref<Expr>& res) {
#ifdef FLA_DEBUG
  printf("\n StrCompareExpr::makeAliasConstraint \n");
#endif

  klee::ref<StrDataExpr> str_exp = cast<StrDataExpr>(str);

  klee::ref<StrDataExpr> tocompare_exp = cast<StrDataExpr>(tocompare);
  unsigned tocompare_len = tocompare_exp->getSize();

  klee::ref<ConstantExpr> pos_exp;
  if (!constr.getValue(pos, pos_exp))
    assert(0 && "StrCompare: the pos must have a concrete value!");
//   unsigned pos_val = pos_exp->getZExtValue();
  // printf("pos = %u \n", pos_val);

  klee::ref<ConstantExpr> len_exp;
  if (!constr.getValue(len, len_exp))
    assert(0 && "StrCompare: the len must have a concrete value!");
  unsigned len_val = len_exp->getZExtValue();
  // printf("len = %u \n", len_val);

  klee::ref<ConstantExpr> res_exp;
  if (!constr.getValue(res, res_exp))
    assert(0 && "StrCompare: the result must have a concrete value!");
  unsigned res_val = res_exp->getZExtValue();
  // printf("res = %u \n", res_val);

  assert((res_val || len_val == tocompare_len) &&
	 "StrCompare: the lengths are incorrect!");

  if (len_val == 0) {    //empty string
    return ConstantExpr::create(1, Bool);
  }

  if (res_val == 0) {  // equal
    klee::ref<Expr> str_v = createMultRead(str_exp->array, 0, len_val);
    klee::ref<Expr> tocompare_v = tocompare_exp->resolveStrExpr(constr, MakeStrExpr);
    return createLongEq(str_v, tocompare_v, len_val);
  }
  else if (len_val == tocompare_len) { // the lengths are equal; but the contents are different
    klee::ref<Expr> str_v = createMultRead(str_exp->array, 0, len_val);
    klee::ref<Expr> tocompare_v = tocompare_exp->resolveStrExpr(constr, MakeStrExpr);
    return createLongEq(str_v, tocompare_v, len_val, false);
  }
  else  // the lengths are different; no need to compare the contents
    return NeExpr::create(res, ConstantExpr::create(0, Int32));
}


/*********************     ==    *********************/


klee::ref<Expr> StrEqExpr::resolveStrExpr(StrConstraint& constr, TravPurpose pur) {

  constr.involveStr = true;

  if (pur == MakeStrLen) {
    klee::ref<Expr> left_len = left->resolveStrExpr(constr, MakeStrLen);
    klee::ref<Expr> right_len = right->resolveStrExpr(constr, MakeStrLen);
    return ZExtExpr::create(EqExpr::create(left_len, right_len), Int8);
  }
  else {
    klee::ref<Expr> e = EqExpr::create(left->resolveStrExpr(constr, pur),
				 right->resolveStrExpr(constr, pur));
    return ZExtExpr::create(e, Int8);
  }
}


klee::ref<Expr> StrEqExpr::makeAliasConstraint(StrConstraint& constr, klee::ref<Expr>& res) {
#ifdef FLA_DEBUG
  printf("\n StrEqExpr::makeAliasConstraint \n");
#endif

  klee::ref<StrDataExpr> left_exp = cast<StrDataExpr>(left);
  unsigned left_len = left_exp->getSize();
  // printf("left_len = %u \n", left_len);

  klee::ref<StrDataExpr> right_exp = cast<StrDataExpr>(right);
  unsigned right_len = right_exp->getSize();
  // printf("right_len = %u \n", right_len);

//   klee::ref<ConstantExpr> res_exp;
//   if (!constr.getValue(res, res_exp))
//     assert(0 && "StrEq: the result must have a concrete value!");
//   unsigned res_val = res_exp->getZExtValue();
  // printf("res = %u \n", res_val);

  if (left_len != right_len) {  // no need to check the contents
    return EqExpr::create(res, ConstantExpr::create(0, Int8));
  }

  if (left_len == 0) {  // empty string
     return EqExpr::create(res, ConstantExpr::create(1, Int8));
  }

  klee::ref<Expr> left_v = left->resolveStrExpr(constr, MakeStrExpr);
  klee::ref<Expr> right_v = right->resolveStrExpr(constr, MakeStrExpr);
  klee::ref<Expr> cmp_v = createLongEq(left_v, right_v, left_len);
  klee::ref<Expr> v = ZExtExpr::create(cmp_v, Int8);
  return EqExpr::create(v, res);
}
