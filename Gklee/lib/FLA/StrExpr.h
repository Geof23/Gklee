//===-- StrExpr.h --------------------------------------------------*- C++ -*-===//
//
//                     KLEE++ by FLA
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef KLEE_STREXPR_H
#define KLEE_STREXPR_H

#include "klee/Expr.h"
#include <sstream>
#include "llvm/ADT/StringExtras.h"

// ****************************************************************
// By FLA
// the IR for strings
// ****************************************************************


namespace klee {


#define KLEE_INFO llvm::errs() << "KLEE++: "
#define KLEE_INFO2 llvm::errs() << "\nKLEE++: "

  // Enabling flag FLA_DEBUG will print more detailed info for debugging

// #define FLA_DEBUG

//////////////////////////////////////////////////////////////////////


/*******************************************************************
 A symbolic variable representing the length of a var-len array 
 ******************************************************************/

class LengthVar : public Array {

protected:
  Array* arr;

public:
  unsigned value;

public:

  LengthVar(const std::string &_name, uint64_t _size, 
	    Array* _arr,   // _arr is the associated var-len array
	    const ref<ConstantExpr> *constantValuesBegin = 0,
	    const ref<ConstantExpr> *constantValuesEnd = 0)
    : Array(_name, 
	    _size, 
	    constantValuesBegin, 
	    constantValuesEnd) {
    arr = _arr;
    kind = LenVarKind;
    name = _name + ".len";
  };

  ~LengthVar();

  // specify the length of the associated var-len array
  void setArrayLength(unsigned _len) {
    arr->size = _len;
  }

  Array* getLinkedArray() {   // get the associated var-len array
    return arr;
  }

  static bool classof(const Array *A) {
    return A->kind == Array::LenVarKind;
  }
  static bool classof(const LengthVar*) { return true; }

};


/*******************************************************************
 Arrays with variant lengths, used to model the values of strings.
 Values min and max give the mimimum and maximum lengths of the array 
 respectively. 
 ******************************************************************/

class VarLenArray : public Array {

public:
  unsigned min;
  unsigned max;

  LengthVar* lenvar;   // the variable for the (symbolic) length of the string

  bool withDependentLength;   // the length is dependent of another expression
                              // if yes, then min and max are ignored 

public:

  VarLenArray(const std::string &_name, uint64_t _min, uint64_t _max, 
	      const bool _withDependentLength = false,
	      const ref<ConstantExpr> *constantValuesBegin = 0,
	      const ref<ConstantExpr> *constantValuesEnd = 0)
    : Array(_name, 
	    _max, // the initial value is set to be maximum
	    constantValuesBegin, 
	    constantValuesEnd) {
    kind = VarLenKind;
    min = _min;
    max = _max;
    withDependentLength = _withDependentLength;

    std::ostringstream buf;
    buf << _name;
    if (!withDependentLength) 
      buf << "[" << _min << "," << _max << "]";       // Format: name[min,max]
    name = buf.str();

    lenvar = new LengthVar(name, 1, this);

//     if (min < max)
//       lenvar = new LengthVar(array->name, 1, this);
  };


  ~VarLenArray();

//   bool isSymbolicArray() const { return constantValues.empty(); }
//   bool isConstantArray() const { return !isSymbolicArray(); }

  static bool classof(const Array *A) {
    return A->kind == Array::VarLenKind;
  }
  static bool classof(const VarLenArray*) { return true; }

};


/*******************************************************************
 The basic class for string constraints 
 Defined in "StrConstraint.h"
 ******************************************************************/

  class StrConstraint;


/*******************************************************************
 The basic class for string expressions 
 It is a "virtual" class such that no objects should be created
 ******************************************************************/

class StrExpr : public Expr {

public:

  static const unsigned npos = 0xffffffff;   // the npos value in the string library
                                             // obviously we assume 32-bit words

  static unsigned len_id;     // for creating a new interger variable
  static unsigned str_id;     // for creating a new string variable
  
  Array* array_alias;         // the array associated with this expression;
                              // for creating read expressions

  // read multiple bytes from an array
  static ref<Expr> createMultRead(const Array *array, unsigned index, unsigned nbytes);
  static ref<Expr> createValueRead(const Array *array, Expr::Width w);

  // create a new variable which usually records the length of a string 
  static ref<Expr> createLengthExpr(const Array *array) {
    UpdateList ul(array, 0);
    ref<Expr> e = ReadExpr::create(ul, ConstantExpr::alloc(0, Expr::Int32));
    return SExtExpr::create(e, Int32);
  }

  // KLEE seems to have problems in comparing bit-vectors longer than 8 bytes;
  // here we divide the bit-vectors into chuncks of 8 bytes and then compare them
  static ref<Expr> createLongEq(ref<Expr> e1, ref<Expr> e2, unsigned size, 
				bool is_eq = true);


public:

  ~StrExpr() {};

  enum SubKind {     // types of the string operations
    StrData = 0,
    StrAlias,
    StrLen,
    StrEq,
    StrNe,
    StrFindLastOf,
    StrFind,
    StrCompare,
    StrSubStr
  };

  virtual SubKind getSubKind() const = 0;

  // static const Kind kind = String;

  Kind getKind() const { return String; }

  unsigned hashStr (const std::string& str) const
  {
    unsigned h = str[0];
    for (int i = 1; str[i] != 0; ++i)
      h = (h << 4) + str[i];
    return h % 1000; // remainder
  }

  /// (Re)computes the hash of the current expression.
  /// Returns the hash value. 
  unsigned computeHash() { return hash(); }

  int compareStrContents(const StrExpr* e) const {
    return 0;
  };

  int compareContents(const Expr &b) const {
    const StrExpr* e = (const StrExpr*) (&b);
    SubKind ak = getSubKind(), bk = e->getSubKind();
    if (ak != bk)   // compare the subkind
      return (ak < bk) ? -1 : 1;
    else
      return compareStrContents(e);
  };


public:
  static bool classof(const Expr *E) {
    return (E->getKind() == Expr::String);
  }
  static bool classof(const StrExpr *) { return true; }

  // virtual ref<Expr> resolveStrExpr(StrConstraint & constr, TravPurpose pur);

  // traverse a string expressions and generate constraints
  virtual ref<Expr> makeAliasConstraint(StrConstraint& constr, ref<Expr>& e) {
    return EqExpr::create(ref<Expr>(this), e);
  }

};  // end StrExpr

/*******************************************************************
 A class recording the alias of an expression 
 (it has the same semantics as EqStr) 
 ******************************************************************/

class StrAliasExpr : public StrExpr { 
public:
  static const SubKind subkind = StrAlias;
  static const unsigned numKids = 2;
  
  ref<Expr> left, right;

public:  
  StrAliasExpr(const ref<Expr> &l,
	       const ref<Expr> &r) : left(l), right(r) {};

  static ref<Expr> alloc(const ref<Expr> &l, const ref<Expr> &r) {
    ref<Expr> e(new StrAliasExpr(l, r));
    e->computeHash();
    return e;
  }

  static ref<Expr> create(const ref<Expr> &l, const ref<Expr> &r) {
    return alloc(l, r);
  }
  
   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return left->hash() * getSubKind() + right->hash(); }

  void print(std::ostream &os) const {
    left->print(os);
    os << " = ";
    right->print(os);
  }

  Width getWidth() const { return Expr::Bool; }
  SubKind getSubKind() const { return StrAlias; }

  unsigned getNumKids() const { return numKids; }
  ref<Expr> getKid(unsigned i) const {
    switch (i) {
    case 0:
      return left;
    default:
      return right;
    }
  }

  ref<Expr> rebuild(ref<Expr> kids[]) const { 
    return alloc(kids[0], kids[1]);
  }

public:
  static bool classof(const Expr *E) {
    if (E->getKind() != Expr::String)
      return false;
    return ((const StrExpr*) E)->getSubKind() == StrAlias;
  }
  static bool classof(const StrAliasExpr *) { return true; }

  bool isSymbolic() { return left->isSymbolic() || right->isSymbolic(); };
  
  // Invoke the functions according to the purpose
  ref<Expr> resolveStrExpr(StrConstraint & constr, TravPurpose pur) {
    switch (pur) {
    case MakeStrLen: {  // for the lengths of the strings
      ref<Expr> e = EqExpr::create(left, right);
      return e->resolveStrExpr(constr, pur);
    }
    default: {    // for the contents in the strings
      if (isa<StrExpr>(right)) {
	ref<StrExpr> e = cast<StrExpr>(right);
	return e->makeAliasConstraint(constr, left);  // the function doing the real things!
      }
      else {
	ref<Expr> e = EqExpr::create(left, right);
	return e->resolveStrExpr(constr, pur);
      }
    }
    } // end switch
  }
};


/*******************************************************************
 The class for single string data
 ******************************************************************/

class StrDataExpr : public StrExpr {

public:

  static const SubKind subkind = StrData;
  static const unsigned numKids = 0;

  std::string value;               // for constant values
  VarLenArray* array;              // for unknown/symbolic values

private:

public:
  StrDataExpr(const std::string &_value, VarLenArray* _array = NULL) : 
    value(_value), array(_array) {}

  StrDataExpr(VarLenArray* _array) : 
    value("unset"), array(_array) {}
  
  // Width getWidth() const { return value.length(); }
  Width getWidth() const { return Expr::Int32; }

  SubKind getSubKind() const { return StrData; }
  unsigned getNumKids() const { return 0; }
  ref<Expr> getKid(unsigned i) const { return 0; }

  void print(std::ostream &os) const {
    if (array != NULL)
      os << array->name;
    else
      os << "'" << value << "'";
  }

   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return hashStr(value); }  

  // Given an array of new kids return a copy of the expression
  // but using those children. 
  ref<Expr> rebuild(ref<Expr> kids[/* getNumKids() */]) const {
    assert(0 && "rebuild() on StrDataExpr"); 
    return (Expr*) this;
  };

  // compare with another StrDataExpr
  int compareStrContents(const StrExpr* expr) {
    const StrDataExpr* e = (const StrDataExpr*) expr;
    if (array != NULL) // compare the array pointers
      return array - e->array;
    else    // compare the values
      return value.compare(e->value);
  };

  VarLenArray* getArray() { return array; }

  bool isSymbolic() { 
    return array != NULL; 
    // return array->isSymbolicArray();
  };

  bool isConstant() { return array == NULL; };
  int getMin () { return array->min; }
  int getMax () { return array->max; }
  int getSize () { return array == NULL ? value.length() : array->size; }

  ref<Expr> makeLengthConstraint(StrConstraint &constr);
  ref<Expr> resolveStrExpr(StrConstraint &constr, TravPurpose pur);

  ref<Expr> makeAliasConstraint(StrConstraint& constr, ref<Expr>& e);

};  // end StrExpr


/*******************************************************************
 The class for the length of a string
 ******************************************************************/

class StrLenExpr : public StrExpr { 
public:
  static const SubKind subkind = StrLen;
  static const unsigned numKids = 1;
  
  ref<Expr> str;

public:  
  StrLenExpr(const ref<Expr> &_str) : str(_str) {}

  static ref<Expr> alloc(const ref<Expr> &_str) {
    ref<Expr> r(new StrLenExpr(_str));
    r->computeHash();
    return r;
  }
  
   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return str->hash() * getSubKind(); }

  void print(std::ostream &os) const {
    str->print(os);
    os << ".len()";
  }

  static ref<Expr> create(const ref<Expr> &e) {
    return alloc(e);
  }

  Width getWidth() const { return Expr::Int32; }
  SubKind getSubKind() const { return StrLen; }

  unsigned getNumKids() const { return numKids; }
  ref<Expr> getKid(unsigned i) const { return str; }

  ref<Expr> rebuild(ref<Expr> kids[]) const { 
    return create(kids[0]);
  }

public:
  static bool classof(const Expr *E) {
    if (E->getKind() != Expr::String)
      return false;
    return ((const StrExpr*) E)->getSubKind() == StrLen; 
  }
  static bool classof(const StrLenExpr *) { return true; }

  bool isSymbolic() { return str->isSymbolic(); };

  ref<Expr> resolveStrExpr(StrConstraint & constr, TravPurpose pur);

};


/*******************************************************************
 The class for the "find_last_of" operation
 ******************************************************************/

class StrFindLastOfExpr : public StrExpr { 
public:
  static const SubKind subkind = StrFindLastOf;
  static const unsigned numKids = 3;
  
  ref<Expr> str, tofind, pos;

public:  
  StrFindLastOfExpr(const ref<Expr> &_str, const ref<Expr> &_tofind,
		    const ref<Expr>& _pos = ConstantExpr::create(0, Int32)) : 
    str(_str), tofind(_tofind), pos(_pos) {}

  static ref<Expr> alloc(const ref<Expr> &_str, const ref<Expr> &_tofind,
			 const ref<Expr> _pos = ConstantExpr::create(0, Int32)) {
    ref<Expr> r(new StrFindLastOfExpr(_str, _tofind, _pos));
    r->computeHash();
    return r;
  }

  static ref<Expr> create(const ref<Expr> &e1, const ref<Expr> &e2, const ref<Expr> &e3) {
    return alloc(e1, e2, e3);
  }
  
   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return str->hash() * getSubKind(); }

  void print(std::ostream &os) const {
    str->print(os);
    os << ".find_last_of(";
    tofind->print(os);
//     os << ",";
//     pos->print(os);
    os << ")";
  }

  Width getWidth() const { return Expr::Int32; }
  SubKind getSubKind() const { return StrFindLastOf; }

  unsigned getNumKids() const { return numKids; }
  ref<Expr> getKid(unsigned i) const {
    switch (i) {
    case 0:
      return str;
    case 1:
      return tofind;
    default:
      return pos;
    }
  }

  ref<Expr> rebuild(ref<Expr> kids[]) const { 
    return alloc(kids[0], kids[1], kids[2]);
  }

public:
  static bool classof(const Expr *E) {
    if (E->getKind() != Expr::String)
      return false;
    return ((const StrExpr*) E)->getSubKind() == StrFindLastOf;
  }
  static bool classof(const StrFindLastOfExpr *) { return true; }

  bool isSymbolic() { return str->isSymbolic() || tofind->isSymbolic(); };
  
  ref<Expr> makeLengthConstraint(StrConstraint &constr);
  ref<Expr> resolveStrExpr(StrConstraint & constr, TravPurpose pur);
  ref<Expr> makeAliasConstraint(StrConstraint& constr, ref<Expr>& e);

};


/*******************************************************************
 The class for the "substr" operation
 ******************************************************************/

class StrSubStrExpr : public StrExpr { 
public:
  static const SubKind subkind = StrSubStr;
  static const unsigned numKids = 3;
  
  ref<Expr> str, pos, len;

public:  
  StrSubStrExpr(const ref<Expr> &_str, const ref<Expr> &_pos,
		const ref<Expr> _len = ConstantExpr::create(npos, Int32)) : 
    str(_str), pos(_pos), len(_len) {
  }

  static ref<Expr> alloc(const ref<Expr> &_str, const ref<Expr> &_pos,
			 const ref<Expr> _len = ConstantExpr::create(npos, Int32)) {
    ref<Expr> r(new StrSubStrExpr(_str, _pos, _len));
    r->computeHash();
    return r;
  }

  static ref<Expr> create(const ref<Expr> &e1, const ref<Expr> &e2, const ref<Expr> &e3) {
    return alloc(e1, e2, e3);
  }
  
   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return str->hash() * getSubKind(); }

  void print(std::ostream &os) const {
    str->print(os);
    os << ".substr(";
    pos->print(os);
    if (isa<ConstantExpr>(len)) {
      ref<ConstantExpr> e = cast<ConstantExpr>(len);
      if (e->getZExtValue() != npos)
	os << "," << len;
    }
    else
      os << "," << len;
    os << ")";
  }

  Width getWidth() const { return Expr::Int32; }
  SubKind getSubKind() const { return StrSubStr; }

  unsigned getNumKids() const { return numKids; }
  ref<Expr> getKid(unsigned i) const {
    switch (i) {
    case 0:
      return str;
    case 1:
      return pos;
    default:
      return len;
    }
  }

  ref<Expr> rebuild(ref<Expr> kids[]) const { 
    return alloc(kids[0], kids[1], kids[2]);
  }

public:
  static bool classof(const Expr *E) {
    if (E->getKind() != Expr::String)
      return false;
    return ((const StrExpr*) E)->getSubKind() == StrSubStr; 
  }
  static bool classof(const StrSubStrExpr *) { return true; }

  bool isSymbolic() { return str->isSymbolic(); };
  
  ref<Expr> makeLengthConstraint(StrConstraint &constr);
  ref<Expr> resolveStrExpr(StrConstraint &constr, TravPurpose pur);
  ref<Expr> makeAliasConstraint(StrConstraint& constr, ref<Expr>& e);

};



/*******************************************************************
 The class for the "find" operation
 ******************************************************************/

class StrFindExpr : public StrExpr { 
public:
  static const SubKind subkind = StrFindLastOf;
  static const unsigned numKids = 2;
  
  ref<Expr> str, tofind;

public:  
  StrFindExpr(const ref<Expr> &_str, const ref<Expr> &_tofind) : 
    str(_str), tofind(_tofind) {}

  static ref<Expr> alloc(const ref<Expr> &_str, const ref<Expr> &_tofind) {
    ref<Expr> r(new StrFindExpr(_str, _tofind));
    r->computeHash();
    return r;
  }
  static ref<Expr> create(const ref<Expr> &e1, const ref<Expr> &e2) {
    return alloc(e1, e2);
  }
  
   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return str->hash() * getSubKind(); }

  void print(std::ostream &os) const {
    str->print(os);
    os << ".find(";
    tofind->print(os);
    os << ")";
  }

  Width getWidth() const { return Expr::Int32; }
  SubKind getSubKind() const { return StrFind; }

  unsigned getNumKids() const { return numKids; }
  ref<Expr> getKid(unsigned i) const {
    switch (i) {
    case 0:
      return str;
    case 1:
      return tofind;
    default:
      return tofind;
    }
  }

  ref<Expr> rebuild(ref<Expr> kids[]) const 
  { return alloc(kids[0], kids[1]); }

public:
  static bool classof(const Expr *E) {
    if (E->getKind() != Expr::String)
      return false;
    return ((const StrExpr*) E)->getSubKind() == StrFind; 
  }
  static bool classof(const StrFindExpr *) { return true; }

  bool isSymbolic() { return str->isSymbolic() || tofind->isSymbolic(); };
  
  ref<Expr> resolveStrExpr(StrConstraint & constr, TravPurpose pur);
  ref<Expr> makeAliasConstraint(StrConstraint& constr, ref<Expr>& e);

};


/*******************************************************************
 The class for the "compare" operation
 ******************************************************************/

class StrCompareExpr : public StrExpr { 
public:
  static const SubKind subkind = StrCompare;
  static const unsigned numKids = 4;
  
  ref<Expr> str, pos, len, tocompare;

public:  
  StrCompareExpr(const ref<Expr> &_str, const ref<Expr> &_pos, 
		 const ref<Expr> &_len, const ref<Expr> &_tocompare) : 
    str(_str), pos(_pos), len(_len), tocompare(_tocompare) {}

  static ref<Expr> alloc(const ref<Expr> &_str, const ref<Expr> &_pos, 
			 const ref<Expr> &_len, const ref<Expr> &_tocompare) {
    ref<Expr> r(new StrCompareExpr(_str, _pos, _len, _tocompare));
    r->computeHash();
    return r;
  }
  static ref<Expr> create(const ref<Expr> &_str, const ref<Expr> &_pos, 
			  const ref<Expr> &_len, const ref<Expr> &_tocompare) {
    return alloc(_str, _pos, _len, _tocompare);
  }
  
   /// Returns the pre-computed hash of the current expression
  unsigned hash() const { return str->hash() * getSubKind() + tocompare->hash(); }

  void print(std::ostream &os) const {
    str->print(os);
    os << ".compare(";
    pos->print(os);
    os << ",";
    len->print(os);
    os << ",";
    tocompare->print(os);
    os << ")";
  }

  Width getWidth() const { return Expr::Int32; }
  SubKind getSubKind() const { return StrCompare; }

  unsigned getNumKids() const { return numKids; }
  ref<Expr> getKid(unsigned i) const {
    switch (i) {
    case 0:
      return str;
    case 1:
      return pos;
    case 2:
      return len;
    default:
      return tocompare;
    }
  }

  ref<Expr> rebuild(ref<Expr> kids[]) const { 
    return create(kids[0], kids[1], kids[2], kids[3]); 
  }

public:
  static bool classof(const Expr *E) {
    if (E->getKind() != Expr::String)
      return false;
    return ((const StrExpr*) E)->getSubKind() == StrCompare; 
  }
  static bool classof(const StrCompareExpr *) { return true; }

  bool isSymbolic() { return str->isSymbolic() || tocompare->isSymbolic(); };
  ref<Expr> resolveStrExpr(StrConstraint & constr, TravPurpose pur);
  ref<Expr> makeAliasConstraint(StrConstraint &constr, ref<Expr> &e);

};


/*******************************************************************
 The classes for binary string operations
 ******************************************************************/

#define STRING_EXPR_CLASS(_class_kind)                               \
class _class_kind ## Expr : public StrExpr {                         \
  ref<Expr> left;                                                    \
  ref<Expr> right;                                                   \
public:                                                              \
  static const SubKind subkind = _class_kind;                        \
  static const unsigned numKids = 2;                                 \
public:                                                              \
    _class_kind ## Expr(const ref<Expr> &l,                          \
                        const ref<Expr> &r) : left(l), right(r) {}   \
    static ref<Expr> alloc(const ref<Expr> &l, const ref<Expr> &r) { \
      ref<Expr> res(new _class_kind ## Expr (l, r));                 \
      res->computeHash();                                            \
      return res;                                                    \
    }                                                                \
    void print(std::ostream &os) const {                             \
      left->print(os);                                               \
      os << " == ";                                                  \
      right->print(os);                                              \
    }                                                                \
    static ref<Expr> create(const ref<Expr> &l, const ref<Expr> &r) {\
      return alloc(l, r);					     \
    };								     \
    Width getWidth() const { return Expr::Int8; }		     \
    SubKind getSubKind() const { return _class_kind; }               \
    virtual ref<Expr> rebuild(ref<Expr> kids[]) const {              \
      return create(kids[0], kids[1]);                               \
    }                                                                \
    unsigned getNumKids() const { return numKids; }                  \
    ref<Expr> getKid(unsigned i) const                               \
       { return i == 0 ? left : right; }			     \
    static bool classof(const Expr *E) {                             \
      return E->getKind() == Expr::String;                           \
    }                                                                \
    static bool classof(const _class_kind ## Expr *)  {              \
      return true;                                                   \
    }                                                                \
    unsigned hash() const {                                          \
      return (left->hash() + right->hash()) * getSubKind();          \
    }	                                                             \
    bool isSymbolic() {                                              \
      return (left->isSymbolic() || right->isSymbolic()); }	     \
    ref<Expr> resolveStrExpr(StrConstraint &constr, TravPurpose pur);\
    ref<Expr> makeAliasConstraint(StrConstraint &constr, ref<Expr> &e); \
};                                                                   \



  STRING_EXPR_CLASS(StrEq)             // s1 == s2


} // End klee namespace

#endif
