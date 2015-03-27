//===-- ExprHashMap.h -------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_EXPRHASHMAP_H
#define KLEE_EXPRHASHMAP_H

#include "klee/Expr.h"
#include <tr1/unordered_map>
#include <tr1/unordered_set>

namespace klee {

  namespace util {
    struct ExprHash  {
      unsigned operator()(const klee::ref<Expr> e) const {
        return e->hash();
      }
    };
    
    struct ExprCmp {
      bool operator()(const klee::ref<Expr> &a, const klee::ref<Expr> &b) const {
        return a==b;
      }
    };
  }
  
  template<class T> 
  class ExprHashMap : 

    public std::tr1::unordered_map<klee::ref<Expr>,
				   T,
				   klee::util::ExprHash,
				   klee::util::ExprCmp> {
  };
  
  typedef std::tr1::unordered_set<klee::ref<Expr>,
				  klee::util::ExprHash,
				  klee::util::ExprCmp> ExprHashSet;

}

#endif
