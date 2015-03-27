//===-- ExprBuilder.h -------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_EXPRBUILDER_H
#define KLEE_EXPRBUILDER_H

#include "Expr.h"

namespace klee {
  /// ExprBuilder - Base expression builder class.
  class ExprBuilder {
  protected:
    ExprBuilder();

  public:
    virtual ~ExprBuilder();

    // Expressions

    virtual klee::ref<Expr> Constant(const llvm::APInt &Value) = 0;
    virtual klee::ref<Expr> NotOptimized(const klee::ref<Expr> &Index) = 0;
    virtual klee::ref<Expr> Read(const UpdateList &Updates, 
                           const klee::ref<Expr> &Index) = 0;
    virtual klee::ref<Expr> Select(const klee::ref<Expr> &Cond,
                             const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Concat(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Extract(const klee::ref<Expr> &LHS, 
                              unsigned Offset, Expr::Width W) = 0;
    virtual klee::ref<Expr> ZExt(const klee::ref<Expr> &LHS, Expr::Width W) = 0;
    virtual klee::ref<Expr> SExt(const klee::ref<Expr> &LHS, Expr::Width W) = 0;
    virtual klee::ref<Expr> Add(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Sub(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Mul(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> UDiv(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> SDiv(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> URem(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> SRem(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Not(const klee::ref<Expr> &LHS) = 0;
    virtual klee::ref<Expr> And(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Or(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Xor(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Shl(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> LShr(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> AShr(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Eq(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Ne(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Ult(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Ule(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Ugt(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Uge(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Slt(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Sle(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Sgt(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;
    virtual klee::ref<Expr> Sge(const klee::ref<Expr> &LHS, const klee::ref<Expr> &RHS) = 0;

    // Utility functions

    klee::ref<Expr> False() { return ConstantExpr::alloc(0, Expr::Bool); }

    klee::ref<Expr> True() { return ConstantExpr::alloc(1, Expr::Bool); }

    klee::ref<Expr> Constant(uint64_t Value, Expr::Width W) {
      return Constant(llvm::APInt(W, Value));
    }
  };

  /// createDefaultExprBuilder - Create an expression builder which does no
  /// folding.
  ExprBuilder *createDefaultExprBuilder();

  /// createConstantFoldingExprBuilder - Create an expression builder which
  /// folds constant expressions.
  ///
  /// Base - The base builder to use when constructing expressions.
  ExprBuilder *createConstantFoldingExprBuilder(ExprBuilder *Base);

  /// createSimplifyingExprBuilder - Create an expression builder which attemps
  /// to fold redundant expressions and normalize expressions for improved
  /// caching.
  ///
  /// Base - The base builder to use when constructing expressions.
  ExprBuilder *createSimplifyingExprBuilder(ExprBuilder *Base);
}

#endif
