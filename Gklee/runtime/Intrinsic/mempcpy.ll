; ModuleID = 'mempcpy.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @mempcpy(i8* %destaddr, i8* nocapture %srcaddr, i64 %len) nounwind {
entry:
  %0 = icmp eq i64 %len, 0                        ; <i1> [#uses=1]
  br i1 %0, label %bb2, label %bb

bb:                                               ; preds = %bb, %entry
  %indvar = phi i64 [ %indvar.next, %bb ], [ 0, %entry ] ; <i64> [#uses=3]
  %dest.06 = getelementptr i8* %destaddr, i64 %indvar ; <i8*> [#uses=1]
  %src.05 = getelementptr i8* %srcaddr, i64 %indvar ; <i8*> [#uses=1]
  %1 = load i8* %src.05, align 1                  ; <i8> [#uses=1]
  store i8 %1, i8* %dest.06, align 1
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %len      ; <i1> [#uses=1]
  br i1 %exitcond, label %bb1.bb2_crit_edge, label %bb

bb1.bb2_crit_edge:                                ; preds = %bb
  %scevgep = getelementptr i8* %destaddr, i64 %len ; <i8*> [#uses=1]
  ret i8* %scevgep

bb2:                                              ; preds = %entry
  ret i8* %destaddr
}
