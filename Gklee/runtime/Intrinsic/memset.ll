; ModuleID = 'memset.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @memset(i8* %dst, i32 %s, i64 %count) nounwind {
entry:
  %0 = icmp eq i64 %count, 0                      ; <i1> [#uses=1]
  br i1 %0, label %bb2, label %bb.nph

bb.nph:                                           ; preds = %entry
  %1 = trunc i32 %s to i8                         ; <i8> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i64> [#uses=2]
  %a.05 = getelementptr i8* %dst, i64 %indvar     ; <i8*> [#uses=1]
  store i8 %1, i8* %a.05, align 1
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %count    ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  ret i8* %dst
}
