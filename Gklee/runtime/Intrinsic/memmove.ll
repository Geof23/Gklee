; ModuleID = 'memmove.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @memmove(i8* %dst, i8* %src, i64 %count) nounwind {
entry:
  %0 = icmp eq i8* %src, %dst                     ; <i1> [#uses=1]
  br i1 %0, label %bb8, label %bb1

bb1:                                              ; preds = %entry
  %1 = icmp ugt i8* %src, %dst                    ; <i1> [#uses=1]
  %2 = icmp eq i64 %count, 0                      ; <i1> [#uses=2]
  br i1 %1, label %bb3.preheader, label %bb4

bb3.preheader:                                    ; preds = %bb1
  br i1 %2, label %bb8, label %bb2

bb2:                                              ; preds = %bb2, %bb3.preheader
  %indvar20 = phi i64 [ %indvar.next21, %bb2 ], [ 0, %bb3.preheader ] ; <i64> [#uses=3]
  %a.016 = getelementptr i8* %dst, i64 %indvar20  ; <i8*> [#uses=1]
  %b.014 = getelementptr i8* %src, i64 %indvar20  ; <i8*> [#uses=1]
  %3 = load i8* %b.014, align 1                   ; <i8> [#uses=1]
  store i8 %3, i8* %a.016, align 1
  %indvar.next21 = add i64 %indvar20, 1           ; <i64> [#uses=2]
  %exitcond22 = icmp eq i64 %indvar.next21, %count ; <i1> [#uses=1]
  br i1 %exitcond22, label %bb8, label %bb2

bb4:                                              ; preds = %bb1
  br i1 %2, label %bb8, label %bb.nph

bb.nph:                                           ; preds = %bb4
  %tmp18 = add i64 %count, -1                     ; <i64> [#uses=1]
  br label %bb5

bb5:                                              ; preds = %bb5, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb5 ] ; <i64> [#uses=2]
  %tmp19 = sub i64 %tmp18, %indvar                ; <i64> [#uses=2]
  %a.113 = getelementptr i8* %dst, i64 %tmp19     ; <i8*> [#uses=1]
  %b.111 = getelementptr i8* %src, i64 %tmp19     ; <i8*> [#uses=1]
  %4 = load i8* %b.111, align 1                   ; <i8> [#uses=1]
  store i8 %4, i8* %a.113, align 1
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %count    ; <i1> [#uses=1]
  br i1 %exitcond, label %bb8, label %bb5

bb8:                                              ; preds = %bb5, %bb4, %bb2, %bb3.preheader, %entry
  ret i8* %dst
}
