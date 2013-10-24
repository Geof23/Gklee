; ModuleID = 'klee_range.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private constant [13 x i8] c"klee_range.c\00", align 1 ; <[13 x i8]*> [#uses=1]
@.str1 = private constant [14 x i8] c"invalid range\00", align 1 ; <[14 x i8]*> [#uses=1]
@.str2 = private constant [5 x i8] c"user\00", align 1 ; <[5 x i8]*> [#uses=1]

define i32 @klee_range(i32 %start, i32 %end, i8* %name) nounwind {
entry:
  %x = alloca i32, align 4                        ; <i32*> [#uses=4]
  %0 = icmp slt i32 %start, %end                  ; <i1> [#uses=1]
  br i1 %0, label %bb1, label %bb

bb:                                               ; preds = %entry
  call void @klee_report_error(i8* getelementptr inbounds ([13 x i8]* @.str, i64 0, i64 0), i32 17, i8* getelementptr inbounds ([14 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8]* @.str2, i64 0, i64 0)) noreturn nounwind
  unreachable

bb1:                                              ; preds = %entry
  %1 = add nsw i32 %start, 1                      ; <i32> [#uses=1]
  %2 = icmp eq i32 %1, %end                       ; <i1> [#uses=1]
  br i1 %2, label %bb8, label %bb3

bb3:                                              ; preds = %bb1
  %x4 = bitcast i32* %x to i8*                    ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %x4, i32 4, i8* %name) nounwind
  %3 = icmp eq i32 %start, 0                      ; <i1> [#uses=1]
  %4 = load i32* %x, align 4                      ; <i32> [#uses=2]
  br i1 %3, label %bb5, label %bb6

bb5:                                              ; preds = %bb3
  %5 = icmp ult i32 %4, %end                      ; <i1> [#uses=1]
  %6 = zext i1 %5 to i32                          ; <i32> [#uses=1]
  call void @klee_assume(i32 %6) nounwind
  br label %bb7

bb6:                                              ; preds = %bb3
  %7 = icmp sge i32 %4, %start                    ; <i1> [#uses=1]
  %8 = zext i1 %7 to i32                          ; <i32> [#uses=1]
  call void @klee_assume(i32 %8) nounwind
  %9 = load i32* %x, align 4                      ; <i32> [#uses=1]
  %10 = icmp slt i32 %9, %end                     ; <i1> [#uses=1]
  %11 = zext i1 %10 to i32                        ; <i32> [#uses=1]
  call void @klee_assume(i32 %11) nounwind
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  %12 = load i32* %x, align 4                     ; <i32> [#uses=1]
  ret i32 %12

bb8:                                              ; preds = %bb1
  ret i32 %start
}

declare void @klee_report_error(i8*, i32, i8*, i8*) noreturn

declare void @klee_make_symbolic(i8*, i32, i8*)

declare void @klee_assume(i32)
