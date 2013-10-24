; ModuleID = 'klee_div_zero_check.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private constant [22 x i8] c"klee_div_zero_check.c\00", align 1 ; <[22 x i8]*> [#uses=1]
@.str1 = private constant [15 x i8] c"divide by zero\00", align 1 ; <[15 x i8]*> [#uses=1]
@.str2 = private constant [8 x i8] c"div.err\00", align 1 ; <[8 x i8]*> [#uses=1]

define void @klee_div_zero_check(i64 %z) nounwind {
entry:
  %0 = icmp eq i64 %z, 0                          ; <i1> [#uses=1]
  br i1 %0, label %bb, label %return

bb:                                               ; preds = %entry
  tail call void @klee_report_error(i8* getelementptr inbounds ([22 x i8]* @.str, i64 0, i64 0), i32 14, i8* getelementptr inbounds ([15 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8]* @.str2, i64 0, i64 0)) noreturn nounwind
  unreachable

return:                                           ; preds = %entry
  ret void
}

declare void @klee_report_error(i8*, i32, i8*, i8*) noreturn
