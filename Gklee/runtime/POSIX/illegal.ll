; ModuleID = 'illegal.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@.str = private constant [18 x i8] c"ignoring (ENOMEM)\00", align 1 ; <[18 x i8]*> [#uses=1]
@.str1 = private constant [18 x i8] c"ignoring (EACCES)\00", align 1 ; <[18 x i8]*> [#uses=1]
@.str2 = private constant [17 x i8] c"ignoring (EPERM)\00", align 1 ; <[17 x i8]*> [#uses=1]
@.str3 = private constant [10 x i8] c"illegal.c\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str4 = private constant [20 x i8] c"longjmp unsupported\00", align 1 ; <[20 x i8]*> [#uses=1]
@.str5 = private constant [8 x i8] c"xxx.err\00", align 1 ; <[8 x i8]*> [#uses=1]
@.str6 = private constant [9 x i8] c"ignoring\00", align 1 ; <[9 x i8]*> [#uses=1]

define i32 @kill(i32 %pid, i32 %sig) nounwind {
entry:
  tail call void @klee_warning(i8* getelementptr inbounds ([17 x i8]* @.str2, i64 0, i64 0)) nounwind
  %0 = tail call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 1, i32* %0, align 4
  ret i32 -1
}

define i32 @fork() nounwind {
entry:
  tail call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str, i64 0, i64 0)) nounwind
  %0 = tail call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 12, i32* %0, align 4
  ret i32 -1
}

declare void @klee_warning(i8*)

declare i32* @__errno_location() nounwind readnone

define i32 @vfork() nounwind {
entry:
  tail call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str, i64 0, i64 0)) nounwind
  %0 = tail call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 12, i32* %0, align 4
  ret i32 -1
}

define weak i32 @execve(i8* %file, i8** %argv, i8** %envp) nounwind {
entry:
  tail call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str1, i64 0, i64 0)) nounwind
  %0 = tail call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 13, i32* %0, align 4
  ret i32 -1
}

define weak i32 @execvp(i8* %file, i8** %argv) nounwind {
entry:
  tail call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str1, i64 0, i64 0)) nounwind
  %0 = tail call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 13, i32* %0, align 4
  ret i32 -1
}

define weak i32 @execv(i8* %path, i8** %argv) nounwind {
entry:
  tail call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str1, i64 0, i64 0)) nounwind
  %0 = tail call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 13, i32* %0, align 4
  ret i32 -1
}

define weak i32 @execle(i8* %path, i8* %arg, ...) nounwind {
entry:
  call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str1, i64 0, i64 0)) nounwind
  %0 = call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 13, i32* %0, align 4
  ret i32 -1
}

define weak i32 @execlp(i8* %file, i8* %arg, ...) nounwind {
entry:
  call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str1, i64 0, i64 0)) nounwind
  %0 = call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 13, i32* %0, align 4
  ret i32 -1
}

define weak i32 @execl(i8* %path, i8* %arg, ...) nounwind {
entry:
  call void @klee_warning(i8* getelementptr inbounds ([18 x i8]* @.str1, i64 0, i64 0)) nounwind
  %0 = call i32* @__errno_location() nounwind readnone ; <i32*> [#uses=1]
  store i32 13, i32* %0, align 4
  ret i32 -1
}

define void @longjmp(%struct.__jmp_buf_tag* nocapture %env, i32 %val) noreturn nounwind {
entry:
  tail call void @klee_report_error(i8* getelementptr inbounds ([10 x i8]* @.str3, i64 0, i64 0), i32 35, i8* getelementptr inbounds ([20 x i8]* @.str4, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8]* @.str5, i64 0, i64 0)) noreturn nounwind
  unreachable
}

declare void @klee_report_error(i8*, i32, i8*, i8*) noreturn

define weak i32 @_setjmp(%struct.__jmp_buf_tag* %__env) nounwind {
entry:
  tail call void @klee_warning_once(i8* getelementptr inbounds ([9 x i8]* @.str6, i64 0, i64 0)) nounwind
  ret i32 0
}

declare void @klee_warning_once(i8*)
