; ModuleID = 'klee_int.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @klee_int(i8* %name) nounwind {
entry:
  %x = alloca i32, align 4                        ; <i32*> [#uses=2]
  %x1 = bitcast i32* %x to i8*                    ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %x1, i32 4, i8* %name) nounwind
  %0 = load i32* %x, align 4                      ; <i32> [#uses=1]
  ret i32 %0
}

declare void @klee_make_symbolic(i8*, i32, i8*)
