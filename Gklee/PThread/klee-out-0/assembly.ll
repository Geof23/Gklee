; ModuleID = 't1.o'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32 }
%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8*, i32 }
%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i32 }
%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }* }
%llvm.dbg.subprogram.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1 }
%struct.FILE = type { i16, [2 x i8], i32, i8*, i8*, i8*, i8*, i8*, i8*, %struct.FILE*, [2 x i32], %struct.__mbstate_t }
%struct.__mbstate_t = type { i32, i32 }
%struct.pthread_attr_t = type { i64, [48 x i8] }

@stderr = external global %struct.FILE*           ; <%struct.FILE**> [#uses=3]
@.str = private constant [8 x i8] c"%s: %s\0A\00", align 1 ; <[8 x i8]*> [#uses=1]
@.str1 = private constant [15 x i8] c"pthread_create\00", align 1 ; <[15 x i8]*> [#uses=1]
@.str2 = private constant [13 x i8] c"pthread_join\00", align 1 ; <[13 x i8]*> [#uses=1]
@.str3 = private constant [33 x i8] c"thread %d terminated abnormally\0A\00", align 1 ; <[33 x i8]*> [#uses=1]
@.str4 = private constant [22 x i8] c"Hello, world, I'm %d\0A\00", align 1 ; <[22 x i8]*> [#uses=1]
@.str5 = internal constant [9 x i8] c"memcpy.c\00", section "llvm.metadata" ; <[9 x i8]*> [#uses=1]
@.str16 = internal constant [43 x i8] c"/home/gli/klee-FLA/klee/runtime/Intrinsic/\00", section "llvm.metadata" ; <[43 x i8]*> [#uses=1]
@.str27 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5649) (LLVM build)\00", section "llvm.metadata" ; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str5, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str16, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str27, i32 0,
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i6
@.str48 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata" ; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([18 x i8]* @.str48, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @l
@.str59 = internal constant [9 x i8] c"stddef.h\00", section "llvm.metadata" ; <[9 x i8]*> [#uses=1]
@.str6 = internal constant [90 x i8] c"/home/gli/klee-FLA/llvm-gcc-4.2-2.6/bin/../lib/gcc/x86_64-unknown-linux-gnu/4.2.1/include\00", section "llvm.metadata" ; <[90 x i8]*> [#uses=1]
@llvm.dbg.compile_unit7 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str59, i32 0, i32 0), i8* getelementptr ([90 x i8]* @.str6, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str27, i32 0
@.str8 = internal constant [7 x i8] c"size_t\00", section "llvm.metadata" ; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype9 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type*
@llvm.dbg.array = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derive
@llvm.dbg.composite = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 0,
@.str10 = internal constant [7 x i8] c"memcpy\00", section "llvm.metadata" ; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([7 x i8]* @.str10, i32 0, i32 0), i8* getelementptr ([7 x i8]* @.st
@.str11 = internal constant [10 x i8] c"memmove.c\00", section "llvm.metadata" ; <[10 x i8]*> [#uses=1]
@.str112 = internal constant [43 x i8] c"/home/gli/klee-FLA/klee/runtime/Intrinsic/\00", section "llvm.metadata" ; <[43 x i8]*> [#uses=1]
@.str213 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5649) (LLVM build)\00", section "llvm.metadata" ; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit14 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([10 x i8]* @.str11, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str112, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str213, 
@llvm.dbg.derivedtype15 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i32
@.str416 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata" ; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype17 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i8* getelementptr ([18 x i8]* @.str416, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.typ
@.str518 = internal constant [9 x i8] c"stddef.h\00", section "llvm.metadata" ; <[9 x i8]*> [#uses=1]
@.str619 = internal constant [90 x i8] c"/home/gli/klee-FLA/llvm-gcc-4.2-2.6/bin/../lib/gcc/x86_64-unknown-linux-gnu/4.2.1/include\00", section "llvm.metadata" ; <[90 x i8]*> [#uses=1]
@llvm.dbg.compile_unit720 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str518, i32 0, i32 0), i8* getelementptr ([90 x i8]* @.str619, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str213,
@.str821 = internal constant [7 x i8] c"size_t\00", section "llvm.metadata" ; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype922 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i8* getelementptr ([7 x i8]* @.str821, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit
@llvm.dbg.array23 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype15 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype15 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.
@llvm.dbg.composite24 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i32 0, 
@.str1025 = internal constant [8 x i8] c"memmove\00", section "llvm.metadata" ; <[8 x i8]*> [#uses=1]
@llvm.dbg.subprogram26 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*), i8* getelementptr ([8 x i8]* @.str1025, i32 0, i32 0), i8* getelementptr ([8 x i8]
@.str28 = internal constant [9 x i8] c"memset.c\00", section "llvm.metadata" ; <[9 x i8]*> [#uses=1]
@.str129 = internal constant [43 x i8] c"/home/gli/klee-FLA/klee/runtime/Intrinsic/\00", section "llvm.metadata" ; <[43 x i8]*> [#uses=1]
@.str230 = internal constant [52 x i8] c"4.2.1 (Based on Apple Inc. build 5649) (LLVM build)\00", section "llvm.metadata" ; <[52 x i8]*> [#uses=1]
@llvm.dbg.compile_unit31 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str28, i32 0, i32 0), i8* getelementptr ([43 x i8]* @.str129, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str230, i
@llvm.dbg.derivedtype32 = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i32
@.str333 = internal constant [4 x i8] c"int\00", section "llvm.metadata" ; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype34 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i8* getelementptr ([4 x i8]* @.str333, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type
@.str435 = internal constant [18 x i8] c"long unsigned int\00", section "llvm.metadata" ; <[18 x i8]*> [#uses=1]
@llvm.dbg.basictype5 = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i8* getelementptr ([18 x i8]* @.str435, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type
@.str636 = internal constant [9 x i8] c"stddef.h\00", section "llvm.metadata" ; <[9 x i8]*> [#uses=1]
@.str7 = internal constant [90 x i8] c"/home/gli/klee-FLA/llvm-gcc-4.2-2.6/bin/../lib/gcc/x86_64-unknown-linux-gnu/4.2.1/include\00", section "llvm.metadata" ; <[90 x i8]*> [#uses=1]
@llvm.dbg.compile_unit8 = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* null, i32 1, i8* getelementptr ([9 x i8]* @.str636, i32 0, i32 0), i8* getelementptr ([90 x i8]* @.str7, i32 0, i32 0), i8* getelementptr ([52 x i8]* @.str230, i32
@.str9 = internal constant [7 x i8] c"size_t\00", section "llvm.metadata" ; <[7 x i8]*> [#uses=1]
@llvm.dbg.derivedtype10 = internal constant %llvm.dbg.derivedtype.type { i32 458774, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i8* getelementptr ([7 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.ty
@llvm.dbg.array37 = internal constant [4 x { }*] [{ }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype32 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype32 to { }*), { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.ba
@llvm.dbg.composite38 = internal constant %llvm.dbg.composite.type { i32 458773, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i32 0, 
@.str1139 = internal constant [7 x i8] c"memset\00", section "llvm.metadata" ; <[7 x i8]*> [#uses=1]
@llvm.dbg.subprogram40 = internal constant %llvm.dbg.subprogram.type { i32 458798, { }* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*), i8* getelementptr ([7 x i8]* @.str1139, i32 0, i32 0), i8* getelementptr ([7 x i8]

define i8* @hola(i8* %arg) nounwind {
entry:
  %0 = bitcast i8* %arg to i32*                   ; <i32*> [#uses=1]
  %1 = load i32* %0, align 4                      ; <i32> [#uses=1]
  %2 = tail call i32 (i8*, ...)* @printf(i8* noalias getelementptr inbounds ([22 x i8]* @.str4, i64 0, i64 0), i32 %1) nounwind ; <i32> [#uses=0]
  ret i8* %arg
}

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
  %status = alloca i32*, align 8                  ; <i32**> [#uses=2]
  %ids = alloca [4 x i32], align 4                ; <[4 x i32]*> [#uses=1]
  %threads = alloca [4 x i64], align 8            ; <[4 x i64]*> [#uses=2]
  br label %bb3

bb:                                               ; preds = %bb3
  %scevgep25 = getelementptr [4 x i64]* %threads, i64 0, i64 %indvar23 ; <i64*> [#uses=1]
  %scevgep2627 = bitcast i32* %scevgep26 to i8*   ; <i8*> [#uses=1]
  store i32 %worker.0, i32* %scevgep26, align 4
  %0 = call i32 @pthread_create(i64* noalias %scevgep25, %struct.pthread_attr_t* noalias null, i8* (i8*)* @hola, i8* noalias %scevgep2627) nounwind ; <i32> [#uses=2]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %2 = call i8* @strerror(i32 %0) nounwind        ; <i8*> [#uses=1]
  %3 = load %struct.FILE** @stderr, align 8       ; <%struct.FILE*> [#uses=1]
  %4 = call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* noalias %3, i8* noalias getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([15 x i8]* @.str1, i64 0, i64 0), i8* %2) nounwind ; <i32> [#uses=0]
  ret i32 1

bb2:                                              ; preds = %bb
  %indvar.next24 = add i64 %indvar23, 1           ; <i64> [#uses=1]
  br label %bb3

bb3:                                              ; preds = %bb2, %entry
  %indvar23 = phi i64 [ 0, %entry ], [ %indvar.next24, %bb2 ] ; <i64> [#uses=4]
  %worker.0 = trunc i64 %indvar23 to i32          ; <i32> [#uses=2]
  %scevgep26 = getelementptr [4 x i32]* %ids, i64 0, i64 %indvar23 ; <i32*> [#uses=2]
  %5 = icmp sgt i32 %worker.0, 3                  ; <i1> [#uses=1]
  br i1 %5, label %bb12.loopexit, label %bb

bb5:                                              ; preds = %bb12
  %scevgep = getelementptr [4 x i64]* %threads, i64 0, i64 %indvar ; <i64*> [#uses=1]
  %6 = load i64* %scevgep, align 8                ; <i64> [#uses=1]
  %7 = call i32 @pthread_join(i64 %6, i8** %status6) nounwind ; <i32> [#uses=2]
  %8 = icmp eq i32 %7, 0                          ; <i1> [#uses=1]
  br i1 %8, label %bb9, label %bb8

bb8:                                              ; preds = %bb5
  %9 = call i8* @strerror(i32 %7) nounwind        ; <i8*> [#uses=1]
  %10 = load %struct.FILE** @stderr, align 8      ; <%struct.FILE*> [#uses=1]
  %11 = call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* noalias %10, i8* noalias getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8]* @.str2, i64 0, i64 0), i8* %9) nounwind ; <i32> [#uses=0]
  ret i32 1

bb9:                                              ; preds = %bb5
  %12 = load i32** %status, align 8               ; <i32*> [#uses=1]
  %13 = load i32* %12, align 4                    ; <i32> [#uses=1]
  %14 = icmp eq i32 %13, %worker.1                ; <i1> [#uses=1]
  br i1 %14, label %bb11, label %bb10

bb10:                                             ; preds = %bb9
  %15 = load %struct.FILE** @stderr, align 8      ; <%struct.FILE*> [#uses=1]
  %16 = call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* noalias %15, i8* noalias getelementptr inbounds ([33 x i8]* @.str3, i64 0, i64 0), i32 %worker.1) nounwind ; <i32> [#uses=0]
  ret i32 1

bb11:                                             ; preds = %bb9
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %bb12

bb12.loopexit:                                    ; preds = %bb3
  %status6 = bitcast i32** %status to i8**        ; <i8**> [#uses=1]
  br label %bb12

bb12:                                             ; preds = %bb12.loopexit, %bb11
  %indvar = phi i64 [ 0, %bb12.loopexit ], [ %indvar.next, %bb11 ] ; <i64> [#uses=3]
  %worker.1 = trunc i64 %indvar to i32            ; <i32> [#uses=3]
  %17 = icmp sgt i32 %worker.1, 3                 ; <i1> [#uses=1]
  br i1 %17, label %bb13, label %bb5

bb13:                                             ; preds = %bb12
  ret i32 0
}

declare i32 @pthread_create(i64* noalias, %struct.pthread_attr_t* noalias, i8* (i8*)*, i8* noalias) nounwind

declare i8* @strerror(i32) nounwind

declare i32 @fprintf(%struct.FILE* noalias nocapture, i8* noalias nocapture, ...) nounwind

declare i32 @pthread_join(i64, i8**)

declare i32 @printf(i8* nocapture, ...) nounwind

define i8* @memcpy(i8* %destaddr, i8* nocapture %srcaddr, i64 %len) nounwind {
entry:
  tail call void @llvm.dbg.stoppoint(i32 16, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
  %0 = icmp eq i64 %len, 0                        ; <i1> [#uses=1]
  br i1 %0, label %bb2, label %bb

bb:                                               ; preds = %bb, %entry
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %bb ] ; <i64> [#uses=3]
  %src.05 = getelementptr i8* %srcaddr, i64 %indvar ; <i8*> [#uses=1]
  %dest.06 = getelementptr i8* %destaddr, i64 %indvar ; <i8*> [#uses=1]
  %1 = load i8* %src.05, align 1                  ; <i8> [#uses=1]
  store i8 %1, i8* %dest.06, align 1
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond1 = icmp eq i64 %indvar.next, %len     ; <i1> [#uses=1]
  br i1 %exitcond1, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  tail call void @llvm.dbg.stoppoint(i32 18, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
  ret i8* %destaddr
}

declare void @llvm.dbg.func.start({ }*) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind readnone

declare void @llvm.dbg.region.end({ }*) nounwind readnone

define i8* @memmove(i8* %dst, i8* %src, i64 %count) nounwind {
entry:
  tail call void @llvm.dbg.stoppoint(i32 18, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
  %0 = icmp eq i8* %src, %dst                     ; <i1> [#uses=1]
  br i1 %0, label %bb8, label %bb1

bb1:                                              ; preds = %entry
  tail call void @llvm.dbg.stoppoint(i32 22, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
  %1 = icmp ugt i8* %src, %dst                    ; <i1> [#uses=1]
  %2 = icmp eq i64 %count, 0                      ; <i1> [#uses=2]
  br i1 %1, label %bb3.preheader, label %bb4

bb3.preheader:                                    ; preds = %bb1
  tail call void @llvm.dbg.stoppoint(i32 23, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
  br i1 %2, label %bb8, label %bb2

bb2:                                              ; preds = %bb2, %bb3.preheader
  %indvar20 = phi i64 [ 0, %bb3.preheader ], [ %indvar.next21, %bb2 ] ; <i64> [#uses=3]
  %b.014 = getelementptr i8* %src, i64 %indvar20  ; <i8*> [#uses=1]
  %a.016 = getelementptr i8* %dst, i64 %indvar20  ; <i8*> [#uses=1]
  %3 = load i8* %b.014, align 1                   ; <i8> [#uses=1]
  store i8 %3, i8* %a.016, align 1
  %indvar.next21 = add i64 %indvar20, 1           ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next21, %count  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb8, label %bb2

bb4:                                              ; preds = %bb1
  tail call void @llvm.dbg.stoppoint(i32 29, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
  br i1 %2, label %bb8, label %bb.nph

bb.nph:                                           ; preds = %bb4
  %tmp18 = add i64 %count, -1                     ; <i64> [#uses=1]
  br label %bb5

bb5:                                              ; preds = %bb5, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb5 ] ; <i64> [#uses=2]
  %tmp3 = sub i64 %tmp18, %indvar                 ; <i64> [#uses=2]
  %b.111 = getelementptr i8* %src, i64 %tmp3      ; <i8*> [#uses=1]
  %a.113 = getelementptr i8* %dst, i64 %tmp3      ; <i8*> [#uses=1]
  %4 = load i8* %b.111, align 1                   ; <i8> [#uses=1]
  store i8 %4, i8* %a.113, align 1
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond2 = icmp eq i64 %indvar.next, %count   ; <i1> [#uses=1]
  br i1 %exitcond2, label %bb8, label %bb5

bb8:                                              ; preds = %bb5, %bb4, %bb2, %bb3.preheader, %entry
  tail call void @llvm.dbg.stoppoint(i32 34, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit14 to { }*))
  ret i8* %dst
}

define i8* @memset(i8* %dst, i32 %s, i64 %count) nounwind {
entry:
  tail call void @llvm.dbg.stoppoint(i32 14, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*))
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
  %exitcond1 = icmp eq i64 %indvar.next, %count   ; <i1> [#uses=1]
  br i1 %exitcond1, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  tail call void @llvm.dbg.stoppoint(i32 16, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit31 to { }*))
  ret i8* %dst
}
