; ModuleID = 'fd_32.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%struct.__fsid_t = type { [2 x i32] }
%struct.__va_list_tag = type { i32, i32, i8*, i8* }
%struct.dirent = type { i64, i64, i16, i8, [256 x i8] }
%struct.stat = type { i64, i64, i64, i32, i32, i32, i32, i64, i64, i64, i64, %struct.timespec, %struct.timespec, %struct.timespec, [3 x i64] }
%struct.statfs = type { i64, i64, i64, i64, i64, i64, i64, %struct.__fsid_t, i64, i64, [5 x i64] }
%struct.timespec = type { i64, i64 }

@__getdents = alias i64 (i32, %struct.dirent*, i64)* @getdents ; <i64 (i32, %struct.dirent*, i64)*> [#uses=0]

define i32 @open(i8* %pathname, i32 %flags, ...) nounwind {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 8 ; <[1 x %struct.__va_list_tag]*> [#uses=4]
  %0 = and i32 %flags, 64                         ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %bb8, label %bb

bb:                                               ; preds = %entry
  %ap12 = bitcast [1 x %struct.__va_list_tag]* %ap to i8* ; <i8*> [#uses=2]
  call void @llvm.va_start(i8* %ap12)
  %2 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 0 ; <i32*> [#uses=2]
  %3 = load i32* %2, align 8                      ; <i32> [#uses=3]
  %4 = icmp ult i32 %3, 48                        ; <i1> [#uses=1]
  br i1 %4, label %bb3, label %bb4

bb3:                                              ; preds = %bb
  %5 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 3 ; <i8**> [#uses=1]
  %6 = load i8** %5, align 8                      ; <i8*> [#uses=1]
  %7 = inttoptr i32 %3 to i8*                     ; <i8*> [#uses=1]
  %8 = ptrtoint i8* %6 to i64                     ; <i64> [#uses=1]
  %9 = ptrtoint i8* %7 to i64                     ; <i64> [#uses=1]
  %10 = add i64 %9, %8                            ; <i64> [#uses=1]
  %11 = inttoptr i64 %10 to i8*                   ; <i8*> [#uses=1]
  %12 = add i32 %3, 8                             ; <i32> [#uses=1]
  store i32 %12, i32* %2, align 8
  br label %bb5

bb4:                                              ; preds = %bb
  %13 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 2 ; <i8**> [#uses=2]
  %14 = load i8** %13, align 8                    ; <i8*> [#uses=2]
  %15 = getelementptr inbounds i8* %14, i64 8     ; <i8*> [#uses=1]
  store i8* %15, i8** %13, align 8
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  %addr.24.0 = phi i8* [ %14, %bb4 ], [ %11, %bb3 ] ; <i8*> [#uses=1]
  %16 = bitcast i8* %addr.24.0 to i32*            ; <i32*> [#uses=1]
  %17 = load i32* %16, align 4                    ; <i32> [#uses=1]
  call void @llvm.va_end(i8* %ap12)
  br label %bb8

bb8:                                              ; preds = %bb5, %entry
  %mode.0 = phi i32 [ %17, %bb5 ], [ 0, %entry ]  ; <i32> [#uses=1]
  %18 = call i32 @__fd_open(i8* %pathname, i32 %flags, i32 %mode.0) nounwind ; <i32> [#uses=1]
  ret i32 %18
}

define weak i32 @fstat64(i32 %fd, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_fstat(i32 %fd, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_fstat(i32, %struct.stat*)

define i32 @fstat(i32 %fd, %struct.stat* nocapture %buf) nounwind {
entry:
  %tmp = alloca %struct.stat, align 8             ; <%struct.stat*> [#uses=14]
  %0 = call i32 @__fd_fstat(i32 %fd, %struct.stat* %tmp) nounwind ; <i32> [#uses=1]
  %1 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 0 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %3 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 0 ; <i64*> [#uses=1]
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 1 ; <i64*> [#uses=1]
  %5 = load i64* %4, align 8                      ; <i64> [#uses=1]
  %6 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 1 ; <i64*> [#uses=1]
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 3 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 8                      ; <i32> [#uses=1]
  %9 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 %8, i32* %9, align 8
  %10 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 2 ; <i64*> [#uses=1]
  %11 = load i64* %10, align 8                    ; <i64> [#uses=1]
  %12 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 2 ; <i64*> [#uses=1]
  store i64 %11, i64* %12, align 8
  %13 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 4 ; <i32*> [#uses=1]
  %14 = load i32* %13, align 4                    ; <i32> [#uses=1]
  %15 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 4 ; <i32*> [#uses=1]
  store i32 %14, i32* %15, align 4
  %16 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 5 ; <i32*> [#uses=1]
  %17 = load i32* %16, align 8                    ; <i32> [#uses=1]
  %18 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 5 ; <i32*> [#uses=1]
  store i32 %17, i32* %18, align 8
  %19 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 7 ; <i64*> [#uses=1]
  %20 = load i64* %19, align 8                    ; <i64> [#uses=1]
  %21 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 7 ; <i64*> [#uses=1]
  store i64 %20, i64* %21, align 8
  %22 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 8 ; <i64*> [#uses=1]
  %23 = load i64* %22, align 8                    ; <i64> [#uses=1]
  %24 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 8 ; <i64*> [#uses=1]
  store i64 %23, i64* %24, align 8
  %25 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  %27 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  store i64 %26, i64* %27, align 8
  %28 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %29 = load i64* %28, align 8                    ; <i64> [#uses=1]
  %30 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  store i64 %29, i64* %30, align 8
  %31 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %32 = load i64* %31, align 8                    ; <i64> [#uses=1]
  %33 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  store i64 %32, i64* %33, align 8
  %34 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 9 ; <i64*> [#uses=1]
  %35 = load i64* %34, align 8                    ; <i64> [#uses=1]
  %36 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 9 ; <i64*> [#uses=1]
  store i64 %35, i64* %36, align 8
  %37 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 10 ; <i64*> [#uses=1]
  %38 = load i64* %37, align 8                    ; <i64> [#uses=1]
  %39 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 10 ; <i64*> [#uses=1]
  store i64 %38, i64* %39, align 8
  ret i32 %0
}

define i32 @__fxstat(i32 %vers, i32 %fd, %struct.stat* nocapture %buf) nounwind {
entry:
  %tmp = alloca %struct.stat, align 8             ; <%struct.stat*> [#uses=14]
  %0 = call i32 @__fd_fstat(i32 %fd, %struct.stat* %tmp) nounwind ; <i32> [#uses=1]
  %1 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 0 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %3 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 0 ; <i64*> [#uses=1]
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 1 ; <i64*> [#uses=1]
  %5 = load i64* %4, align 8                      ; <i64> [#uses=1]
  %6 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 1 ; <i64*> [#uses=1]
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 3 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 8                      ; <i32> [#uses=1]
  %9 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 %8, i32* %9, align 8
  %10 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 2 ; <i64*> [#uses=1]
  %11 = load i64* %10, align 8                    ; <i64> [#uses=1]
  %12 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 2 ; <i64*> [#uses=1]
  store i64 %11, i64* %12, align 8
  %13 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 4 ; <i32*> [#uses=1]
  %14 = load i32* %13, align 4                    ; <i32> [#uses=1]
  %15 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 4 ; <i32*> [#uses=1]
  store i32 %14, i32* %15, align 4
  %16 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 5 ; <i32*> [#uses=1]
  %17 = load i32* %16, align 8                    ; <i32> [#uses=1]
  %18 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 5 ; <i32*> [#uses=1]
  store i32 %17, i32* %18, align 8
  %19 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 7 ; <i64*> [#uses=1]
  %20 = load i64* %19, align 8                    ; <i64> [#uses=1]
  %21 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 7 ; <i64*> [#uses=1]
  store i64 %20, i64* %21, align 8
  %22 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 8 ; <i64*> [#uses=1]
  %23 = load i64* %22, align 8                    ; <i64> [#uses=1]
  %24 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 8 ; <i64*> [#uses=1]
  store i64 %23, i64* %24, align 8
  %25 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  %27 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  store i64 %26, i64* %27, align 8
  %28 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %29 = load i64* %28, align 8                    ; <i64> [#uses=1]
  %30 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  store i64 %29, i64* %30, align 8
  %31 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %32 = load i64* %31, align 8                    ; <i64> [#uses=1]
  %33 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  store i64 %32, i64* %33, align 8
  %34 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 9 ; <i64*> [#uses=1]
  %35 = load i64* %34, align 8                    ; <i64> [#uses=1]
  %36 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 9 ; <i64*> [#uses=1]
  store i64 %35, i64* %36, align 8
  %37 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 10 ; <i64*> [#uses=1]
  %38 = load i64* %37, align 8                    ; <i64> [#uses=1]
  %39 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 10 ; <i64*> [#uses=1]
  store i64 %38, i64* %39, align 8
  ret i32 %0
}

define weak i32 @lstat64(i8* %path, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_lstat(i8* %path, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_lstat(i8*, %struct.stat*)

define i32 @lstat(i8* %path, %struct.stat* nocapture %buf) nounwind {
entry:
  %tmp = alloca %struct.stat, align 8             ; <%struct.stat*> [#uses=14]
  %0 = call i32 @__fd_lstat(i8* %path, %struct.stat* %tmp) nounwind ; <i32> [#uses=1]
  %1 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 0 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %3 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 0 ; <i64*> [#uses=1]
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 1 ; <i64*> [#uses=1]
  %5 = load i64* %4, align 8                      ; <i64> [#uses=1]
  %6 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 1 ; <i64*> [#uses=1]
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 3 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 8                      ; <i32> [#uses=1]
  %9 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 %8, i32* %9, align 8
  %10 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 2 ; <i64*> [#uses=1]
  %11 = load i64* %10, align 8                    ; <i64> [#uses=1]
  %12 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 2 ; <i64*> [#uses=1]
  store i64 %11, i64* %12, align 8
  %13 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 4 ; <i32*> [#uses=1]
  %14 = load i32* %13, align 4                    ; <i32> [#uses=1]
  %15 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 4 ; <i32*> [#uses=1]
  store i32 %14, i32* %15, align 4
  %16 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 5 ; <i32*> [#uses=1]
  %17 = load i32* %16, align 8                    ; <i32> [#uses=1]
  %18 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 5 ; <i32*> [#uses=1]
  store i32 %17, i32* %18, align 8
  %19 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 7 ; <i64*> [#uses=1]
  %20 = load i64* %19, align 8                    ; <i64> [#uses=1]
  %21 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 7 ; <i64*> [#uses=1]
  store i64 %20, i64* %21, align 8
  %22 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 8 ; <i64*> [#uses=1]
  %23 = load i64* %22, align 8                    ; <i64> [#uses=1]
  %24 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 8 ; <i64*> [#uses=1]
  store i64 %23, i64* %24, align 8
  %25 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  %27 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  store i64 %26, i64* %27, align 8
  %28 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %29 = load i64* %28, align 8                    ; <i64> [#uses=1]
  %30 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  store i64 %29, i64* %30, align 8
  %31 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %32 = load i64* %31, align 8                    ; <i64> [#uses=1]
  %33 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  store i64 %32, i64* %33, align 8
  %34 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 9 ; <i64*> [#uses=1]
  %35 = load i64* %34, align 8                    ; <i64> [#uses=1]
  %36 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 9 ; <i64*> [#uses=1]
  store i64 %35, i64* %36, align 8
  %37 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 10 ; <i64*> [#uses=1]
  %38 = load i64* %37, align 8                    ; <i64> [#uses=1]
  %39 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 10 ; <i64*> [#uses=1]
  store i64 %38, i64* %39, align 8
  ret i32 %0
}

define i32 @__lxstat(i32 %vers, i8* %path, %struct.stat* nocapture %buf) nounwind {
entry:
  %tmp = alloca %struct.stat, align 8             ; <%struct.stat*> [#uses=14]
  %0 = call i32 @__fd_lstat(i8* %path, %struct.stat* %tmp) nounwind ; <i32> [#uses=1]
  %1 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 0 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %3 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 0 ; <i64*> [#uses=1]
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 1 ; <i64*> [#uses=1]
  %5 = load i64* %4, align 8                      ; <i64> [#uses=1]
  %6 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 1 ; <i64*> [#uses=1]
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 3 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 8                      ; <i32> [#uses=1]
  %9 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 %8, i32* %9, align 8
  %10 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 2 ; <i64*> [#uses=1]
  %11 = load i64* %10, align 8                    ; <i64> [#uses=1]
  %12 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 2 ; <i64*> [#uses=1]
  store i64 %11, i64* %12, align 8
  %13 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 4 ; <i32*> [#uses=1]
  %14 = load i32* %13, align 4                    ; <i32> [#uses=1]
  %15 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 4 ; <i32*> [#uses=1]
  store i32 %14, i32* %15, align 4
  %16 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 5 ; <i32*> [#uses=1]
  %17 = load i32* %16, align 8                    ; <i32> [#uses=1]
  %18 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 5 ; <i32*> [#uses=1]
  store i32 %17, i32* %18, align 8
  %19 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 7 ; <i64*> [#uses=1]
  %20 = load i64* %19, align 8                    ; <i64> [#uses=1]
  %21 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 7 ; <i64*> [#uses=1]
  store i64 %20, i64* %21, align 8
  %22 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 8 ; <i64*> [#uses=1]
  %23 = load i64* %22, align 8                    ; <i64> [#uses=1]
  %24 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 8 ; <i64*> [#uses=1]
  store i64 %23, i64* %24, align 8
  %25 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  %27 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  store i64 %26, i64* %27, align 8
  %28 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %29 = load i64* %28, align 8                    ; <i64> [#uses=1]
  %30 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  store i64 %29, i64* %30, align 8
  %31 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %32 = load i64* %31, align 8                    ; <i64> [#uses=1]
  %33 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  store i64 %32, i64* %33, align 8
  %34 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 9 ; <i64*> [#uses=1]
  %35 = load i64* %34, align 8                    ; <i64> [#uses=1]
  %36 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 9 ; <i64*> [#uses=1]
  store i64 %35, i64* %36, align 8
  %37 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 10 ; <i64*> [#uses=1]
  %38 = load i64* %37, align 8                    ; <i64> [#uses=1]
  %39 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 10 ; <i64*> [#uses=1]
  store i64 %38, i64* %39, align 8
  ret i32 %0
}

define weak i32 @stat64(i8* %path, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_stat(i8* %path, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_stat(i8*, %struct.stat*)

define i32 @stat(i8* %path, %struct.stat* nocapture %buf) nounwind {
entry:
  %tmp = alloca %struct.stat, align 8             ; <%struct.stat*> [#uses=14]
  %0 = call i32 @__fd_stat(i8* %path, %struct.stat* %tmp) nounwind ; <i32> [#uses=1]
  %1 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 0 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %3 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 0 ; <i64*> [#uses=1]
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 1 ; <i64*> [#uses=1]
  %5 = load i64* %4, align 8                      ; <i64> [#uses=1]
  %6 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 1 ; <i64*> [#uses=1]
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 3 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 8                      ; <i32> [#uses=1]
  %9 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 %8, i32* %9, align 8
  %10 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 2 ; <i64*> [#uses=1]
  %11 = load i64* %10, align 8                    ; <i64> [#uses=1]
  %12 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 2 ; <i64*> [#uses=1]
  store i64 %11, i64* %12, align 8
  %13 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 4 ; <i32*> [#uses=1]
  %14 = load i32* %13, align 4                    ; <i32> [#uses=1]
  %15 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 4 ; <i32*> [#uses=1]
  store i32 %14, i32* %15, align 4
  %16 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 5 ; <i32*> [#uses=1]
  %17 = load i32* %16, align 8                    ; <i32> [#uses=1]
  %18 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 5 ; <i32*> [#uses=1]
  store i32 %17, i32* %18, align 8
  %19 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 7 ; <i64*> [#uses=1]
  %20 = load i64* %19, align 8                    ; <i64> [#uses=1]
  %21 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 7 ; <i64*> [#uses=1]
  store i64 %20, i64* %21, align 8
  %22 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 8 ; <i64*> [#uses=1]
  %23 = load i64* %22, align 8                    ; <i64> [#uses=1]
  %24 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 8 ; <i64*> [#uses=1]
  store i64 %23, i64* %24, align 8
  %25 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  %27 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  store i64 %26, i64* %27, align 8
  %28 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %29 = load i64* %28, align 8                    ; <i64> [#uses=1]
  %30 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  store i64 %29, i64* %30, align 8
  %31 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %32 = load i64* %31, align 8                    ; <i64> [#uses=1]
  %33 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  store i64 %32, i64* %33, align 8
  %34 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 9 ; <i64*> [#uses=1]
  %35 = load i64* %34, align 8                    ; <i64> [#uses=1]
  %36 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 9 ; <i64*> [#uses=1]
  store i64 %35, i64* %36, align 8
  %37 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 10 ; <i64*> [#uses=1]
  %38 = load i64* %37, align 8                    ; <i64> [#uses=1]
  %39 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 10 ; <i64*> [#uses=1]
  store i64 %38, i64* %39, align 8
  ret i32 %0
}

define i32 @__xstat(i32 %vers, i8* %path, %struct.stat* nocapture %buf) nounwind {
entry:
  %tmp = alloca %struct.stat, align 8             ; <%struct.stat*> [#uses=14]
  %0 = call i32 @__fd_stat(i8* %path, %struct.stat* %tmp) nounwind ; <i32> [#uses=1]
  %1 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 0 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %3 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 0 ; <i64*> [#uses=1]
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 1 ; <i64*> [#uses=1]
  %5 = load i64* %4, align 8                      ; <i64> [#uses=1]
  %6 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 1 ; <i64*> [#uses=1]
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 3 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 8                      ; <i32> [#uses=1]
  %9 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 %8, i32* %9, align 8
  %10 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 2 ; <i64*> [#uses=1]
  %11 = load i64* %10, align 8                    ; <i64> [#uses=1]
  %12 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 2 ; <i64*> [#uses=1]
  store i64 %11, i64* %12, align 8
  %13 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 4 ; <i32*> [#uses=1]
  %14 = load i32* %13, align 4                    ; <i32> [#uses=1]
  %15 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 4 ; <i32*> [#uses=1]
  store i32 %14, i32* %15, align 4
  %16 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 5 ; <i32*> [#uses=1]
  %17 = load i32* %16, align 8                    ; <i32> [#uses=1]
  %18 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 5 ; <i32*> [#uses=1]
  store i32 %17, i32* %18, align 8
  %19 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 7 ; <i64*> [#uses=1]
  %20 = load i64* %19, align 8                    ; <i64> [#uses=1]
  %21 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 7 ; <i64*> [#uses=1]
  store i64 %20, i64* %21, align 8
  %22 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 8 ; <i64*> [#uses=1]
  %23 = load i64* %22, align 8                    ; <i64> [#uses=1]
  %24 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 8 ; <i64*> [#uses=1]
  store i64 %23, i64* %24, align 8
  %25 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  %27 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  store i64 %26, i64* %27, align 8
  %28 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %29 = load i64* %28, align 8                    ; <i64> [#uses=1]
  %30 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  store i64 %29, i64* %30, align 8
  %31 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %32 = load i64* %31, align 8                    ; <i64> [#uses=1]
  %33 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  store i64 %32, i64* %33, align 8
  %34 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 9 ; <i64*> [#uses=1]
  %35 = load i64* %34, align 8                    ; <i64> [#uses=1]
  %36 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 9 ; <i64*> [#uses=1]
  store i64 %35, i64* %36, align 8
  %37 = getelementptr inbounds %struct.stat* %tmp, i64 0, i32 10 ; <i64*> [#uses=1]
  %38 = load i64* %37, align 8                    ; <i64> [#uses=1]
  %39 = getelementptr inbounds %struct.stat* %buf, i64 0, i32 10 ; <i64*> [#uses=1]
  store i64 %38, i64* %39, align 8
  ret i32 %0
}

define weak i64 @lseek64(i32 %fd, i64 %off, i32 %whence) nounwind {
entry:
  %0 = tail call i64 @__fd_lseek(i32 %fd, i64 %off, i32 %whence) nounwind ; <i64> [#uses=1]
  ret i64 %0
}

declare i64 @__fd_lseek(i32, i64, i32)

define i64 @lseek(i32 %fd, i64 %off, i32 %whence) nounwind {
entry:
  %0 = tail call i64 @__fd_lseek(i32 %fd, i64 %off, i32 %whence) nounwind ; <i64> [#uses=1]
  ret i64 %0
}

define weak i32 @open64(i8* %pathname, i32 %flags, ...) nounwind {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 8 ; <[1 x %struct.__va_list_tag]*> [#uses=4]
  %0 = and i32 %flags, 64                         ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %bb8, label %bb

bb:                                               ; preds = %entry
  %ap12 = bitcast [1 x %struct.__va_list_tag]* %ap to i8* ; <i8*> [#uses=2]
  call void @llvm.va_start(i8* %ap12)
  %2 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 0 ; <i32*> [#uses=2]
  %3 = load i32* %2, align 8                      ; <i32> [#uses=3]
  %4 = icmp ult i32 %3, 48                        ; <i1> [#uses=1]
  br i1 %4, label %bb3, label %bb4

bb3:                                              ; preds = %bb
  %5 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 3 ; <i8**> [#uses=1]
  %6 = load i8** %5, align 8                      ; <i8*> [#uses=1]
  %7 = inttoptr i32 %3 to i8*                     ; <i8*> [#uses=1]
  %8 = ptrtoint i8* %6 to i64                     ; <i64> [#uses=1]
  %9 = ptrtoint i8* %7 to i64                     ; <i64> [#uses=1]
  %10 = add i64 %9, %8                            ; <i64> [#uses=1]
  %11 = inttoptr i64 %10 to i8*                   ; <i8*> [#uses=1]
  %12 = add i32 %3, 8                             ; <i32> [#uses=1]
  store i32 %12, i32* %2, align 8
  br label %bb5

bb4:                                              ; preds = %bb
  %13 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 2 ; <i8**> [#uses=2]
  %14 = load i8** %13, align 8                    ; <i8*> [#uses=2]
  %15 = getelementptr inbounds i8* %14, i64 8     ; <i8*> [#uses=1]
  store i8* %15, i8** %13, align 8
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  %addr.29.0 = phi i8* [ %14, %bb4 ], [ %11, %bb3 ] ; <i8*> [#uses=1]
  %16 = bitcast i8* %addr.29.0 to i32*            ; <i32*> [#uses=1]
  %17 = load i32* %16, align 4                    ; <i32> [#uses=1]
  call void @llvm.va_end(i8* %ap12)
  br label %bb8

bb8:                                              ; preds = %bb5, %entry
  %mode.0 = phi i32 [ %17, %bb5 ], [ 0, %entry ]  ; <i32> [#uses=1]
  %18 = call i32 @__fd_open(i8* %pathname, i32 %flags, i32 %mode.0) nounwind ; <i32> [#uses=1]
  ret i32 %18
}

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind

declare i32 @__fd_open(i8*, i32, i32)

define i64 @getdents(i32 %fd, %struct.dirent* %dirp, i64 %nbytes) nounwind {
entry:
  %0 = trunc i64 %nbytes to i32                   ; <i32> [#uses=1]
  %1 = tail call i32 @__fd_getdents(i32 %fd, %struct.dirent* %dirp, i32 %0) nounwind ; <i32> [#uses=2]
  %2 = sext i32 %1 to i64                         ; <i64> [#uses=2]
  %3 = icmp sgt i32 %1, 0                         ; <i1> [#uses=1]
  br i1 %3, label %bb, label %bb3

bb:                                               ; preds = %entry
  %4 = bitcast %struct.dirent* %dirp to i8*       ; <i8*> [#uses=1]
  %5 = getelementptr inbounds i8* %4, i64 %2      ; <i8*> [#uses=2]
  %6 = bitcast i8* %5 to %struct.dirent*          ; <%struct.dirent*> [#uses=1]
  %7 = icmp ugt %struct.dirent* %6, %dirp         ; <i1> [#uses=1]
  br i1 %7, label %bb1, label %bb3

bb1:                                              ; preds = %bb1, %bb
  %dp64.05 = phi %struct.dirent* [ %13, %bb1 ], [ %dirp, %bb ] ; <%struct.dirent*> [#uses=2]
  %8 = getelementptr inbounds %struct.dirent* %dp64.05, i64 0, i32 2 ; <i16*> [#uses=1]
  %9 = bitcast %struct.dirent* %dp64.05 to i8*    ; <i8*> [#uses=1]
  %10 = load i16* %8, align 8                     ; <i16> [#uses=1]
  %11 = zext i16 %10 to i64                       ; <i64> [#uses=1]
  %12 = getelementptr inbounds i8* %9, i64 %11    ; <i8*> [#uses=2]
  %13 = bitcast i8* %12 to %struct.dirent*        ; <%struct.dirent*> [#uses=1]
  %14 = icmp ult i8* %12, %5                      ; <i1> [#uses=1]
  br i1 %14, label %bb1, label %bb3

bb3:                                              ; preds = %bb1, %bb, %entry
  ret i64 %2
}

declare i32 @__fd_getdents(i32, %struct.dirent*, i32)

define i32 @statfs(i8* %path, %struct.statfs* %buf32) nounwind {
entry:
  %0 = tail call i32 @__fd_statfs(i8* %path, %struct.statfs* %buf32) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_statfs(i8*, %struct.statfs*)

define i32 @ftruncate(i32 %fd, i64 %length) nounwind {
entry:
  %0 = tail call i32 @__fd_ftruncate(i32 %fd, i64 %length) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_ftruncate(i32, i64)
