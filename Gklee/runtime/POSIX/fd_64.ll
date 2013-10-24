; ModuleID = 'fd_64.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%struct.__fsid_t = type { [2 x i32] }
%struct.__va_list_tag = type { i32, i32, i8*, i8* }
%struct.dirent = type { i64, i64, i16, i8, [256 x i8] }
%struct.stat = type { i64, i64, i64, i32, i32, i32, i32, i64, i64, i64, i64, %struct.timespec, %struct.timespec, %struct.timespec, [3 x i64] }
%struct.statfs = type { i64, i64, i64, i64, i64, i64, i64, %struct.__fsid_t, i64, i64, [5 x i64] }
%struct.timespec = type { i64, i64 }

@__getdents64 = alias i32 (i32, %struct.dirent*, i32)* @getdents64 ; <i32 (i32, %struct.dirent*, i32)*> [#uses=0]

define i32 @"\01open64"(i8* %pathname, i32 %flags, ...) nounwind {
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

define i32 @getdents64(i32 %fd, %struct.dirent* %dirp, i32 %count) nounwind {
entry:
  %0 = tail call i32 @__fd_getdents(i32 %fd, %struct.dirent* %dirp, i32 %count) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_getdents(i32, %struct.dirent*, i32)

define weak i32 @"\01statfs64"(i8* %path, %struct.statfs* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_statfs(i8* %path, %struct.statfs* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_statfs(i8*, %struct.statfs*)

define i32 @ftruncate64(i32 %fd, i64 %length) nounwind {
entry:
  %0 = tail call i32 @__fd_ftruncate(i32 %fd, i64 %length) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_ftruncate(i32, i64)

define i32 @"\01fstat64"(i32 %fd, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_fstat(i32 %fd, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_fstat(i32, %struct.stat*)

define i32 @"\01__fxstat64"(i32 %vers, i32 %fd, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_fstat(i32 %fd, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

define i32 @"\01lstat64"(i8* %path, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_lstat(i8* %path, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_lstat(i8*, %struct.stat*)

define i32 @"\01__lxstat64"(i32 %vers, i8* %path, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_lstat(i8* %path, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

define i32 @"\01stat64"(i8* %path, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_stat(i8* %path, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

declare i32 @__fd_stat(i8*, %struct.stat*)

define i32 @"\01__xstat64"(i32 %vers, i8* %path, %struct.stat* %buf) nounwind {
entry:
  %0 = tail call i32 @__fd_stat(i8* %path, %struct.stat* %buf) nounwind ; <i32> [#uses=1]
  ret i32 %0
}

define i64 @"\01lseek64"(i32 %fd, i64 %offset, i32 %whence) nounwind {
entry:
  %0 = tail call i64 @__fd_lseek(i32 %fd, i64 %offset, i32 %whence) nounwind ; <i64> [#uses=1]
  ret i64 %0
}

declare i64 @__fd_lseek(i32, i64, i32)

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind

declare i32 @__fd_open(i8*, i32, i32)
