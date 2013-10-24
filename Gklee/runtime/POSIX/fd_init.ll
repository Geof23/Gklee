; ModuleID = 'fd_init.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%struct.exe_disk_file_t = type { i32, i8*, %struct.stat64* }
%struct.exe_file_system_t = type { i32, %struct.exe_disk_file_t*, %struct.exe_disk_file_t*, i32, %struct.exe_disk_file_t*, i32, i32*, i32*, i32*, i32*, i32*, i32*, i32* }
%struct.exe_file_t = type { i32, i32, i64, %struct.exe_disk_file_t* }
%struct.exe_sym_env_t = type { [32 x %struct.exe_file_t], i32, i32, i32 }
%struct.stat64 = type { i64, i64, i64, i32, i32, i32, i32, i64, i64, i64, i64, %struct.timespec, %struct.timespec, %struct.timespec, [3 x i64] }
%struct.timespec = type { i64, i64 }

@__exe_env = global %struct.exe_sym_env_t { [32 x %struct.exe_file_t] [%struct.exe_file_t { i32 0, i32 5, i64 0, %struct.exe_disk_file_t* null }, %struct.exe_file_t { i32 1, i32 9, i64 0, %struct.exe_disk_file_t* null }, %struct.exe_file_t { i32 2, i32 9, i64 0, %struct.exe_disk_file_t* null }, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer, %struct.exe_file_t zeroinitializer], i32 18, i32 0, i32 0 }, align 32 ; <%struct.exe_sym_env_t*> [#uses=4]
@.str = private constant [6 x i8] c"-stat\00", align 1 ; <[6 x i8]*> [#uses=1]
@.str1 = private constant [5 x i8] c"size\00", align 1 ; <[5 x i8]*> [#uses=1]
@.str2 = private constant [10 x i8] c"fd_init.c\00", align 1 ; <[10 x i8]*> [#uses=1]
@__PRETTY_FUNCTION__.4025 = internal constant [19 x i8] c"__create_new_dfile\00", align 16 ; <[19 x i8]*> [#uses=1]
@.str4 = private constant [2 x i8] c".\00", align 1 ; <[2 x i8]*> [#uses=1]
@__exe_fs = common global %struct.exe_file_system_t zeroinitializer, align 32 ; <%struct.exe_file_system_t*> [#uses=11]
@.str5 = private constant [6 x i8] c"stdin\00", align 1 ; <[6 x i8]*> [#uses=1]
@.str6 = private constant [10 x i8] c"read_fail\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str7 = private constant [11 x i8] c"write_fail\00", align 1 ; <[11 x i8]*> [#uses=1]
@.str8 = private constant [11 x i8] c"close_fail\00", align 1 ; <[11 x i8]*> [#uses=1]
@.str9 = private constant [15 x i8] c"ftruncate_fail\00", align 1 ; <[15 x i8]*> [#uses=1]
@.str10 = private constant [12 x i8] c"getcwd_fail\00", align 1 ; <[12 x i8]*> [#uses=1]
@.str11 = private constant [7 x i8] c"stdout\00", align 1 ; <[7 x i8]*> [#uses=1]
@.str12 = private constant [14 x i8] c"model_version\00", align 1 ; <[14 x i8]*> [#uses=1]

declare void @klee_make_symbolic(i8*, i32, i8*)

declare i32 @__xstat64(i32, i8*, %struct.stat64*) nounwind

define internal fastcc void @__create_new_dfile(%struct.exe_disk_file_t* nocapture %dfile, i32 %size, i8* %name, %struct.stat64* nocapture %defaults) nounwind {
entry:
  %sname = alloca [64 x i8], align 1              ; <[64 x i8]*> [#uses=3]
  %0 = malloc [18 x i64]                          ; <[18 x i64]*> [#uses=14]
  %.sub = getelementptr inbounds [18 x i64]* %0, i64 0, i64 0 ; <i64*> [#uses=1]
  %tmpcast = bitcast [18 x i64]* %0 to i8*        ; <i8*> [#uses=16]
  %1 = bitcast [18 x i64]* %0 to %struct.stat64*  ; <%struct.stat64*> [#uses=1]
  %2 = load i8* %name, align 1                    ; <i8> [#uses=1]
  %3 = icmp eq i8 %2, 0                           ; <i1> [#uses=1]
  %4 = ptrtoint i8* %name to i64                  ; <i64> [#uses=1]
  br i1 %3, label %bb2, label %bb

bb:                                               ; preds = %bb, %entry
  %indvar = phi i64 [ %tmp, %bb ], [ 0, %entry ]  ; <i64> [#uses=2]
  %5 = phi i64 [ %11, %bb ], [ 0, %entry ]        ; <i64> [#uses=1]
  %sp.010 = getelementptr i8* %name, i64 %indvar  ; <i8*> [#uses=1]
  %tmp = add i64 %indvar, 1                       ; <i64> [#uses=2]
  %scevgep = getelementptr i8* %name, i64 %tmp    ; <i8*> [#uses=2]
  %6 = load i8* %sp.010, align 1                  ; <i8> [#uses=1]
  %7 = getelementptr inbounds [64 x i8]* %sname, i64 0, i64 %5 ; <i8*> [#uses=1]
  store i8 %6, i8* %7, align 1
  %8 = load i8* %scevgep, align 1                 ; <i8> [#uses=1]
  %9 = icmp eq i8 %8, 0                           ; <i1> [#uses=1]
  %10 = ptrtoint i8* %scevgep to i64              ; <i64> [#uses=1]
  %11 = sub i64 %10, %4                           ; <i64> [#uses=2]
  br i1 %9, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  %.lcssa = phi i64 [ 0, %entry ], [ %11, %bb ]   ; <i64> [#uses=1]
  %12 = getelementptr inbounds [64 x i8]* %sname, i64 0, i64 %.lcssa ; <i8*> [#uses=1]
  call void @llvm.memcpy.i64(i8* %12, i8* getelementptr inbounds ([6 x i8]* @.str, i64 0, i64 0), i64 6, i32 1)
  %13 = icmp eq i32 %size, 0                      ; <i1> [#uses=1]
  br i1 %13, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  call void @__assert_fail(i8* getelementptr inbounds ([5 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8]* @.str2, i64 0, i64 0), i32 55, i8* getelementptr inbounds ([19 x i8]* @__PRETTY_FUNCTION__.4025, i64 0, i64 0)) noreturn nounwind
  unreachable

bb4:                                              ; preds = %bb2
  %14 = getelementptr inbounds %struct.exe_disk_file_t* %dfile, i64 0, i32 0 ; <i32*> [#uses=2]
  store i32 %size, i32* %14, align 8
  %15 = malloc i8, i32 %size                      ; <i8*> [#uses=2]
  %16 = getelementptr inbounds %struct.exe_disk_file_t* %dfile, i64 0, i32 1 ; <i8**> [#uses=1]
  store i8* %15, i8** %16, align 8
  call void @klee_make_symbolic(i8* %15, i32 %size, i8* %name) nounwind
  %sname5 = getelementptr inbounds [64 x i8]* %sname, i64 0, i64 0 ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %tmpcast, i32 144, i8* %sname5) nounwind
  %17 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 1 ; <i64*> [#uses=4]
  %18 = load i64* %17                             ; <i64> [#uses=1]
  %19 = trunc i64 %18 to i32                      ; <i32> [#uses=1]
  %20 = call i32 @klee_is_symbolic(i32 %19) nounwind ; <i32> [#uses=1]
  %21 = icmp eq i32 %20, 0                        ; <i1> [#uses=1]
  br i1 %21, label %bb6, label %bb8

bb6:                                              ; preds = %bb4
  %22 = load i64* %17                             ; <i64> [#uses=1]
  %23 = and i64 %22, 2147483647                   ; <i64> [#uses=1]
  %24 = icmp eq i64 %23, 0                        ; <i1> [#uses=1]
  br i1 %24, label %bb7, label %bb8

bb7:                                              ; preds = %bb6
  %25 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 1 ; <i64*> [#uses=1]
  %26 = load i64* %25, align 8                    ; <i64> [#uses=1]
  store i64 %26, i64* %17
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6, %bb4
  %27 = load i64* %17                             ; <i64> [#uses=1]
  %28 = and i64 %27, 2147483647                   ; <i64> [#uses=1]
  %29 = icmp ne i64 %28, 0                        ; <i1> [#uses=1]
  %30 = zext i1 %29 to i32                        ; <i32> [#uses=1]
  call void @klee_assume(i32 %30) nounwind
  %31 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 7 ; <i64*> [#uses=2]
  %32 = load i64* %31                             ; <i64> [#uses=1]
  %33 = icmp ult i64 %32, 65536                   ; <i1> [#uses=1]
  %34 = zext i1 %33 to i32                        ; <i32> [#uses=1]
  call void @klee_assume(i32 %34) nounwind
  %35 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 3 ; <i64*> [#uses=1]
  %36 = bitcast i64* %35 to i32*                  ; <i32*> [#uses=5]
  %37 = load i32* %36, align 8                    ; <i32> [#uses=1]
  %38 = and i32 %37, -61952                       ; <i32> [#uses=1]
  %39 = icmp eq i32 %38, 0                        ; <i1> [#uses=1]
  %40 = zext i1 %39 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %40) nounwind
  %41 = load i64* %.sub, align 8                  ; <i64> [#uses=1]
  %42 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 0 ; <i64*> [#uses=1]
  %43 = load i64* %42, align 8                    ; <i64> [#uses=1]
  %44 = icmp eq i64 %41, %43                      ; <i1> [#uses=1]
  %45 = zext i1 %44 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %45) nounwind
  %46 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 5 ; <i64*> [#uses=1]
  %47 = load i64* %46                             ; <i64> [#uses=1]
  %48 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 7 ; <i64*> [#uses=1]
  %49 = load i64* %48, align 8                    ; <i64> [#uses=1]
  %50 = icmp eq i64 %47, %49                      ; <i1> [#uses=1]
  %51 = zext i1 %50 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %51) nounwind
  %52 = load i32* %36, align 8                    ; <i32> [#uses=1]
  %53 = and i32 %52, 448                          ; <i32> [#uses=1]
  %54 = icmp eq i32 %53, 384                      ; <i1> [#uses=1]
  %55 = zext i1 %54 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %55) nounwind
  %56 = load i32* %36, align 8                    ; <i32> [#uses=1]
  %57 = and i32 %56, 56                           ; <i32> [#uses=1]
  %58 = icmp eq i32 %57, 16                       ; <i1> [#uses=1]
  %59 = zext i1 %58 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %59) nounwind
  %60 = load i32* %36, align 8                    ; <i32> [#uses=1]
  %61 = and i32 %60, 7                            ; <i32> [#uses=1]
  %62 = icmp eq i32 %61, 2                        ; <i1> [#uses=1]
  %63 = zext i1 %62 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %63) nounwind
  %64 = load i32* %36, align 8                    ; <i32> [#uses=1]
  %65 = and i32 %64, 61440                        ; <i32> [#uses=1]
  %66 = icmp eq i32 %65, 32768                    ; <i1> [#uses=1]
  %67 = zext i1 %66 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %67) nounwind
  %68 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 2 ; <i64*> [#uses=1]
  %69 = load i64* %68                             ; <i64> [#uses=1]
  %70 = icmp eq i64 %69, 1                        ; <i1> [#uses=1]
  %71 = zext i1 %70 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %71) nounwind
  %72 = getelementptr inbounds i8* %tmpcast, i64 28 ; <i8*> [#uses=1]
  %73 = bitcast i8* %72 to i32*                   ; <i32*> [#uses=1]
  %74 = load i32* %73, align 4                    ; <i32> [#uses=1]
  %75 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 4 ; <i32*> [#uses=1]
  %76 = load i32* %75, align 4                    ; <i32> [#uses=1]
  %77 = icmp eq i32 %74, %76                      ; <i1> [#uses=1]
  %78 = zext i1 %77 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %78) nounwind
  %79 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 4 ; <i64*> [#uses=1]
  %80 = bitcast i64* %79 to i32*                  ; <i32*> [#uses=1]
  %81 = load i32* %80, align 8                    ; <i32> [#uses=1]
  %82 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 5 ; <i32*> [#uses=1]
  %83 = load i32* %82, align 8                    ; <i32> [#uses=1]
  %84 = icmp eq i32 %81, %83                      ; <i1> [#uses=1]
  %85 = zext i1 %84 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %85) nounwind
  %86 = load i64* %31                             ; <i64> [#uses=1]
  %87 = icmp eq i64 %86, 4096                     ; <i1> [#uses=1]
  %88 = zext i1 %87 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %88) nounwind
  %89 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 9 ; <i64*> [#uses=1]
  %90 = load i64* %89                             ; <i64> [#uses=1]
  %91 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 11, i32 0 ; <i64*> [#uses=1]
  %92 = load i64* %91, align 8                    ; <i64> [#uses=1]
  %93 = icmp eq i64 %90, %92                      ; <i1> [#uses=1]
  %94 = zext i1 %93 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %94) nounwind
  %95 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 11 ; <i64*> [#uses=1]
  %96 = load i64* %95                             ; <i64> [#uses=1]
  %97 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 12, i32 0 ; <i64*> [#uses=1]
  %98 = load i64* %97, align 8                    ; <i64> [#uses=1]
  %99 = icmp eq i64 %96, %98                      ; <i1> [#uses=1]
  %100 = zext i1 %99 to i32                       ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %100) nounwind
  %101 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 13 ; <i64*> [#uses=1]
  %102 = load i64* %101                           ; <i64> [#uses=1]
  %103 = getelementptr inbounds %struct.stat64* %defaults, i64 0, i32 13, i32 0 ; <i64*> [#uses=1]
  %104 = load i64* %103, align 8                  ; <i64> [#uses=1]
  %105 = icmp eq i64 %102, %104                   ; <i1> [#uses=1]
  %106 = zext i1 %105 to i32                      ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %tmpcast, i32 %106) nounwind
  %107 = load i32* %14, align 8                   ; <i32> [#uses=1]
  %108 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 6 ; <i64*> [#uses=1]
  %.c = zext i32 %107 to i64                      ; <i64> [#uses=1]
  store i64 %.c, i64* %108
  %109 = getelementptr inbounds [18 x i64]* %0, i64 0, i64 8 ; <i64*> [#uses=1]
  store i64 8, i64* %109
  %110 = getelementptr inbounds %struct.exe_disk_file_t* %dfile, i64 0, i32 2 ; <%struct.stat64**> [#uses=1]
  store %struct.stat64* %1, %struct.stat64** %110, align 8
  ret void
}

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

declare void @__assert_fail(i8*, i8*, i32, i8*) noreturn nounwind

declare i32 @klee_is_symbolic(i32)

declare void @klee_assume(i32)

declare void @klee_prefer_cex(i8*, i32)

define void @klee_init_fds(i32 %n_files, i32 %file_length, i32 %sym_stdout_flag, i32 %save_all_writes_flag, i32 %max_failures) nounwind {
entry:
  %x.i = alloca i32, align 4                      ; <i32*> [#uses=2]
  %s = alloca %struct.stat64, align 8             ; <%struct.stat64*> [#uses=4]
  %name = alloca [7 x i8], align 1                ; <[7 x i8]*> [#uses=7]
  %0 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 0 ; <i8*> [#uses=3]
  store i8 63, i8* %0, align 1
  %1 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 1 ; <i8*> [#uses=1]
  store i8 45, i8* %1, align 1
  %2 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 2 ; <i8*> [#uses=1]
  store i8 100, i8* %2, align 1
  %3 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 3 ; <i8*> [#uses=1]
  store i8 97, i8* %3, align 1
  %4 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 4 ; <i8*> [#uses=1]
  store i8 116, i8* %4, align 1
  %5 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 5 ; <i8*> [#uses=1]
  store i8 97, i8* %5, align 1
  %6 = getelementptr inbounds [7 x i8]* %name, i64 0, i64 6 ; <i8*> [#uses=1]
  store i8 0, i8* %6, align 1
  %7 = call i32 @__xstat64(i32 1, i8* getelementptr inbounds ([2 x i8]* @.str4, i64 0, i64 0), %struct.stat64* %s) nounwind ; <i32> [#uses=0]
  store i32 %n_files, i32* getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 0), align 32
  %8 = malloc %struct.exe_disk_file_t, i32 %n_files ; <%struct.exe_disk_file_t*> [#uses=1]
  store %struct.exe_disk_file_t* %8, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 4), align 32
  %9 = icmp eq i32 %n_files, 0                    ; <i1> [#uses=1]
  br i1 %9, label %bb3, label %bb.nph

bb.nph:                                           ; preds = %entry
  %tmp = zext i32 %n_files to i64                 ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i64> [#uses=3]
  %indvar22 = trunc i64 %indvar to i8             ; <i8> [#uses=1]
  %tmp21 = add i8 %indvar22, 65                   ; <i8> [#uses=1]
  store i8 %tmp21, i8* %0, align 1
  %10 = load %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 4), align 32 ; <%struct.exe_disk_file_t*> [#uses=1]
  %scevgep = getelementptr %struct.exe_disk_file_t* %10, i64 %indvar ; <%struct.exe_disk_file_t*> [#uses=1]
  call fastcc void @__create_new_dfile(%struct.exe_disk_file_t* %scevgep, i32 %file_length, i8* %0, %struct.stat64* %s) nounwind
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %tmp      ; <i1> [#uses=1]
  br i1 %exitcond, label %bb3, label %bb

bb3:                                              ; preds = %bb, %entry
  %11 = icmp eq i32 %file_length, 0               ; <i1> [#uses=1]
  br i1 %11, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  %12 = malloc %struct.exe_disk_file_t            ; <%struct.exe_disk_file_t*> [#uses=2]
  store %struct.exe_disk_file_t* %12, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 1), align 8
  call fastcc void @__create_new_dfile(%struct.exe_disk_file_t* %12, i32 %file_length, i8* getelementptr inbounds ([6 x i8]* @.str5, i64 0, i64 0), %struct.stat64* %s) nounwind
  %13 = load %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 1), align 8 ; <%struct.exe_disk_file_t*> [#uses=1]
  store %struct.exe_disk_file_t* %13, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_sym_env_t* @__exe_env, i64 0, i32 0, i64 0, i32 3), align 16
  br label %bb6

bb5:                                              ; preds = %bb3
  store %struct.exe_disk_file_t* null, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 1), align 8
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4
  store i32 %max_failures, i32* getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 5), align 8
  %14 = icmp eq i32 %max_failures, 0              ; <i1> [#uses=1]
  br i1 %14, label %bb8, label %bb7

bb7:                                              ; preds = %bb6
  %15 = malloc i32                                ; <i32*> [#uses=2]
  store i32* %15, i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 6), align 16
  %16 = malloc i32                                ; <i32*> [#uses=1]
  store i32* %16, i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 7), align 8
  %17 = malloc i32                                ; <i32*> [#uses=1]
  store i32* %17, i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 8), align 32
  %18 = malloc i32                                ; <i32*> [#uses=1]
  store i32* %18, i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 9), align 8
  %19 = malloc i32                                ; <i32*> [#uses=1]
  store i32* %19, i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 10), align 16
  %20 = bitcast i32* %15 to i8*                   ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %20, i32 4, i8* getelementptr inbounds ([10 x i8]* @.str6, i64 0, i64 0)) nounwind
  %21 = load i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 7), align 8 ; <i32*> [#uses=1]
  %22 = bitcast i32* %21 to i8*                   ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %22, i32 4, i8* getelementptr inbounds ([11 x i8]* @.str7, i64 0, i64 0)) nounwind
  %23 = load i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 8), align 32 ; <i32*> [#uses=1]
  %24 = bitcast i32* %23 to i8*                   ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %24, i32 4, i8* getelementptr inbounds ([11 x i8]* @.str8, i64 0, i64 0)) nounwind
  %25 = load i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 9), align 8 ; <i32*> [#uses=1]
  %26 = bitcast i32* %25 to i8*                   ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %26, i32 4, i8* getelementptr inbounds ([15 x i8]* @.str9, i64 0, i64 0)) nounwind
  %27 = load i32** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 10), align 16 ; <i32*> [#uses=1]
  %28 = bitcast i32* %27 to i8*                   ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %28, i32 4, i8* getelementptr inbounds ([12 x i8]* @.str10, i64 0, i64 0)) nounwind
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  %29 = icmp eq i32 %sym_stdout_flag, 0           ; <i1> [#uses=1]
  br i1 %29, label %bb10, label %bb9

bb9:                                              ; preds = %bb8
  %30 = malloc %struct.exe_disk_file_t            ; <%struct.exe_disk_file_t*> [#uses=2]
  store %struct.exe_disk_file_t* %30, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 2), align 16
  call fastcc void @__create_new_dfile(%struct.exe_disk_file_t* %30, i32 1024, i8* getelementptr inbounds ([7 x i8]* @.str11, i64 0, i64 0), %struct.stat64* %s) nounwind
  %31 = load %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 2), align 16 ; <%struct.exe_disk_file_t*> [#uses=1]
  store %struct.exe_disk_file_t* %31, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_sym_env_t* @__exe_env, i64 0, i32 0, i64 1, i32 3), align 8
  store i32 0, i32* getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 3), align 8
  br label %bb11

bb10:                                             ; preds = %bb8
  store %struct.exe_disk_file_t* null, %struct.exe_disk_file_t** getelementptr inbounds (%struct.exe_file_system_t* @__exe_fs, i64 0, i32 2), align 16
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  store i32 %save_all_writes_flag, i32* getelementptr inbounds (%struct.exe_sym_env_t* @__exe_env, i64 0, i32 3), align 8
  %x1.i = bitcast i32* %x.i to i8*                ; <i8*> [#uses=1]
  call void @klee_make_symbolic(i8* %x1.i, i32 4, i8* getelementptr inbounds ([14 x i8]* @.str12, i64 0, i64 0)) nounwind
  %32 = load i32* %x.i, align 4                   ; <i32> [#uses=2]
  store i32 %32, i32* getelementptr inbounds (%struct.exe_sym_env_t* @__exe_env, i64 0, i32 2), align 4
  %33 = icmp eq i32 %32, 1                        ; <i1> [#uses=1]
  %34 = zext i1 %33 to i32                        ; <i32> [#uses=1]
  call void @klee_assume(i32 %34) nounwind
  ret void
}
