; ModuleID = 'klee_init_env.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private constant [16 x i8] c"klee_init_env.c\00", align 1 ; <[16 x i8]*> [#uses=1]
@.str1 = private constant [9 x i8] c"user.err\00", align 1 ; <[9 x i8]*> [#uses=1]
@.str2 = private constant [37 x i8] c"too many arguments for klee_init_env\00", align 1 ; <[37 x i8]*> [#uses=1]
@.str4 = private constant [7 x i8] c"--help\00", align 1 ; <[7 x i8]*> [#uses=1]
@.str5 = private constant [593 x i8] c"klee_init_env\0A\0Ausage: (klee_init_env) [options] [program arguments]\0A  -sym-arg <N>              - Replace by a symbolic argument with length N\0A  -sym-args <MIN> <MAX> <N> - Replace by at least MIN arguments and at most\0A                              MAX arguments, each with maximum length N\0A  -sym-files <NUM> <N>      - Make stdin and up to NUM symbolic files, each\0A                              with maximum size N.\0A  -sym-stdout               - Make stdout symbolic.\0A  -max-fail <N>             - Allow up to <N> injected failures\0A  -fd-fail                  - Shortcut for '-max-fail 1'\0A\0A\00", align 1 ; <[593 x i8]*> [#uses=1]
@.str6 = private constant [10 x i8] c"--sym-arg\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str7 = private constant [9 x i8] c"-sym-arg\00", align 1 ; <[9 x i8]*> [#uses=1]
@.str8 = private constant [48 x i8] c"--sym-arg expects an integer argument <max-len>\00", align 1 ; <[48 x i8]*> [#uses=1]
@.str9 = private constant [11 x i8] c"--sym-args\00", align 1 ; <[11 x i8]*> [#uses=1]
@.str10 = private constant [10 x i8] c"-sym-args\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str11 = private constant [77 x i8] c"--sym-args expects three integer arguments <min-argvs> <max-argvs> <max-len>\00", align 1 ; <[77 x i8]*> [#uses=1]
@.str12 = private constant [7 x i8] c"n_args\00", align 1 ; <[7 x i8]*> [#uses=1]
@.str13 = private constant [12 x i8] c"--sym-files\00", align 1 ; <[12 x i8]*> [#uses=1]
@.str14 = private constant [11 x i8] c"-sym-files\00", align 1 ; <[11 x i8]*> [#uses=1]
@.str15 = private constant [72 x i8] c"--sym-files expects two integer arguments <no-sym-files> <sym-file-len>\00", align 1 ; <[72 x i8]*> [#uses=1]
@.str16 = private constant [13 x i8] c"--sym-stdout\00", align 1 ; <[13 x i8]*> [#uses=1]
@.str17 = private constant [12 x i8] c"-sym-stdout\00", align 1 ; <[12 x i8]*> [#uses=1]
@.str18 = private constant [18 x i8] c"--save-all-writes\00", align 1 ; <[18 x i8]*> [#uses=1]
@.str19 = private constant [17 x i8] c"-save-all-writes\00", align 1 ; <[17 x i8]*> [#uses=1]
@.str20 = private constant [10 x i8] c"--fd-fail\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str21 = private constant [9 x i8] c"-fd-fail\00", align 1 ; <[9 x i8]*> [#uses=1]
@.str22 = private constant [11 x i8] c"--max-fail\00", align 1 ; <[11 x i8]*> [#uses=1]
@.str23 = private constant [10 x i8] c"-max-fail\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str24 = private constant [54 x i8] c"--max-fail expects an integer argument <max-failures>\00", align 1 ; <[54 x i8]*> [#uses=1]

define void @klee_init_env(i32* nocapture %argcPtr, i8*** nocapture %argvPtr) nounwind {
entry:
  %sym_arg_name = alloca [5 x i8], align 1        ; <[5 x i8]*> [#uses=5]
  %new_argv = alloca [1024 x i8*], align 8        ; <[1024 x i8*]*> [#uses=5]
  %0 = load i32* %argcPtr, align 4                ; <i32> [#uses=6]
  %1 = load i8*** %argvPtr, align 8               ; <i8**> [#uses=9]
  %2 = getelementptr inbounds [5 x i8]* %sym_arg_name, i64 0, i64 0 ; <i8*> [#uses=4]
  store i8 97, i8* %2, align 1
  %3 = getelementptr inbounds [5 x i8]* %sym_arg_name, i64 0, i64 1 ; <i8*> [#uses=1]
  store i8 114, i8* %3, align 1
  %4 = getelementptr inbounds [5 x i8]* %sym_arg_name, i64 0, i64 2 ; <i8*> [#uses=1]
  store i8 103, i8* %4, align 1
  %5 = getelementptr inbounds [5 x i8]* %sym_arg_name, i64 0, i64 3 ; <i8*> [#uses=4]
  store i8 0, i8* %5, align 1
  %6 = getelementptr inbounds [5 x i8]* %sym_arg_name, i64 0, i64 4 ; <i8*> [#uses=1]
  store i8 0, i8* %6, align 1
  %7 = icmp eq i32 %0, 2                          ; <i1> [#uses=1]
  br i1 %7, label %bb, label %bb42

bb:                                               ; preds = %entry
  %8 = getelementptr inbounds i8** %1, i64 1      ; <i8**> [#uses=1]
  %9 = load i8** %8, align 8                      ; <i8*> [#uses=1]
  br label %bb3.i

bb.i:                                             ; preds = %bb3.i
  %10 = icmp eq i8 %11, 0                         ; <i1> [#uses=1]
  br i1 %10, label %bb4, label %bb2.i

bb2.i:                                            ; preds = %bb.i
  %indvar.next.i = add i64 %indvar.i, 1           ; <i64> [#uses=1]
  br label %bb3.i

bb3.i:                                            ; preds = %bb2.i, %bb
  %indvar.i = phi i64 [ 0, %bb ], [ %indvar.next.i, %bb2.i ] ; <i64> [#uses=3]
  %b_addr.0.i = getelementptr [7 x i8]* @.str4, i64 0, i64 %indvar.i ; <i8*> [#uses=1]
  %a_addr.0.i = getelementptr i8* %9, i64 %indvar.i ; <i8*> [#uses=1]
  %11 = load i8* %a_addr.0.i, align 1             ; <i8> [#uses=2]
  %12 = load i8* %b_addr.0.i, align 1             ; <i8> [#uses=1]
  %13 = icmp eq i8 %11, %12                       ; <i1> [#uses=1]
  br i1 %13, label %bb.i, label %bb42

bb4:                                              ; preds = %bb.i
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([593 x i8]* @.str5, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5:                                              ; preds = %bb42
  %14 = sext i32 %k.0 to i64                      ; <i64> [#uses=1]
  %15 = getelementptr inbounds i8** %1, i64 %14   ; <i8**> [#uses=1]
  %16 = load i8** %15, align 8                    ; <i8*> [#uses=15]
  br label %bb3.i268

bb.i262:                                          ; preds = %bb3.i268
  %17 = icmp eq i8 %18, 0                         ; <i1> [#uses=1]
  br i1 %17, label %bb7, label %bb2.i264

bb2.i264:                                         ; preds = %bb.i262
  %indvar.next.i263 = add i64 %indvar.i265, 1     ; <i64> [#uses=1]
  br label %bb3.i268

bb3.i268:                                         ; preds = %bb2.i264, %bb5
  %indvar.i265 = phi i64 [ 0, %bb5 ], [ %indvar.next.i263, %bb2.i264 ] ; <i64> [#uses=3]
  %b_addr.0.i266 = getelementptr [10 x i8]* @.str6, i64 0, i64 %indvar.i265 ; <i8*> [#uses=1]
  %a_addr.0.i267 = getelementptr i8* %16, i64 %indvar.i265 ; <i8*> [#uses=1]
  %18 = load i8* %a_addr.0.i267, align 1          ; <i8> [#uses=2]
  %19 = load i8* %b_addr.0.i266, align 1          ; <i8> [#uses=1]
  %20 = icmp eq i8 %18, %19                       ; <i1> [#uses=1]
  br i1 %20, label %bb.i262, label %bb3.i258

bb.i252:                                          ; preds = %bb3.i258
  %21 = icmp eq i8 %22, 0                         ; <i1> [#uses=1]
  br i1 %21, label %bb7, label %bb2.i254

bb2.i254:                                         ; preds = %bb.i252
  %indvar.next.i253 = add i64 %indvar.i255, 1     ; <i64> [#uses=1]
  br label %bb3.i258

bb3.i258:                                         ; preds = %bb2.i254, %bb3.i268
  %indvar.i255 = phi i64 [ %indvar.next.i253, %bb2.i254 ], [ 0, %bb3.i268 ] ; <i64> [#uses=3]
  %b_addr.0.i256 = getelementptr [9 x i8]* @.str7, i64 0, i64 %indvar.i255 ; <i8*> [#uses=1]
  %a_addr.0.i257 = getelementptr i8* %16, i64 %indvar.i255 ; <i8*> [#uses=1]
  %22 = load i8* %a_addr.0.i257, align 1          ; <i8> [#uses=2]
  %23 = load i8* %b_addr.0.i256, align 1          ; <i8> [#uses=1]
  %24 = icmp eq i8 %22, %23                       ; <i1> [#uses=1]
  br i1 %24, label %bb.i252, label %bb3.i226

bb7:                                              ; preds = %bb.i252, %bb.i262
  %25 = add i32 %k.0, 1                           ; <i32> [#uses=2]
  %26 = icmp eq i32 %25, %0                       ; <i1> [#uses=1]
  br i1 %26, label %bb8, label %bb9

bb8:                                              ; preds = %bb7
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([48 x i8]* @.str8, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb9:                                              ; preds = %bb7
  %27 = sext i32 %25 to i64                       ; <i64> [#uses=1]
  %28 = getelementptr inbounds i8** %1, i64 %27   ; <i8**> [#uses=1]
  %29 = load i8** %28, align 8                    ; <i8*> [#uses=2]
  %30 = add i32 %k.0, 2                           ; <i32> [#uses=1]
  %31 = load i8* %29, align 1                     ; <i8> [#uses=1]
  %32 = icmp eq i8 %31, 0                         ; <i1> [#uses=1]
  br i1 %32, label %bb.i241, label %bb5.i249

bb.i241:                                          ; preds = %bb9
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([48 x i8]* @.str8, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i242:                                         ; preds = %bb5.i249
  %33 = add i8 %40, -48                           ; <i8> [#uses=1]
  %34 = icmp ugt i8 %33, 9                        ; <i1> [#uses=1]
  br i1 %34, label %bb4.i245, label %bb3.i244

bb3.i244:                                         ; preds = %bb2.i242
  %35 = mul i64 %res.0.i247, 10                   ; <i64> [#uses=1]
  %36 = sext i8 %40 to i32                        ; <i32> [#uses=1]
  %37 = add i32 %36, -48                          ; <i32> [#uses=1]
  %38 = sext i32 %37 to i64                       ; <i64> [#uses=1]
  %39 = add nsw i64 %38, %35                      ; <i64> [#uses=1]
  %indvar.next.i243 = add i64 %indvar.i246, 1     ; <i64> [#uses=1]
  br label %bb5.i249

bb4.i245:                                         ; preds = %bb2.i242
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([48 x i8]* @.str8, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i249:                                         ; preds = %bb3.i244, %bb9
  %indvar.i246 = phi i64 [ %indvar.next.i243, %bb3.i244 ], [ 0, %bb9 ] ; <i64> [#uses=2]
  %res.0.i247 = phi i64 [ %39, %bb3.i244 ], [ 0, %bb9 ] ; <i64> [#uses=3]
  %s_addr.0.i248 = getelementptr i8* %29, i64 %indvar.i246 ; <i8*> [#uses=1]
  %40 = load i8* %s_addr.0.i248, align 1          ; <i8> [#uses=3]
  %41 = icmp eq i8 %40, 0                         ; <i1> [#uses=1]
  br i1 %41, label %__str_to_int.exit250, label %bb2.i242

__str_to_int.exit250:                             ; preds = %bb5.i249
  %42 = trunc i64 %res.0.i247 to i32              ; <i32> [#uses=3]
  %43 = trunc i32 %sym_arg_num.0 to i8            ; <i8> [#uses=1]
  %44 = add i8 %43, 48                            ; <i8> [#uses=1]
  store i8 %44, i8* %5, align 1
  %45 = add i32 %sym_arg_num.0, 1                 ; <i32> [#uses=1]
  %46 = add nsw i32 %42, 1                        ; <i32> [#uses=2]
  %47 = malloc i8, i32 %46                        ; <i8*> [#uses=6]
  call void @klee_mark_global(i8* %47) nounwind
  call void @klee_make_symbolic(i8* %47, i32 %46, i8* %2) nounwind
  %48 = icmp sgt i32 %42, 0                       ; <i1> [#uses=1]
  br i1 %48, label %bb.nph.i233, label %__get_sym_str.exit240

bb.nph.i233:                                      ; preds = %__str_to_int.exit250
  %tmp495 = and i64 %res.0.i247, 4294967295       ; <i64> [#uses=1]
  br label %bb.i238

bb.i238:                                          ; preds = %bb.i238, %bb.nph.i233
  %indvar.i234 = phi i64 [ 0, %bb.nph.i233 ], [ %indvar.next.i236, %bb.i238 ] ; <i64> [#uses=2]
  %scevgep.i235 = getelementptr i8* %47, i64 %indvar.i234 ; <i8*> [#uses=1]
  %49 = load i8* %scevgep.i235, align 1           ; <i8> [#uses=1]
  %50 = add i8 %49, -32                           ; <i8> [#uses=1]
  %51 = icmp ult i8 %50, 95                       ; <i1> [#uses=1]
  %52 = zext i1 %51 to i32                        ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %47, i32 %52) nounwind
  %indvar.next.i236 = add i64 %indvar.i234, 1     ; <i64> [#uses=2]
  %exitcond496 = icmp eq i64 %indvar.next.i236, %tmp495 ; <i1> [#uses=1]
  br i1 %exitcond496, label %__get_sym_str.exit240, label %bb.i238

__get_sym_str.exit240:                            ; preds = %bb.i238, %__str_to_int.exit250
  %53 = sext i32 %42 to i64                       ; <i64> [#uses=1]
  %54 = getelementptr inbounds i8* %47, i64 %53   ; <i8*> [#uses=1]
  store i8 0, i8* %54, align 1
  %55 = icmp eq i32 %new_argc.0, 1024             ; <i1> [#uses=1]
  br i1 %55, label %bb.i230, label %__add_arg.exit231

bb.i230:                                          ; preds = %__get_sym_str.exit240
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([37 x i8]* @.str2, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

__add_arg.exit231:                                ; preds = %__get_sym_str.exit240
  %56 = sext i32 %new_argc.0 to i64               ; <i64> [#uses=1]
  %57 = getelementptr inbounds [1024 x i8*]* %new_argv, i64 0, i64 %56 ; <i8**> [#uses=1]
  store i8* %47, i8** %57, align 8
  %58 = add i32 %new_argc.0, 1                    ; <i32> [#uses=1]
  br label %bb42

bb.i220:                                          ; preds = %bb3.i226
  %59 = icmp eq i8 %60, 0                         ; <i1> [#uses=1]
  br i1 %59, label %bb14, label %bb2.i222

bb2.i222:                                         ; preds = %bb.i220
  %indvar.next.i221 = add i64 %indvar.i223, 1     ; <i64> [#uses=1]
  br label %bb3.i226

bb3.i226:                                         ; preds = %bb2.i222, %bb3.i258
  %indvar.i223 = phi i64 [ %indvar.next.i221, %bb2.i222 ], [ 0, %bb3.i258 ] ; <i64> [#uses=3]
  %b_addr.0.i224 = getelementptr [11 x i8]* @.str9, i64 0, i64 %indvar.i223 ; <i8*> [#uses=1]
  %a_addr.0.i225 = getelementptr i8* %16, i64 %indvar.i223 ; <i8*> [#uses=1]
  %60 = load i8* %a_addr.0.i225, align 1          ; <i8> [#uses=2]
  %61 = load i8* %b_addr.0.i224, align 1          ; <i8> [#uses=1]
  %62 = icmp eq i8 %60, %61                       ; <i1> [#uses=1]
  br i1 %62, label %bb.i220, label %bb3.i216

bb.i210:                                          ; preds = %bb3.i216
  %63 = icmp eq i8 %64, 0                         ; <i1> [#uses=1]
  br i1 %63, label %bb14, label %bb2.i212

bb2.i212:                                         ; preds = %bb.i210
  %indvar.next.i211 = add i64 %indvar.i213, 1     ; <i64> [#uses=1]
  br label %bb3.i216

bb3.i216:                                         ; preds = %bb2.i212, %bb3.i226
  %indvar.i213 = phi i64 [ %indvar.next.i211, %bb2.i212 ], [ 0, %bb3.i226 ] ; <i64> [#uses=3]
  %b_addr.0.i214 = getelementptr [10 x i8]* @.str10, i64 0, i64 %indvar.i213 ; <i8*> [#uses=1]
  %a_addr.0.i215 = getelementptr i8* %16, i64 %indvar.i213 ; <i8*> [#uses=1]
  %64 = load i8* %a_addr.0.i215, align 1          ; <i8> [#uses=2]
  %65 = load i8* %b_addr.0.i214, align 1          ; <i8> [#uses=1]
  %66 = icmp eq i8 %64, %65                       ; <i1> [#uses=1]
  br i1 %66, label %bb.i210, label %bb3.i169

bb14:                                             ; preds = %bb.i210, %bb.i220
  %67 = add i32 %k.0, 3                           ; <i32> [#uses=2]
  %68 = icmp slt i32 %67, %0                      ; <i1> [#uses=1]
  br i1 %68, label %bb16, label %bb15

bb15:                                             ; preds = %bb14
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb16:                                             ; preds = %bb14
  %69 = add i32 %k.0, 1                           ; <i32> [#uses=1]
  %70 = sext i32 %69 to i64                       ; <i64> [#uses=1]
  %71 = getelementptr inbounds i8** %1, i64 %70   ; <i8**> [#uses=1]
  %72 = load i8** %71, align 8                    ; <i8*> [#uses=2]
  %73 = add i32 %k.0, 2                           ; <i32> [#uses=1]
  %74 = load i8* %72, align 1                     ; <i8> [#uses=1]
  %75 = icmp eq i8 %74, 0                         ; <i1> [#uses=1]
  br i1 %75, label %bb.i199, label %bb5.i207

bb.i199:                                          ; preds = %bb16
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i200:                                         ; preds = %bb5.i207
  %76 = add i8 %83, -48                           ; <i8> [#uses=1]
  %77 = icmp ugt i8 %76, 9                        ; <i1> [#uses=1]
  br i1 %77, label %bb4.i203, label %bb3.i202

bb3.i202:                                         ; preds = %bb2.i200
  %78 = mul i64 %res.0.i205, 10                   ; <i64> [#uses=1]
  %79 = sext i8 %83 to i32                        ; <i32> [#uses=1]
  %80 = add i32 %79, -48                          ; <i32> [#uses=1]
  %81 = sext i32 %80 to i64                       ; <i64> [#uses=1]
  %82 = add nsw i64 %81, %78                      ; <i64> [#uses=1]
  %indvar.next.i201 = add i64 %indvar.i204, 1     ; <i64> [#uses=1]
  br label %bb5.i207

bb4.i203:                                         ; preds = %bb2.i200
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i207:                                         ; preds = %bb3.i202, %bb16
  %indvar.i204 = phi i64 [ %indvar.next.i201, %bb3.i202 ], [ 0, %bb16 ] ; <i64> [#uses=2]
  %res.0.i205 = phi i64 [ %82, %bb3.i202 ], [ 0, %bb16 ] ; <i64> [#uses=2]
  %s_addr.0.i206 = getelementptr i8* %72, i64 %indvar.i204 ; <i8*> [#uses=1]
  %83 = load i8* %s_addr.0.i206, align 1          ; <i8> [#uses=3]
  %84 = icmp eq i8 %83, 0                         ; <i1> [#uses=1]
  br i1 %84, label %__str_to_int.exit208, label %bb2.i200

__str_to_int.exit208:                             ; preds = %bb5.i207
  %85 = trunc i64 %res.0.i205 to i32              ; <i32> [#uses=1]
  %86 = sext i32 %73 to i64                       ; <i64> [#uses=1]
  %87 = getelementptr inbounds i8** %1, i64 %86   ; <i8**> [#uses=1]
  %88 = load i8** %87, align 8                    ; <i8*> [#uses=2]
  %89 = load i8* %88, align 1                     ; <i8> [#uses=1]
  %90 = icmp eq i8 %89, 0                         ; <i1> [#uses=1]
  br i1 %90, label %bb.i189, label %bb5.i197

bb.i189:                                          ; preds = %__str_to_int.exit208
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i190:                                         ; preds = %bb5.i197
  %91 = add i8 %98, -48                           ; <i8> [#uses=1]
  %92 = icmp ugt i8 %91, 9                        ; <i1> [#uses=1]
  br i1 %92, label %bb4.i193, label %bb3.i192

bb3.i192:                                         ; preds = %bb2.i190
  %93 = mul i64 %res.0.i195, 10                   ; <i64> [#uses=1]
  %94 = sext i8 %98 to i32                        ; <i32> [#uses=1]
  %95 = add i32 %94, -48                          ; <i32> [#uses=1]
  %96 = sext i32 %95 to i64                       ; <i64> [#uses=1]
  %97 = add nsw i64 %96, %93                      ; <i64> [#uses=1]
  %indvar.next.i191 = add i64 %indvar.i194, 1     ; <i64> [#uses=1]
  br label %bb5.i197

bb4.i193:                                         ; preds = %bb2.i190
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i197:                                         ; preds = %bb3.i192, %__str_to_int.exit208
  %indvar.i194 = phi i64 [ %indvar.next.i191, %bb3.i192 ], [ 0, %__str_to_int.exit208 ] ; <i64> [#uses=2]
  %res.0.i195 = phi i64 [ %97, %bb3.i192 ], [ 0, %__str_to_int.exit208 ] ; <i64> [#uses=2]
  %s_addr.0.i196 = getelementptr i8* %88, i64 %indvar.i194 ; <i8*> [#uses=1]
  %98 = load i8* %s_addr.0.i196, align 1          ; <i8> [#uses=3]
  %99 = icmp eq i8 %98, 0                         ; <i1> [#uses=1]
  br i1 %99, label %__str_to_int.exit198, label %bb2.i190

__str_to_int.exit198:                             ; preds = %bb5.i197
  %100 = trunc i64 %res.0.i195 to i32             ; <i32> [#uses=1]
  %101 = sext i32 %67 to i64                      ; <i64> [#uses=1]
  %102 = getelementptr inbounds i8** %1, i64 %101 ; <i8**> [#uses=1]
  %103 = load i8** %102, align 8                  ; <i8*> [#uses=2]
  %104 = add i32 %k.0, 4                          ; <i32> [#uses=1]
  %105 = load i8* %103, align 1                   ; <i8> [#uses=1]
  %106 = icmp eq i8 %105, 0                       ; <i1> [#uses=1]
  br i1 %106, label %bb.i179, label %bb5.i187

bb.i179:                                          ; preds = %__str_to_int.exit198
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i180:                                         ; preds = %bb5.i187
  %107 = add i8 %114, -48                         ; <i8> [#uses=1]
  %108 = icmp ugt i8 %107, 9                      ; <i1> [#uses=1]
  br i1 %108, label %bb4.i183, label %bb3.i182

bb3.i182:                                         ; preds = %bb2.i180
  %109 = mul i64 %res.0.i185, 10                  ; <i64> [#uses=1]
  %110 = sext i8 %114 to i32                      ; <i32> [#uses=1]
  %111 = add i32 %110, -48                        ; <i32> [#uses=1]
  %112 = sext i32 %111 to i64                     ; <i64> [#uses=1]
  %113 = add nsw i64 %112, %109                   ; <i64> [#uses=1]
  %indvar.next.i181 = add i64 %indvar.i184, 1     ; <i64> [#uses=1]
  br label %bb5.i187

bb4.i183:                                         ; preds = %bb2.i180
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([77 x i8]* @.str11, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i187:                                         ; preds = %bb3.i182, %__str_to_int.exit198
  %indvar.i184 = phi i64 [ %indvar.next.i181, %bb3.i182 ], [ 0, %__str_to_int.exit198 ] ; <i64> [#uses=2]
  %res.0.i185 = phi i64 [ %113, %bb3.i182 ], [ 0, %__str_to_int.exit198 ] ; <i64> [#uses=3]
  %s_addr.0.i186 = getelementptr i8* %103, i64 %indvar.i184 ; <i8*> [#uses=1]
  %114 = load i8* %s_addr.0.i186, align 1         ; <i8> [#uses=3]
  %115 = icmp eq i8 %114, 0                       ; <i1> [#uses=1]
  br i1 %115, label %__str_to_int.exit188, label %bb2.i180

__str_to_int.exit188:                             ; preds = %bb5.i187
  %116 = trunc i64 %res.0.i185 to i32             ; <i32> [#uses=3]
  %117 = add i32 %100, 1                          ; <i32> [#uses=1]
  %118 = call i32 @klee_range(i32 %85, i32 %117, i8* getelementptr inbounds ([7 x i8]* @.str12, i64 0, i64 0)) nounwind ; <i32> [#uses=2]
  %119 = add nsw i32 %116, 1                      ; <i32> [#uses=4]
  %120 = icmp sgt i32 %116, 0                     ; <i1> [#uses=1]
  %121 = sext i32 %116 to i64                     ; <i64> [#uses=2]
  br i1 %120, label %__str_to_int.exit188.split.us, label %__str_to_int.exit188.split

__str_to_int.exit188.split.us:                    ; preds = %__str_to_int.exit188
  %tmp422 = and i64 %res.0.i185, 4294967295       ; <i64> [#uses=1]
  %tmp432 = sext i32 %new_argc.0 to i64           ; <i64> [#uses=1]
  %tmp434 = trunc i32 %sym_arg_num.0 to i8        ; <i8> [#uses=1]
  %tmp435 = add i8 %tmp434, 48                    ; <i8> [#uses=1]
  br label %bb20.us

bb20.us:                                          ; preds = %__add_arg.exit174.us, %__str_to_int.exit188.split.us
  %indvar = phi i64 [ 0, %__str_to_int.exit188.split.us ], [ %indvar.next, %__add_arg.exit174.us ] ; <i64> [#uses=5]
  %indvar642 = trunc i64 %indvar to i32           ; <i32> [#uses=2]
  %new_argc.1.us = add i32 %indvar642, %new_argc.0 ; <i32> [#uses=2]
  %tmp433 = add i64 %indvar, %tmp432              ; <i64> [#uses=1]
  %scevgep = getelementptr [1024 x i8*]* %new_argv, i64 0, i64 %tmp433 ; <i8**> [#uses=1]
  %122 = icmp slt i32 %indvar642, %118            ; <i1> [#uses=1]
  br i1 %122, label %bb17.us, label %bb42.loopexit280.split

__add_arg.exit174.us:                             ; preds = %__get_sym_str.exit.loopexit.us
  store i8* %123, i8** %scevgep, align 8
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %bb20.us

bb17.us:                                          ; preds = %bb20.us
  %indvar638 = trunc i64 %indvar to i8            ; <i8> [#uses=1]
  %tmp438 = add i8 %indvar638, %tmp435            ; <i8> [#uses=1]
  store i8 %tmp438, i8* %5, align 1
  %123 = malloc i8, i32 %119                      ; <i8*> [#uses=6]
  call void @klee_mark_global(i8* %123) nounwind
  call void @klee_make_symbolic(i8* %123, i32 %119, i8* %2) nounwind
  br label %bb.i177.us

bb.i177.us:                                       ; preds = %bb.i177.us, %bb17.us
  %indvar.i175.us = phi i64 [ 0, %bb17.us ], [ %indvar.next.i176.us, %bb.i177.us ] ; <i64> [#uses=2]
  %scevgep.i.us = getelementptr i8* %123, i64 %indvar.i175.us ; <i8*> [#uses=1]
  %124 = load i8* %scevgep.i.us, align 1          ; <i8> [#uses=1]
  %125 = add i8 %124, -32                         ; <i8> [#uses=1]
  %126 = icmp ult i8 %125, 95                     ; <i1> [#uses=1]
  %127 = zext i1 %126 to i32                      ; <i32> [#uses=1]
  call void @klee_prefer_cex(i8* %123, i32 %127) nounwind
  %indvar.next.i176.us = add i64 %indvar.i175.us, 1 ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next.i176.us, %tmp422 ; <i1> [#uses=1]
  br i1 %exitcond, label %__get_sym_str.exit.loopexit.us, label %bb.i177.us

__get_sym_str.exit.loopexit.us:                   ; preds = %bb.i177.us
  %128 = getelementptr inbounds i8* %123, i64 %121 ; <i8*> [#uses=1]
  store i8 0, i8* %128, align 1
  %129 = icmp eq i32 %new_argc.1.us, 1024         ; <i1> [#uses=1]
  br i1 %129, label %bb.i173.split, label %__add_arg.exit174.us

__str_to_int.exit188.split:                       ; preds = %__str_to_int.exit188
  %tmp456 = sext i32 %new_argc.0 to i64           ; <i64> [#uses=1]
  %tmp459 = trunc i32 %sym_arg_num.0 to i8        ; <i8> [#uses=1]
  %tmp460 = add i8 %tmp459, 48                    ; <i8> [#uses=1]
  br label %bb20

bb17:                                             ; preds = %bb20
  %indvar447644 = trunc i64 %indvar447 to i8      ; <i8> [#uses=1]
  %tmp463 = add i8 %indvar447644, %tmp460         ; <i8> [#uses=1]
  store i8 %tmp463, i8* %5, align 1
  %130 = malloc i8, i32 %119                      ; <i8*> [#uses=4]
  call void @klee_mark_global(i8* %130) nounwind
  call void @klee_make_symbolic(i8* %130, i32 %119, i8* %2) nounwind
  %131 = getelementptr inbounds i8* %130, i64 %121 ; <i8*> [#uses=1]
  store i8 0, i8* %131, align 1
  %132 = icmp eq i32 %new_argc.1, 1024            ; <i1> [#uses=1]
  br i1 %132, label %bb.i173.split, label %__add_arg.exit174

bb.i173.split:                                    ; preds = %bb17, %__get_sym_str.exit.loopexit.us
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([37 x i8]* @.str2, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

__add_arg.exit174:                                ; preds = %bb17
  store i8* %130, i8** %scevgep458, align 8
  %indvar.next448 = add i64 %indvar447, 1         ; <i64> [#uses=1]
  br label %bb20

bb20:                                             ; preds = %__add_arg.exit174, %__str_to_int.exit188.split
  %indvar447 = phi i64 [ 0, %__str_to_int.exit188.split ], [ %indvar.next448, %__add_arg.exit174 ] ; <i64> [#uses=5]
  %indvar447648 = trunc i64 %indvar447 to i32     ; <i32> [#uses=2]
  %new_argc.1 = add i32 %indvar447648, %new_argc.0 ; <i32> [#uses=2]
  %tmp457 = add i64 %indvar447, %tmp456           ; <i64> [#uses=1]
  %scevgep458 = getelementptr [1024 x i8*]* %new_argv, i64 0, i64 %tmp457 ; <i8**> [#uses=1]
  %133 = icmp slt i32 %indvar447648, %118         ; <i1> [#uses=1]
  br i1 %133, label %bb17, label %bb42.loopexit280.split

bb.i163:                                          ; preds = %bb3.i169
  %134 = icmp eq i8 %135, 0                       ; <i1> [#uses=1]
  br i1 %134, label %bb23, label %bb2.i165

bb2.i165:                                         ; preds = %bb.i163
  %indvar.next.i164 = add i64 %indvar.i166, 1     ; <i64> [#uses=1]
  br label %bb3.i169

bb3.i169:                                         ; preds = %bb2.i165, %bb3.i216
  %indvar.i166 = phi i64 [ %indvar.next.i164, %bb2.i165 ], [ 0, %bb3.i216 ] ; <i64> [#uses=3]
  %b_addr.0.i167 = getelementptr [12 x i8]* @.str13, i64 0, i64 %indvar.i166 ; <i8*> [#uses=1]
  %a_addr.0.i168 = getelementptr i8* %16, i64 %indvar.i166 ; <i8*> [#uses=1]
  %135 = load i8* %a_addr.0.i168, align 1         ; <i8> [#uses=2]
  %136 = load i8* %b_addr.0.i167, align 1         ; <i8> [#uses=1]
  %137 = icmp eq i8 %135, %136                    ; <i1> [#uses=1]
  br i1 %137, label %bb.i163, label %bb3.i159

bb.i153:                                          ; preds = %bb3.i159
  %138 = icmp eq i8 %139, 0                       ; <i1> [#uses=1]
  br i1 %138, label %bb23, label %bb2.i155

bb2.i155:                                         ; preds = %bb.i153
  %indvar.next.i154 = add i64 %indvar.i156, 1     ; <i64> [#uses=1]
  br label %bb3.i159

bb3.i159:                                         ; preds = %bb2.i155, %bb3.i169
  %indvar.i156 = phi i64 [ %indvar.next.i154, %bb2.i155 ], [ 0, %bb3.i169 ] ; <i64> [#uses=3]
  %b_addr.0.i157 = getelementptr [11 x i8]* @.str14, i64 0, i64 %indvar.i156 ; <i8*> [#uses=1]
  %a_addr.0.i158 = getelementptr i8* %16, i64 %indvar.i156 ; <i8*> [#uses=1]
  %139 = load i8* %a_addr.0.i158, align 1         ; <i8> [#uses=2]
  %140 = load i8* %b_addr.0.i157, align 1         ; <i8> [#uses=1]
  %141 = icmp eq i8 %139, %140                    ; <i1> [#uses=1]
  br i1 %141, label %bb.i153, label %bb3.i128

bb23:                                             ; preds = %bb.i153, %bb.i163
  %142 = add i32 %k.0, 2                          ; <i32> [#uses=2]
  %143 = icmp slt i32 %142, %0                    ; <i1> [#uses=1]
  br i1 %143, label %bb25, label %bb24

bb24:                                             ; preds = %bb23
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([72 x i8]* @.str15, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb25:                                             ; preds = %bb23
  %144 = add i32 %k.0, 1                          ; <i32> [#uses=1]
  %145 = sext i32 %144 to i64                     ; <i64> [#uses=1]
  %146 = getelementptr inbounds i8** %1, i64 %145 ; <i8**> [#uses=1]
  %147 = load i8** %146, align 8                  ; <i8*> [#uses=2]
  %148 = load i8* %147, align 1                   ; <i8> [#uses=1]
  %149 = icmp eq i8 %148, 0                       ; <i1> [#uses=1]
  br i1 %149, label %bb.i142, label %bb5.i150

bb.i142:                                          ; preds = %bb25
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([72 x i8]* @.str15, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i143:                                         ; preds = %bb5.i150
  %150 = add i8 %157, -48                         ; <i8> [#uses=1]
  %151 = icmp ugt i8 %150, 9                      ; <i1> [#uses=1]
  br i1 %151, label %bb4.i146, label %bb3.i145

bb3.i145:                                         ; preds = %bb2.i143
  %152 = mul i64 %res.0.i148, 10                  ; <i64> [#uses=1]
  %153 = sext i8 %157 to i32                      ; <i32> [#uses=1]
  %154 = add i32 %153, -48                        ; <i32> [#uses=1]
  %155 = sext i32 %154 to i64                     ; <i64> [#uses=1]
  %156 = add nsw i64 %155, %152                   ; <i64> [#uses=1]
  %indvar.next.i144 = add i64 %indvar.i147, 1     ; <i64> [#uses=1]
  br label %bb5.i150

bb4.i146:                                         ; preds = %bb2.i143
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([72 x i8]* @.str15, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i150:                                         ; preds = %bb3.i145, %bb25
  %indvar.i147 = phi i64 [ %indvar.next.i144, %bb3.i145 ], [ 0, %bb25 ] ; <i64> [#uses=2]
  %res.0.i148 = phi i64 [ %156, %bb3.i145 ], [ 0, %bb25 ] ; <i64> [#uses=2]
  %s_addr.0.i149 = getelementptr i8* %147, i64 %indvar.i147 ; <i8*> [#uses=1]
  %157 = load i8* %s_addr.0.i149, align 1         ; <i8> [#uses=3]
  %158 = icmp eq i8 %157, 0                       ; <i1> [#uses=1]
  br i1 %158, label %__str_to_int.exit151, label %bb2.i143

__str_to_int.exit151:                             ; preds = %bb5.i150
  %159 = trunc i64 %res.0.i148 to i32             ; <i32> [#uses=1]
  %160 = sext i32 %142 to i64                     ; <i64> [#uses=1]
  %161 = getelementptr inbounds i8** %1, i64 %160 ; <i8**> [#uses=1]
  %162 = load i8** %161, align 8                  ; <i8*> [#uses=2]
  %163 = add i32 %k.0, 3                          ; <i32> [#uses=1]
  %164 = load i8* %162, align 1                   ; <i8> [#uses=1]
  %165 = icmp eq i8 %164, 0                       ; <i1> [#uses=1]
  br i1 %165, label %bb.i132, label %bb5.i140

bb.i132:                                          ; preds = %__str_to_int.exit151
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([72 x i8]* @.str15, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i133:                                         ; preds = %bb5.i140
  %166 = add i8 %173, -48                         ; <i8> [#uses=1]
  %167 = icmp ugt i8 %166, 9                      ; <i1> [#uses=1]
  br i1 %167, label %bb4.i136, label %bb3.i135

bb3.i135:                                         ; preds = %bb2.i133
  %168 = mul i64 %res.0.i138, 10                  ; <i64> [#uses=1]
  %169 = sext i8 %173 to i32                      ; <i32> [#uses=1]
  %170 = add i32 %169, -48                        ; <i32> [#uses=1]
  %171 = sext i32 %170 to i64                     ; <i64> [#uses=1]
  %172 = add nsw i64 %171, %168                   ; <i64> [#uses=1]
  %indvar.next.i134 = add i64 %indvar.i137, 1     ; <i64> [#uses=1]
  br label %bb5.i140

bb4.i136:                                         ; preds = %bb2.i133
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([72 x i8]* @.str15, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i140:                                         ; preds = %bb3.i135, %__str_to_int.exit151
  %indvar.i137 = phi i64 [ %indvar.next.i134, %bb3.i135 ], [ 0, %__str_to_int.exit151 ] ; <i64> [#uses=2]
  %res.0.i138 = phi i64 [ %172, %bb3.i135 ], [ 0, %__str_to_int.exit151 ] ; <i64> [#uses=2]
  %s_addr.0.i139 = getelementptr i8* %162, i64 %indvar.i137 ; <i8*> [#uses=1]
  %173 = load i8* %s_addr.0.i139, align 1         ; <i8> [#uses=3]
  %174 = icmp eq i8 %173, 0                       ; <i1> [#uses=1]
  br i1 %174, label %__str_to_int.exit141, label %bb2.i133

__str_to_int.exit141:                             ; preds = %bb5.i140
  %175 = trunc i64 %res.0.i138 to i32             ; <i32> [#uses=1]
  br label %bb42

bb.i122:                                          ; preds = %bb3.i128
  %176 = icmp eq i8 %177, 0                       ; <i1> [#uses=1]
  br i1 %176, label %bb28, label %bb2.i124

bb2.i124:                                         ; preds = %bb.i122
  %indvar.next.i123 = add i64 %indvar.i125, 1     ; <i64> [#uses=1]
  br label %bb3.i128

bb3.i128:                                         ; preds = %bb2.i124, %bb3.i159
  %indvar.i125 = phi i64 [ %indvar.next.i123, %bb2.i124 ], [ 0, %bb3.i159 ] ; <i64> [#uses=3]
  %b_addr.0.i126 = getelementptr [13 x i8]* @.str16, i64 0, i64 %indvar.i125 ; <i8*> [#uses=1]
  %a_addr.0.i127 = getelementptr i8* %16, i64 %indvar.i125 ; <i8*> [#uses=1]
  %177 = load i8* %a_addr.0.i127, align 1         ; <i8> [#uses=2]
  %178 = load i8* %b_addr.0.i126, align 1         ; <i8> [#uses=1]
  %179 = icmp eq i8 %177, %178                    ; <i1> [#uses=1]
  br i1 %179, label %bb.i122, label %bb3.i118

bb.i112:                                          ; preds = %bb3.i118
  %180 = icmp eq i8 %181, 0                       ; <i1> [#uses=1]
  br i1 %180, label %bb28, label %bb2.i114

bb2.i114:                                         ; preds = %bb.i112
  %indvar.next.i113 = add i64 %indvar.i115, 1     ; <i64> [#uses=1]
  br label %bb3.i118

bb3.i118:                                         ; preds = %bb2.i114, %bb3.i128
  %indvar.i115 = phi i64 [ %indvar.next.i113, %bb2.i114 ], [ 0, %bb3.i128 ] ; <i64> [#uses=3]
  %b_addr.0.i116 = getelementptr [12 x i8]* @.str17, i64 0, i64 %indvar.i115 ; <i8*> [#uses=1]
  %a_addr.0.i117 = getelementptr i8* %16, i64 %indvar.i115 ; <i8*> [#uses=1]
  %181 = load i8* %a_addr.0.i117, align 1         ; <i8> [#uses=2]
  %182 = load i8* %b_addr.0.i116, align 1         ; <i8> [#uses=1]
  %183 = icmp eq i8 %181, %182                    ; <i1> [#uses=1]
  br i1 %183, label %bb.i112, label %bb3.i108

bb28:                                             ; preds = %bb.i112, %bb.i122
  %184 = add i32 %k.0, 1                          ; <i32> [#uses=1]
  br label %bb42

bb.i102:                                          ; preds = %bb3.i108
  %185 = icmp eq i8 %186, 0                       ; <i1> [#uses=1]
  br i1 %185, label %bb31, label %bb2.i104

bb2.i104:                                         ; preds = %bb.i102
  %indvar.next.i103 = add i64 %indvar.i105, 1     ; <i64> [#uses=1]
  br label %bb3.i108

bb3.i108:                                         ; preds = %bb2.i104, %bb3.i118
  %indvar.i105 = phi i64 [ %indvar.next.i103, %bb2.i104 ], [ 0, %bb3.i118 ] ; <i64> [#uses=3]
  %b_addr.0.i106 = getelementptr [18 x i8]* @.str18, i64 0, i64 %indvar.i105 ; <i8*> [#uses=1]
  %a_addr.0.i107 = getelementptr i8* %16, i64 %indvar.i105 ; <i8*> [#uses=1]
  %186 = load i8* %a_addr.0.i107, align 1         ; <i8> [#uses=2]
  %187 = load i8* %b_addr.0.i106, align 1         ; <i8> [#uses=1]
  %188 = icmp eq i8 %186, %187                    ; <i1> [#uses=1]
  br i1 %188, label %bb.i102, label %bb3.i98

bb.i92:                                           ; preds = %bb3.i98
  %189 = icmp eq i8 %190, 0                       ; <i1> [#uses=1]
  br i1 %189, label %bb31, label %bb2.i94

bb2.i94:                                          ; preds = %bb.i92
  %indvar.next.i93 = add i64 %indvar.i95, 1       ; <i64> [#uses=1]
  br label %bb3.i98

bb3.i98:                                          ; preds = %bb2.i94, %bb3.i108
  %indvar.i95 = phi i64 [ %indvar.next.i93, %bb2.i94 ], [ 0, %bb3.i108 ] ; <i64> [#uses=3]
  %b_addr.0.i96 = getelementptr [17 x i8]* @.str19, i64 0, i64 %indvar.i95 ; <i8*> [#uses=1]
  %a_addr.0.i97 = getelementptr i8* %16, i64 %indvar.i95 ; <i8*> [#uses=1]
  %190 = load i8* %a_addr.0.i97, align 1          ; <i8> [#uses=2]
  %191 = load i8* %b_addr.0.i96, align 1          ; <i8> [#uses=1]
  %192 = icmp eq i8 %190, %191                    ; <i1> [#uses=1]
  br i1 %192, label %bb.i92, label %bb3.i88

bb31:                                             ; preds = %bb.i92, %bb.i102
  %193 = add i32 %k.0, 1                          ; <i32> [#uses=1]
  br label %bb42

bb.i82:                                           ; preds = %bb3.i88
  %194 = icmp eq i8 %195, 0                       ; <i1> [#uses=1]
  br i1 %194, label %bb34, label %bb2.i84

bb2.i84:                                          ; preds = %bb.i82
  %indvar.next.i83 = add i64 %indvar.i85, 1       ; <i64> [#uses=1]
  br label %bb3.i88

bb3.i88:                                          ; preds = %bb2.i84, %bb3.i98
  %indvar.i85 = phi i64 [ %indvar.next.i83, %bb2.i84 ], [ 0, %bb3.i98 ] ; <i64> [#uses=3]
  %b_addr.0.i86 = getelementptr [10 x i8]* @.str20, i64 0, i64 %indvar.i85 ; <i8*> [#uses=1]
  %a_addr.0.i87 = getelementptr i8* %16, i64 %indvar.i85 ; <i8*> [#uses=1]
  %195 = load i8* %a_addr.0.i87, align 1          ; <i8> [#uses=2]
  %196 = load i8* %b_addr.0.i86, align 1          ; <i8> [#uses=1]
  %197 = icmp eq i8 %195, %196                    ; <i1> [#uses=1]
  br i1 %197, label %bb.i82, label %bb3.i78

bb.i72:                                           ; preds = %bb3.i78
  %198 = icmp eq i8 %199, 0                       ; <i1> [#uses=1]
  br i1 %198, label %bb34, label %bb2.i74

bb2.i74:                                          ; preds = %bb.i72
  %indvar.next.i73 = add i64 %indvar.i75, 1       ; <i64> [#uses=1]
  br label %bb3.i78

bb3.i78:                                          ; preds = %bb2.i74, %bb3.i88
  %indvar.i75 = phi i64 [ %indvar.next.i73, %bb2.i74 ], [ 0, %bb3.i88 ] ; <i64> [#uses=3]
  %b_addr.0.i76 = getelementptr [9 x i8]* @.str21, i64 0, i64 %indvar.i75 ; <i8*> [#uses=1]
  %a_addr.0.i77 = getelementptr i8* %16, i64 %indvar.i75 ; <i8*> [#uses=1]
  %199 = load i8* %a_addr.0.i77, align 1          ; <i8> [#uses=2]
  %200 = load i8* %b_addr.0.i76, align 1          ; <i8> [#uses=1]
  %201 = icmp eq i8 %199, %200                    ; <i1> [#uses=1]
  br i1 %201, label %bb.i72, label %bb3.i68

bb34:                                             ; preds = %bb.i72, %bb.i82
  %202 = add i32 %k.0, 1                          ; <i32> [#uses=1]
  br label %bb42

bb.i62:                                           ; preds = %bb3.i68
  %203 = icmp eq i8 %204, 0                       ; <i1> [#uses=1]
  br i1 %203, label %bb37, label %bb2.i64

bb2.i64:                                          ; preds = %bb.i62
  %indvar.next.i63 = add i64 %indvar.i65, 1       ; <i64> [#uses=1]
  br label %bb3.i68

bb3.i68:                                          ; preds = %bb2.i64, %bb3.i78
  %indvar.i65 = phi i64 [ %indvar.next.i63, %bb2.i64 ], [ 0, %bb3.i78 ] ; <i64> [#uses=3]
  %b_addr.0.i66 = getelementptr [11 x i8]* @.str22, i64 0, i64 %indvar.i65 ; <i8*> [#uses=1]
  %a_addr.0.i67 = getelementptr i8* %16, i64 %indvar.i65 ; <i8*> [#uses=1]
  %204 = load i8* %a_addr.0.i67, align 1          ; <i8> [#uses=2]
  %205 = load i8* %b_addr.0.i66, align 1          ; <i8> [#uses=1]
  %206 = icmp eq i8 %204, %205                    ; <i1> [#uses=1]
  br i1 %206, label %bb.i62, label %bb3.i58

bb.i52:                                           ; preds = %bb3.i58
  %207 = icmp eq i8 %208, 0                       ; <i1> [#uses=1]
  br i1 %207, label %bb37, label %bb2.i54

bb2.i54:                                          ; preds = %bb.i52
  %indvar.next.i53 = add i64 %indvar.i55, 1       ; <i64> [#uses=1]
  br label %bb3.i58

bb3.i58:                                          ; preds = %bb2.i54, %bb3.i68
  %indvar.i55 = phi i64 [ %indvar.next.i53, %bb2.i54 ], [ 0, %bb3.i68 ] ; <i64> [#uses=3]
  %b_addr.0.i56 = getelementptr [10 x i8]* @.str23, i64 0, i64 %indvar.i55 ; <i8*> [#uses=1]
  %a_addr.0.i57 = getelementptr i8* %16, i64 %indvar.i55 ; <i8*> [#uses=1]
  %208 = load i8* %a_addr.0.i57, align 1          ; <i8> [#uses=2]
  %209 = load i8* %b_addr.0.i56, align 1          ; <i8> [#uses=1]
  %210 = icmp eq i8 %208, %209                    ; <i1> [#uses=1]
  br i1 %210, label %bb.i52, label %bb40

bb37:                                             ; preds = %bb.i52, %bb.i62
  %211 = add i32 %k.0, 1                          ; <i32> [#uses=2]
  %212 = icmp eq i32 %211, %0                     ; <i1> [#uses=1]
  br i1 %212, label %bb38, label %bb39

bb38:                                             ; preds = %bb37
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([54 x i8]* @.str24, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb39:                                             ; preds = %bb37
  %213 = sext i32 %211 to i64                     ; <i64> [#uses=1]
  %214 = getelementptr inbounds i8** %1, i64 %213 ; <i8**> [#uses=1]
  %215 = load i8** %214, align 8                  ; <i8*> [#uses=2]
  %216 = add i32 %k.0, 2                          ; <i32> [#uses=1]
  %217 = load i8* %215, align 1                   ; <i8> [#uses=1]
  %218 = icmp eq i8 %217, 0                       ; <i1> [#uses=1]
  br i1 %218, label %bb.i47, label %bb5.i

bb.i47:                                           ; preds = %bb39
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([54 x i8]* @.str24, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2.i48:                                          ; preds = %bb5.i
  %219 = add i8 %226, -48                         ; <i8> [#uses=1]
  %220 = icmp ugt i8 %219, 9                      ; <i1> [#uses=1]
  br i1 %220, label %bb4.i, label %bb3.i50

bb3.i50:                                          ; preds = %bb2.i48
  %221 = mul i64 %res.0.i, 10                     ; <i64> [#uses=1]
  %222 = sext i8 %226 to i32                      ; <i32> [#uses=1]
  %223 = add i32 %222, -48                        ; <i32> [#uses=1]
  %224 = sext i32 %223 to i64                     ; <i64> [#uses=1]
  %225 = add nsw i64 %224, %221                   ; <i64> [#uses=1]
  %indvar.next.i49 = add i64 %indvar.i51, 1       ; <i64> [#uses=1]
  br label %bb5.i

bb4.i:                                            ; preds = %bb2.i48
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([54 x i8]* @.str24, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

bb5.i:                                            ; preds = %bb3.i50, %bb39
  %indvar.i51 = phi i64 [ %indvar.next.i49, %bb3.i50 ], [ 0, %bb39 ] ; <i64> [#uses=2]
  %res.0.i = phi i64 [ %225, %bb3.i50 ], [ 0, %bb39 ] ; <i64> [#uses=2]
  %s_addr.0.i = getelementptr i8* %215, i64 %indvar.i51 ; <i8*> [#uses=1]
  %226 = load i8* %s_addr.0.i, align 1            ; <i8> [#uses=3]
  %227 = icmp eq i8 %226, 0                       ; <i1> [#uses=1]
  br i1 %227, label %__str_to_int.exit, label %bb2.i48

__str_to_int.exit:                                ; preds = %bb5.i
  %228 = trunc i64 %res.0.i to i32                ; <i32> [#uses=1]
  br label %bb42

bb40:                                             ; preds = %bb3.i58
  %229 = add i32 %k.0, 1                          ; <i32> [#uses=1]
  %230 = icmp eq i32 %new_argc.0, 1024            ; <i1> [#uses=1]
  br i1 %230, label %bb.i46, label %__add_arg.exit

bb.i46:                                           ; preds = %bb40
  call void @klee_report_error(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 22, i8* getelementptr inbounds ([37 x i8]* @.str2, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8]* @.str1, i64 0, i64 0)) noreturn nounwind
  unreachable

__add_arg.exit:                                   ; preds = %bb40
  %231 = sext i32 %new_argc.0 to i64              ; <i64> [#uses=1]
  %232 = getelementptr inbounds [1024 x i8*]* %new_argv, i64 0, i64 %231 ; <i8**> [#uses=1]
  store i8* %16, i8** %232, align 8
  %233 = add i32 %new_argc.0, 1                   ; <i32> [#uses=1]
  br label %bb42

bb42.loopexit280.split:                           ; preds = %bb20, %bb20.us
  %indvar447646.pn.in = phi i64 [ %indvar, %bb20.us ], [ %indvar447, %bb20 ] ; <i64> [#uses=1]
  %new_argc.1.lcssa.us-lcssa = phi i32 [ %new_argc.1.us, %bb20.us ], [ %new_argc.1, %bb20 ] ; <i32> [#uses=1]
  %indvar447646.pn = trunc i64 %indvar447646.pn.in to i32 ; <i32> [#uses=1]
  %sym_arg_num.1.lcssa.us-lcssa = add i32 %indvar447646.pn, %sym_arg_num.0 ; <i32> [#uses=1]
  br label %bb42

bb42:                                             ; preds = %bb42.loopexit280.split, %__add_arg.exit, %__str_to_int.exit, %bb34, %bb31, %bb28, %__str_to_int.exit141, %__add_arg.exit231, %bb3.i, %entry
  %new_argc.0 = phi i32 [ %58, %__add_arg.exit231 ], [ %new_argc.0, %__str_to_int.exit141 ], [ %new_argc.0, %bb28 ], [ %new_argc.0, %bb31 ], [ %new_argc.0, %bb34 ], [ %new_argc.0, %__str_to_int.exit ], [ %233, %__add_arg.exit ], [ %new_argc.1.lcssa.us-lcssa, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=18]
  %sym_arg_num.0 = phi i32 [ %45, %__add_arg.exit231 ], [ %sym_arg_num.0, %__str_to_int.exit141 ], [ %sym_arg_num.0, %bb28 ], [ %sym_arg_num.0, %bb31 ], [ %sym_arg_num.0, %bb34 ], [ %sym_arg_num.0, %__str_to_int.exit ], [ %sym_arg_num.0, %__add_arg.exit ], [ %sym_arg_num.1.lcssa.us-lcssa, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=11]
  %k.0 = phi i32 [ %30, %__add_arg.exit231 ], [ %163, %__str_to_int.exit141 ], [ %184, %bb28 ], [ %193, %bb31 ], [ %202, %bb34 ], [ %216, %__str_to_int.exit ], [ %229, %__add_arg.exit ], [ %104, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=17]
  %fd_fail.0 = phi i32 [ %fd_fail.0, %__add_arg.exit231 ], [ %fd_fail.0, %__str_to_int.exit141 ], [ %fd_fail.0, %bb28 ], [ %fd_fail.0, %bb31 ], [ 1, %bb34 ], [ %228, %__str_to_int.exit ], [ %fd_fail.0, %__add_arg.exit ], [ %fd_fail.0, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=7]
  %save_all_writes_flag.0 = phi i32 [ %save_all_writes_flag.0, %__add_arg.exit231 ], [ %save_all_writes_flag.0, %__str_to_int.exit141 ], [ %save_all_writes_flag.0, %bb28 ], [ 1, %bb31 ], [ %save_all_writes_flag.0, %bb34 ], [ %save_all_writes_flag.0, %__str_to_int.exit ], [ %save_all_writes_flag.0, %__add_arg.exit ], [ %save_all_writes_flag.0, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=8]
  %sym_stdout_flag.0 = phi i32 [ %sym_stdout_flag.0, %__add_arg.exit231 ], [ %sym_stdout_flag.0, %__str_to_int.exit141 ], [ 1, %bb28 ], [ %sym_stdout_flag.0, %bb31 ], [ %sym_stdout_flag.0, %bb34 ], [ %sym_stdout_flag.0, %__str_to_int.exit ], [ %sym_stdout_flag.0, %__add_arg.exit ], [ %sym_stdout_flag.0, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=8]
  %sym_file_len.0 = phi i32 [ %sym_file_len.0, %__add_arg.exit231 ], [ %175, %__str_to_int.exit141 ], [ %sym_file_len.0, %bb28 ], [ %sym_file_len.0, %bb31 ], [ %sym_file_len.0, %bb34 ], [ %sym_file_len.0, %__str_to_int.exit ], [ %sym_file_len.0, %__add_arg.exit ], [ %sym_file_len.0, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=8]
  %sym_files.0 = phi i32 [ %sym_files.0, %__add_arg.exit231 ], [ %159, %__str_to_int.exit141 ], [ %sym_files.0, %bb28 ], [ %sym_files.0, %bb31 ], [ %sym_files.0, %bb34 ], [ %sym_files.0, %__str_to_int.exit ], [ %sym_files.0, %__add_arg.exit ], [ %sym_files.0, %bb42.loopexit280.split ], [ 0, %bb3.i ], [ 0, %entry ] ; <i32> [#uses=8]
  %234 = icmp slt i32 %k.0, %0                    ; <i1> [#uses=1]
  br i1 %234, label %bb5, label %bb43

bb43:                                             ; preds = %bb42
  %235 = add nsw i32 %new_argc.0, 1               ; <i32> [#uses=1]
  %236 = malloc i8*, i32 %235                     ; <i8**> [#uses=3]
  %tmpcast = bitcast i8** %236 to i8*             ; <i8*> [#uses=2]
  call void @klee_mark_global(i8* %tmpcast) nounwind
  %237 = sext i32 %new_argc.0 to i64              ; <i64> [#uses=2]
  %238 = shl i64 %237, 3                          ; <i64> [#uses=1]
  %new_argv4445 = bitcast [1024 x i8*]* %new_argv to i8* ; <i8*> [#uses=1]
  call void @llvm.memcpy.i64(i8* %tmpcast, i8* %new_argv4445, i64 %238, i32 8)
  %239 = getelementptr inbounds i8** %236, i64 %237 ; <i8**> [#uses=1]
  store i8* null, i8** %239, align 8
  store i32 %new_argc.0, i32* %argcPtr, align 4
  store i8** %236, i8*** %argvPtr, align 8
  call void @klee_init_fds(i32 %sym_files.0, i32 %sym_file_len.0, i32 %sym_stdout_flag.0, i32 %save_all_writes_flag.0, i32 %fd_fail.0) nounwind
  ret void
}

declare void @klee_mark_global(i8*)

declare void @klee_make_symbolic(i8*, i32, i8*)

declare void @klee_prefer_cex(i8*, i32)

declare void @klee_report_error(i8*, i32, i8*, i8*) noreturn

declare i32 @klee_range(i32, i32, i8*)

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

declare void @klee_init_fds(i32, i32, i32, i32, i32)
