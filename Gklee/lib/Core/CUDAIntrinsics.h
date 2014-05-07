//===-- ExecutorCUDAIntrinsic.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h> 

static std::string CUDAArithmetic[] = {
  "mulhi",
  "mul64hi",
  "mul24",
  "sad",
  "fdivide",
  "sin",
  "cos",
  "tan",
  "exp",
  "log",
  "pow",
  "__fadd_",
  "__dadd_",
  "__fmul_",
  "__dmul_",
  "fma",
  "rcp",
  "sqrt",
  "__fdiv_",
  "__ddiv_",
  "clz", 
  "ffs", 
  "popc", 
  "brev",
  "byte_perm"
  "hadd",
  "abs",
  "min",
  "max",
  "saturate",
  "round",
  "trunc",
  "floor",
  "ceil"
};

static std::string CUDAConversion[] = {
  "__float2int_",
  "__float2uint_",
  "__int2float_", 
  "__uint2float_",
  "__float2ll_", 
  "__float2ull_", 
  "__ll2float_", 
  "__ull2float_", 
  "__float2half_", 
  "__half2float",   
  "__int2double_",
  "__uint2double_",
  "__ll2double_",
  "__ull2double_"
  "__double2int_",
  "__double2uint_",
  "__double2ll_",
  "__double2ull_",
  "__double2hiint",
  "__double2loint",
  "__hiloint2double",
  "__float_as_int",
  "__int_as_float",
  "__double_as_longlong",
  "__longlong_as_double"
};

static std::string CUDAAtomic[] = {
  "AtomicAdd",
  "AtomicExch",
  "AtomicMin",
  "AtomicMax",
  "AtomicInc",
  "AtomicDec",
  "AtomicCas",
  "AtomicAnd",
  "AtomicOr",
  "AtomicXor"
};

static std::string CUDASync[] = {
  "__syncthreads",
  "__syncthreads_count",
  "__syncthreads_and",
  "__syncthreads_or"
};

static std::string CUDAMemfence[] = {
  "__threadfence",
  "__threadfence_block",
  "__threadfence_system"
};
