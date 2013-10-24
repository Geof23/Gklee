//===-- MemoryManager.cpp -------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "CoreStats.h"
#include "Memory.h"
#include "MemoryManager.h"

#include "klee/ExecutionState.h"
#include "klee/Expr.h"
#include "klee/Solver.h"

#include "llvm/Support/CommandLine.h"

#include "CUDA.h"

using namespace klee;

/***/

MemoryManager::~MemoryManager() {
  while (!objects.empty()) {
    MemoryObject *mo = *objects.begin();
    if (!mo->isFixed)
      free((void *)mo->address);
    objects.erase(mo);
    delete mo;
  }
}

MemoryObject *MemoryManager::allocate(uint64_t size, bool isLocal, 
                                      bool isGlobal, int deviceSet,
                                      bool is_GPU_mode, 
                                      const llvm::Value *allocSite) {
  if (size>10*1024*1024)
    klee_warning_once(0, "Large alloc: %u bytes.  KLEE may run out of memory.", (unsigned) size);

  uint64_t address = (uint64_t) (unsigned long) malloc((unsigned) size);
  if (!address)
    return 0;
  
  ++stats::allocations;
  // memory type inference 
  GPUConfig::CTYPE ctype;
  if (deviceSet == 2)
    ctype = GPUConfig::HOST;
  else if (deviceSet == 1)
    ctype = GPUConfig::DEVICE;
  else  
    ctype = CUDAUtil::getCType(allocSite, is_GPU_mode);

  MemoryObject *res = new MemoryObject(address, size, isLocal, isGlobal, false,
                                       allocSite, this, ctype);
  objects.insert(res);
  return res;
}

MemoryObject *MemoryManager::allocateSymGridDim() {
  uint64_t size = 12;
  uint64_t addr = (uint64_t) (unsigned long) malloc((unsigned)size);

  if (!addr) {
    assert(0 && "fail to allocate memory");
    return 0;
  }
  ++stats::allocations;
  MemoryObject *res = new MemoryObject(addr, size, false, false, 
                                       false, 0, this, GPUConfig::HOST); 
  res->setName("symgdim");
  res->is_builtin = true;          
  objects.insert(res);
  return res;
}

MemoryObject *MemoryManager::allocateMissedBuiltInMO(GPUConfig::CTYPE ctype, 
                                                    std::string vname) {
  uint64_t size = 12; 
  uint64_t addr = (uint64_t) (unsigned long) malloc((unsigned)size);

  if (!addr) {
    assert(0 && "fail to allocate memory");
    return 0;
  }

  ++stats::allocations;
  MemoryObject *res = new MemoryObject(addr, size, false, false, 
                                       false, 0, this, ctype); 
  res->setName(vname);
  res->is_builtin = true;
  objects.insert(res);
  return res;
}

MemoryObject *MemoryManager::allocateSymBlockDim() {
  uint64_t size = 12;
  uint64_t addr = (uint64_t) (unsigned long) malloc((unsigned)size);

  if (!addr) {
    assert(0 && "fail to allocate memory");
    return 0;
  }
  ++stats::allocations;
  MemoryObject *res = new MemoryObject(addr, size, false, false, 
                                       false, 0, this, GPUConfig::HOST); 
  res->setName("symbdim");
  res->is_builtin = true;          
  objects.insert(res);
  return res;
}

MemoryObject *MemoryManager::allocateSharedMO(uint64_t size) {
  if (size>10*1024*1024)
    klee_warning_once(0, "Large alloc: %u bytes.  KLEE may run out of memory.", (unsigned) size);

  uint64_t address = (uint64_t) (unsigned long) malloc((unsigned) size);
  if (!address) {
    assert(0 && "fail to allocate memory");
    return 0;
  }
  
  ++stats::allocations;
  // only in the case for constructing shared memory...
  // memory type inference 
  GPUConfig::CTYPE ctype = GPUConfig::SHARED;
  MemoryObject *res = new MemoryObject(address, size, false, false, 
                                       false, 0, this, ctype);
  objects.insert(res);
  return res;
}

MemoryObject *MemoryManager::allocateFixed(uint64_t address, uint64_t size,
                                           bool is_GPU_mode, const llvm::Value *allocSite) {
#ifndef NDEBUG
  for (objects_ty::iterator it = objects.begin(), ie = objects.end();
       it != ie; ++it) {
    MemoryObject *mo = *it;
    if (address+size > mo->address && address < mo->address+mo->size)
      klee_error("Trying to allocate an overlapping object");
  }
#endif

  ++stats::allocations;
  GPUConfig::CTYPE ctype = CUDAUtil::getCType(allocSite, is_GPU_mode);
  MemoryObject *res = new MemoryObject(address, size, false, true, true,
                                       allocSite, this, ctype);
  objects.insert(res);
  return res;
}

void MemoryManager::deallocate(const MemoryObject *mo) {
  assert(0);
}

void MemoryManager::markFreed(MemoryObject *mo) {
  if (objects.find(mo) != objects.end())
  {
    if (!mo->isFixed)
      free((void *)mo->address);
    objects.erase(mo);
  }
}

MemoryManager::objects_ty & MemoryManager::getMemoryObjects() {
  return objects;
}
