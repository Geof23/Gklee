//===-- MemoryManager.h -----------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_MEMORYMANAGER_H
#define KLEE_MEMORYMANAGER_H

#include <set>
#include <stdint.h>

namespace llvm {
  class Value;
}

namespace klee {
  class MemoryObject;

  class MemoryManager {
  private:
    typedef std::set<MemoryObject*> objects_ty;
    objects_ty objects;

  public:
    MemoryManager() {}
    ~MemoryManager();

    MemoryObject *allocate(uint64_t size, bool isLocal, bool isGlobal, 
                           int deviceSet, bool is_GPU_mode, 
                           const llvm::Value *allocSite);
    MemoryObject *allocateMissedBuiltInMO(GPUConfig::CTYPE ctype, std::string vname);
    MemoryObject *allocateSymGridDim();
    MemoryObject *allocateSymBlockDim();
    MemoryObject *allocateSharedMO(uint64_t size);
    MemoryObject *allocateFixed(uint64_t address, uint64_t size, bool is_GPU_mode,
                                const llvm::Value *allocSite);
    void deallocate(const MemoryObject *mo);
    objects_ty& getMemoryObjects();
    void markFreed(MemoryObject *mo);
  };

} // End klee namespace

#endif
