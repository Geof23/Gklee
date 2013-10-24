//===-- AddressSpace.h ------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_ADDRESSSPACE_H
#define KLEE_ADDRESSSPACE_H

#include "ObjectHolder.h"

#include "CUDA.h"
#include "Memory.h"
#include "klee/Expr.h"
#include "klee/Constraints.h"
#include "klee/Internal/ADT/ImmutableMap.h"
#include "llvm/Function.h"
#include <set>

namespace llvm {
  class Instruction;
  class BranchInst;
}

namespace klee {
  class Executor;
  class ExecutionState;
  class MemoryObject;
  class ObjectState;
  class TimingSolver;

  template<class T> class ref;

  typedef std::pair<const MemoryObject*, const ObjectState*> ObjectPair;
  typedef std::vector<ObjectPair> ResolutionList;

  /// Function object ordering MemoryObject's by address.
  struct MemoryObjectLT {
    bool operator()(const MemoryObject *a, const MemoryObject *b) const;
  };

  struct MemoryAccess {
    const MemoryObject* mo;
    ref<Expr> offset;
    Expr::Width width;
    unsigned bid;          // block id
    unsigned tid;          // thread id
    llvm::Instruction *instr; // the location where this memory access occurs
    unsigned instSeqNum;
    bool isAtomic;
    bool is_write;
    ref<Expr> accessCondExpr; // the path condition corresponding 
                              // to this memory access
    ref<Expr> val;            // written value

    MemoryAccess(const MemoryObject* _mo, ref<Expr> _offset, 
		 Expr::Width _width, unsigned _bid, unsigned _tid, 
                 llvm::Instruction *_instr, unsigned _instSeqNum,
                 bool _isAtomic = false, 
                 bool _is_write = false, 
                 ref<Expr> _accessCondExpr = NULL, 
                 ref<Expr> _val = NULL) :
      offset(_offset), width(_width), bid(_bid), tid(_tid), instr(_instr), 
      instSeqNum(_instSeqNum), isAtomic(_isAtomic), 
      is_write(_is_write),
      accessCondExpr(_accessCondExpr), 
      val(_val) {
        mo = new MemoryObject(_mo);
    };
   
    MemoryAccess(const MemoryAccess &ma) : offset(ma.offset), width(ma.width), 
                                           bid(ma.bid), tid(ma.tid), 
                                           instr(ma.instr),
                                           instSeqNum(ma.instSeqNum),
                                           isAtomic(ma.isAtomic),
                                           is_write(ma.is_write), 
                                           accessCondExpr(ma.accessCondExpr), 
                                           val(ma.val) {
      mo = new MemoryObject(ma.mo);
    };
    
    void dump() const;
    void dump(Executor &executor, ExecutionState &state, ref<Expr> cond) const;
  };
  
  typedef ImmutableMap<const MemoryObject*, ObjectHolder, MemoryObjectLT> MemoryMap;
  typedef std::vector< MemoryAccess > MemoryAccessVec;

  struct InstAccess {
    unsigned bid;
    unsigned tid;
    llvm::Instruction *inst; 
    bool isBr;

    InstAccess(unsigned _bid, unsigned _tid, llvm::Instruction *_inst, bool _isBr) :
          bid(_bid), tid(_tid), inst(_inst), isBr(_isBr) {};
    InstAccess(const InstAccess &access) :
          bid(access.bid), tid(access.tid), inst(access.inst), isBr(access.isBr) {};
    void dump() const;
  };

  // To get finer statistics information, we define
  // this struct ...
  struct WarpDefectInfo {
    unsigned warpID;
    unsigned blockID;
    unsigned occur;
    bool consider;

    unsigned instReadOccur;
    unsigned instReadTotal;

    unsigned instWriteOccur;
    unsigned instWriteTotal;

    WarpDefectInfo(unsigned _warpID, unsigned _blockID) :
    warpID(_warpID), blockID(_blockID), occur(0), consider(false),
    instReadOccur(0), instReadTotal(0), instWriteOccur(0), instWriteTotal(0) {};
  };

  struct BasicBlockAccess {
    std::string funcName;
    llvm::BasicBlock *bb; 
    unsigned startIdx;
    unsigned endIdx;   

    BasicBlockAccess(std::string _funcName, llvm::BasicBlock *_bb, unsigned _startIdx, unsigned _endIdx):
                     funcName(_funcName), bb(_bb), startIdx(_startIdx), endIdx(_endIdx) {};

    ~BasicBlockAccess() {
      bb = NULL;
    }
  };

  // Used in simd-aware canonical schedule   
  struct MemoryAccessSet {
    unsigned bid; // which block
    unsigned warpNum; // which warp
    unsigned biNum; // which BI
    std::vector<MemoryAccessVec> readVecSet;
    std::vector<MemoryAccessVec> writeVecSet;
    
    MemoryAccessSet(unsigned _bid, unsigned _warpNum, unsigned _biNum): 
    bid(_bid), warpNum(_warpNum), biNum(_biNum) {};  

    MemoryAccessSet(const MemoryAccessSet &set): 
    bid(set.bid), warpNum(set.warpNum), biNum(set.biNum), 
    readVecSet(set.readVecSet), writeVecSet(set.writeVecSet) {};  

    ~MemoryAccessSet() {
      readVecSet.clear();
      writeVecSet.clear();
    }
  };

  // Used in pure canonical schedule   
  struct MemoryAccessSetPureCS {
    unsigned bid;
    unsigned biNum;
    MemoryAccessVec readSet;
    MemoryAccessVec writeSet;

    MemoryAccessSetPureCS(unsigned _bid, unsigned _biNum) : 
    bid(_bid), biNum(_biNum) {};

    MemoryAccessSetPureCS(const MemoryAccessSetPureCS &set): 
    bid(set.bid), biNum(set.biNum), 
    readSet(set.readSet), writeSet(set.writeSet) {};  

    ~MemoryAccessSetPureCS() {
      readSet.clear();
      writeSet.clear();
    }
  };

  struct DivRegion {
    unsigned bbStart;
    unsigned bbEnd;
    unsigned startIdx;
    unsigned endIdx;
    bool isEmpty;
    
    DivRegion(unsigned _bbStart, unsigned _bbEnd, unsigned _startIdx, unsigned _endIdx, bool _isEmpty) :
    bbStart(_bbStart), bbEnd(_bbEnd), startIdx(_startIdx), endIdx(_endIdx), isEmpty(_isEmpty) {};
  };

  typedef std::vector< DivRegion > DivRegionSet;
  typedef std::vector< InstAccess > InstAccessSet;
  typedef std::vector< BasicBlockAccess > BBAccessSet;
  typedef std::vector< std::vector<unsigned> > SameInstVec;  
  typedef std::vector< WarpDefectInfo > WarpDefVec;

  struct RefDivRegionSet {
    unsigned otherTid;
    DivRegionSet regionSet;

    RefDivRegionSet(unsigned _otherTid, DivRegionSet _regionSet) : 
    otherTid(_otherTid), regionSet(_regionSet) {};

    ~RefDivRegionSet() {
      regionSet.clear();
    }
  };

  typedef std::vector<RefDivRegionSet> RefDivRegionSetVec;
  typedef std::vector<MemoryAccessSet> MemoryAccessSetVec;
  typedef std::vector<MemoryAccessSetPureCS> MemoryAccessSetVecPureCS;

  class AddressSpace {
  //private:
  public:
    /// Epoch counter used to control ownership of objects.
    mutable unsigned cowKey;

    /// Unsupported, use copy constructor
    AddressSpace &operator=(const AddressSpace&); 

  public:
    /// The MemoryObject -> ObjectState map that constitutes the
    /// address space.
    ///
    /// The set of objects where o->copyOnWriteOwner == cowKey are the
    /// objects that we own.
    ///
    /// \invariant forall o in objects, o->copyOnWriteOwner <= cowKey
    MemoryMap objects;

    // type of the memory
    GPUConfig::CTYPE ctype;

    // for conflict checking
    MemoryAccessVec readSet;
    MemoryAccessVec writeSet;
    
    // for accumulated variable ...
    std::vector<MemoryAccessVec> accumWriteSets;

    // for warps, used in simd-aware canonical schedule;
    std::vector<MemoryAccessSetVec> MemAccessSets; 
    // for blocks, used in pure canonical schedule;
    std::vector<MemoryAccessSetVecPureCS> MemAccessSetsPureCS; 

    std::vector<MemoryAccessVec> symGlobalReadSets; 
    std::vector<MemoryAccessVec> symGlobalWriteSets; 
    
    unsigned numBCBI;     // number of BIs
    unsigned numBC;       // number of BIs which include bank conflicts
    
    unsigned numMCBI;     // number of BIs
    unsigned numMC;       // number of BIs which has the coalesced memory access

  public:
    AddressSpace(GPUConfig::CTYPE _ctype = GPUConfig::LOCAL) : cowKey(1), ctype(_ctype), numBCBI(0), numBC(0), numMCBI(0), numMC(0) {};
    AddressSpace(const AddressSpace &b) : 
      cowKey(++b.cowKey), objects(b.objects), ctype(b.ctype), numBCBI(b.numBCBI), numBC(b.numBC), numMCBI(b.numMCBI), numMC(b.numMC) {
      readSet = b.readSet;
      writeSet = b.writeSet;
      accumWriteSets = b.accumWriteSets;
      MemAccessSets = b.MemAccessSets;
      MemAccessSetsPureCS = b.MemAccessSetsPureCS;
      symGlobalReadSets = b.symGlobalReadSets;
      symGlobalWriteSets = b.symGlobalWriteSets;
    }
    ~AddressSpace() {}

    /// Output an information vector that gklee-mode.el (Emacs)
    /// can easily interpret
    /// \param tid1 ID of 1st thread involved
    /// \param tid2 ID of 2nd thread involved (or -1 == MAX_UINT)
    /// \param i instruction for fetching debug info
    /// \param errInfo the identifier for the information
    static void dumpEmacsInfoVect(unsigned bid1, unsigned bid2,
				  unsigned tid1, unsigned tid2,
				  const llvm::Instruction* i, const llvm::Instruction* i2,
				  std::string errInfo);

    /// Resolve address to an ObjectPair in result.
    /// \return true iff an object was found.
    bool resolveOne(const ref<ConstantExpr> &address, 
                    ObjectPair &result);

    /// Resolve address to an ObjectPair in result.
    ///
    /// \param state The state this address space is part of.
    /// \param solver A solver used to determine possible 
    ///               locations of the \a address.
    /// \param address The address to search for.
    /// \param[out] result An ObjectPair this address can resolve to 
    ///               (when returning true).
    /// \return true iff an object was found at \a address.
    bool resolveOne(ExecutionState &state, 
                    TimingSolver *solver,
                    ref<Expr> address,
                    ObjectPair &result,
                    bool &success);

    /// Resolve address to a list of ObjectPairs it can point to. If
    /// maxResolutions is non-zero then no more than that many pairs
    /// will be returned. 
    ///
    /// \return true iff the resolution is incomplete (maxResolutions
    /// is non-zero and the search terminated early, or a query timed out).
    bool resolve(ExecutionState &state,
                 TimingSolver *solver,
                 ref<Expr> address, 
                 ResolutionList &rl, 
                 unsigned maxResolutions=0,
                 double timeout=0.);

    /***/

    /// Add a binding to the address space.
    void bindObject(const MemoryObject *mo, ObjectState *os);

    /// Remove a binding from the address space.
    void unbindObject(const MemoryObject *mo);

    /// Lookup a binding from a MemoryObject.
    const ObjectState *findObject(const MemoryObject *mo) const;

    /// Lookup a binding from a MemoryObject.
    ObjectState *findNonConstantObject(const MemoryObject *mo) const;

    /// \brief Obtain an ObjectState suitable for writing.
    ///
    /// This returns a writeable object state, creating a new copy of
    /// the given ObjectState if necessary. If the address space owns
    /// the ObjectState then this routine effectively just strips the
    /// const qualifier it.
    ///
    /// \param mo The MemoryObject to get a writeable ObjectState for.
    /// \param os The current binding of the MemoryObject.
    /// \return A writeable ObjectState (\a os or a copy).
    ObjectState *getWriteable(const MemoryObject *mo, const ObjectState *os);

    /// Copy the concrete values of all managed ObjectStates into the
    /// actual system memory location they were allocated at.
    void copyOutConcretes();

    /// Copy the concrete values of all managed ObjectStates back from
    /// the actual system memory location they were allocated
    /// at. ObjectStates will only be written to (and thus,
    /// potentially copied) if the memory values are different from
    /// the current concrete values.
    ///
    /// \retval true The copy succeeded. 
    /// \retval false The copy failed because a read-only object was modified.
    bool copyInConcretes();

    bool belongToSameDivergenceRegion(const MemoryAccess &, const MemoryAccess &, 
                                      std::vector<CorrespondTid> &,  
                                      std::vector< std::vector<BranchDivRegionSet> > &,
                                      std::vector<SameInstVec> &);

    bool checkDivergeBranchRace(Executor &, ExecutionState &, 
                                const MemoryAccessVec &, const MemoryAccessVec &,
                                std::vector<CorrespondTid> &, 
                                std::vector<InstAccessSet> &, 
                                std::vector< std::vector<BranchDivRegionSet> > &,
                                std::vector<SameInstVec> &, 
                                bool, ref<Expr> &, unsigned &); 

    void constructGlobalMemAccessSet(Executor &, ExecutionState &, std::vector<CorrespondTid> &,
                                     std::vector<InstAccessSet> &, std::vector<RefDivRegionSetVec> &, 
                                     std::vector<SameInstVec> &, unsigned BINum);

    /// check races 
    bool hasRaceInSharePureCS(Executor &, ExecutionState &, 
                              ref<Expr> &, unsigned &);
    bool hasRaceInShare(Executor &, ExecutionState &, 
                        std::vector<CorrespondTid> &, 
                        std::vector<InstAccessSet> &, 
                        std::vector<RefDivRegionSetVec> &, 
                        std::vector<SameInstVec> &,
                        std::vector< std::vector<BranchDivRegionSet> > &,
                        ref<Expr> &, unsigned &);
    bool hasRaceInGlobalWithinSameBlockPureCS(Executor &, ExecutionState &);
    bool hasRaceInGlobalAcrossBlocksPureCS(Executor &, ExecutionState &);
    bool hasRaceInGlobalWithinSameBlock(Executor &, ExecutionState &, 
                                        std::vector<CorrespondTid> &, 
                                        std::vector<InstAccessSet>&, 
                                        std::vector<RefDivRegionSetVec> &, 
                                        std::vector< std::vector<BranchDivRegionSet> > &,
                                        std::vector<SameInstVec> &,
                                        ref<Expr> &, unsigned &, unsigned);
    bool hasRaceInGlobalAcrossBlocks(Executor &, ExecutionState &, 
                                     std::vector<CorrespondTid> &, 
                                     ref<Expr> &, unsigned &);
    bool hasSymRaceInSharePureCS(Executor &, ExecutionState &);
    bool hasSymRaceInGlobalWithinBlockPureCS(Executor &, ExecutionState &);
    bool hasSymRaceInShare(Executor &, ExecutionState &);
    bool hasAccumVarInShare(Executor &, ExecutionState &);
    bool hasSymRaceInGlobalWithinBlock(Executor &, ExecutionState &);
    bool hasSymRaceInGlobalAcrossBlocks(Executor &, ExecutionState &, bool);

    /// check bank conflicts 
    bool hasBankConflict(Executor &, ExecutionState &, 
                         unsigned, std::vector<CorrespondTid> &,
                         std::vector<InstAccessSet> &, 
                         std::vector<RefDivRegionSetVec> &, 
                         std::vector<SameInstVec> &, 
                         ref<Expr> &bcCond, WarpDefVec &bcWDVec, 
                         bool &Consider, unsigned &queryNum);
    bool hasSymBankConflict(Executor &, ExecutionState &, unsigned);
    /// check whether there exist race after __syncthread is removed ..
    //bool hasRaceAfterBIRemoval();
    /// Check memory coalescing under device capability 1.0 or 1.x ...
    bool hasMemoryCoalescingCap0(Executor &, ExecutionState &, 
                                 std::vector<CorrespondTid> &, 
                                 std::vector<InstAccessSet> &, 
                                 std::vector<RefDivRegionSetVec> &, 
                                 std::vector<SameInstVec> &, ref<Expr> &, 
                                 WarpDefVec &, bool &, unsigned &);
    bool hasSymMemoryCoalescingCap0(Executor &, ExecutionState &);
    // Check memory coalescing under device capability 1.2 or 1.3 ...
    bool hasMemoryCoalescingCap1(Executor &, ExecutionState &, 
                                 std::vector<CorrespondTid> &, 
                                 std::vector<InstAccessSet> &, 
                                 std::vector<RefDivRegionSetVec> &, 
                                 std::vector<SameInstVec> &, ref<Expr> &, 
                                 WarpDefVec &, bool &, unsigned &);
    bool hasSymMemoryCoalescingCap1(Executor &, ExecutionState &);
    // Check memory coalescing under capability 2.x ...
    bool hasMemoryCoalescingCap2(Executor &, ExecutionState &, 
                                 std::vector<CorrespondTid> &, 
                                 std::vector<InstAccessSet> &, 
                                 std::vector<RefDivRegionSetVec> &, 
                                 std::vector<SameInstVec> &, ref<Expr> &, 
                                 WarpDefVec &, bool &, unsigned &);
    bool hasSymMemoryCoalescingCap2(Executor &, ExecutionState &);
    /// check volatile missing ... 
    bool hasVolatileMissing(Executor &, ExecutionState &, 
                            std::vector<CorrespondTid> &, 
                            std::vector<InstAccessSet> &, 
                            std::vector<RefDivRegionSetVec> &, 
                            std::vector<SameInstVec> &, ref<Expr> &);
    bool hasSymVolatileMissing(Executor &, ExecutionState &);
    /// print out the content of the address space
    void clearGlobalMemoryAccessSets(); 
    void clearSymGlobalMemoryAccessSets(); 
    void dump(bool rwset_only = false);
    void clearAccessSet() { 
      readSet.clear(); 
      writeSet.clear(); 
    };
  };

  class HierAddressSpace {     // CUDA memory hierarchy
    unsigned numWDBI;
    unsigned numWD;

    unsigned raceQueryNum;
    unsigned mcQueryNum;
    unsigned bcQueryNum;
  public:
    AddressSpace cpuMemory;
    AddressSpace deviceMemory;

    std::vector<AddressSpace> sharedMemories;     // for blocks
    std::vector<AddressSpace> localMemories;      // for threads
    
    std::vector<InstAccessSet> instAccessSets; // for threads 
    std::vector<BBAccessSet> bbAccessSets; // for threads 
    std::vector<RefDivRegionSetVec> divRegionSets;   // for threads 
    std::vector<SameInstVec> sameInstVecSets;  // for each warp

    std::vector<BranchDivRegionSet> branchDivRegionSets;  // for each warp ...
    std::vector< std::vector<BranchDivRegionSet> > warpsBranchDivRegionSets;

    bool hasBC;
    bool hasNoMC;
    bool hasVM;

    ref<Expr> bcCondComb;
    ref<Expr> nonMCCondComb;
    ref<Expr> vmCondComb;

    std::vector<WarpDefVec> bcWDSet;
    std::vector<WarpDefVec> nomcWDSet;
    std::vector<WarpDefVec> wdWDSet;

    HierAddressSpace();
    HierAddressSpace(const HierAddressSpace &address);

    AddressSpace& getAddressSpace(GPUConfig::CTYPE ctype = GPUConfig::LOCAL, 
				  unsigned b_t_index = 0);
    /// Add a binding to the address space.
    void bindObject(const MemoryObject *mo, ObjectState *os, unsigned b_t_index = 0);

    /// Remove a binding from the address space.
    void unbindObject(const MemoryObject *mo, unsigned b_t_index = 0);

    /// Lookup a binding from a MemoryObject.
    const ObjectState *findObject(const MemoryObject *mo, unsigned b_t_index = 0);

    /// Lookup a binding from a MemoryObject.
    ObjectState *findNonConstantObject(const MemoryObject *mo, unsigned b_t_index = 0);

    ObjectState *getWriteable(const MemoryObject *mo, const ObjectState *os, 
			      unsigned b_t_index = 0);

    // add an element to the write set
    void addWrite(bool is_GPU_mode, const MemoryObject *mo, ref<Expr> &offset, 
                  ref<Expr> &val, Expr::Width width, 
                  unsigned bid, unsigned tid, 
                  llvm::Instruction *instr, unsigned seqNum,
                  bool isAtomic, unsigned b_t_index = 0, 
                  ref<Expr> accessExpr = NULL);
    // add an element to the read set
    void addRead(bool is_GPU_mode, const MemoryObject *mo, ref<Expr> &offset, 
                 ref<Expr> &val, Expr::Width width, unsigned bid, unsigned tid, 
                 llvm::Instruction *instr, unsigned seqNum,
                 bool isAtomic, unsigned b_t_index = 0, 
                 ref<Expr> accessExpr = NULL);

    // insert instruction
    void insertInst(bool is_GPU_mode, unsigned bid, unsigned tid, 
                    llvm::Instruction *instr, 
                    bool isBr, unsigned &seqNum);
    /// Resolve address to an ObjectPair in result.
    /// \return true iff an object was found.
    bool resolveOne(const ref<ConstantExpr> &address, 
                    ObjectPair &result, 
		    GPUConfig::CTYPE ctype,  
		    unsigned b_t_index = 0);

    /// Resolve address to an ObjectPair in result.
    /// \return true iff an object was found at \a address.
    bool resolveOne(ExecutionState &state, 
                    TimingSolver *solver,
                    ref<Expr> address,
                    ObjectPair &result,
                    bool &success,
		    GPUConfig::CTYPE ctype,
		    unsigned b_t_index = 0);

    /// Resolve address to a list of ObjectPairs it can point to. If
    /// maxResolutions is non-zero then no more than that many pairs
    /// will be returned. 
    ///
    /// \return true iff the resolution is incomplete (maxResolutions
    /// is non-zero and the search terminated early, or a query timed out).
    bool resolve(ExecutionState &state,
                 TimingSolver *solver,
                 ref<Expr> address, 
                 ResolutionList &rl, 
                 unsigned maxResolutions,
                 double timeout,
		 GPUConfig::CTYPE ctype,  
		 unsigned b_t_index = 0);
    

    /// the following two functions are for external function calls

    /// Copy the concrete values of all managed ObjectStates into the
    /// actual system memory location they were allocated at.
    void copyOutConcretes(unsigned tid, unsigned bid = 0);

    /// Copy the concrete values of all managed ObjectStates back from
    /// the actual system memory location they were allocated
    /// at. ObjectStates will only be written to (and thus,
    /// potentially copied) if the memory values are different from
    /// the current concrete values.
    ///
    /// \retval true The copy succeeded. 
    /// \retval false The copy failed because a read-only object was modified.
    bool copyInConcretes(unsigned tid, unsigned bid = 0);

    /// print out the contents of all the address spaces
    void dump(char mask = 7);
    void dumpInstAccessSet();
    void dumpVerboseAddressSpaceMO();

    void clearAccessSet(char mask = 15);
    void clearInstAccessSet(bool clearAll);
    void clearGlobalAccessSet();
    void clearWarpDefectSet();
    
    // check races
    bool hasRaceInShare(Executor &, ExecutionState &, 
                        std::vector<CorrespondTid> &, ref<Expr> &);
    bool hasSymRaceInShare(Executor &, ExecutionState &);
    bool hasRaceInGlobal(Executor &, ExecutionState &, 
                         std::vector<CorrespondTid> &, ref<Expr> &, 
                         unsigned, bool);
    bool hasSymRaceInGlobal(Executor &, ExecutionState &, bool);
    void checkMemoryAccessThreadParametric(Executor &, ExecutionState &);
    void constructGlobalMemAccessSets(Executor &, ExecutionState &, 
                                      std::vector<CorrespondTid>&, unsigned);
    bool foundMismatchBarrierInParametricFlow(ExecutionState &, unsigned, unsigned);
    bool hasMismatchBarrierInParametricFlow(Executor &, ExecutionState &);

    // check bank conflicts
    bool hasBankConflict(Executor &, ExecutionState &, std::vector<CorrespondTid> &, unsigned);
    // check bank conflicts under the symbolic configuration 
    bool hasSymBankConflict(Executor &, ExecutionState &, unsigned);
    // check whether there exist memory coalescing ..
    bool hasMemoryCoalescing(Executor &, ExecutionState &, std::vector<CorrespondTid> &, unsigned);
    // check memory coalescing under the symbolic configuration 
    bool hasSymMemoryCoalescing(Executor &, ExecutionState &, unsigned);
    // check whether there exist warp divergence ..
    bool hasWarpDivergence(std::vector<CorrespondTid> &);
    bool hasSymWarpDivergence(ExecutionState &state);
    bool hasVolatileMissing(Executor &, ExecutionState &, std::vector<CorrespondTid> &);
    bool hasSymVolatileMissing(Executor &, ExecutionState &);

    void forwardingExploreInstSet(unsigned, unsigned);
    void constructDivergRegionSets(std::vector<CorrespondTid> &, unsigned, unsigned);
    void dumpWarpsBranchDivRegionSets(); 
    void ensureThreadsDivergenceRegion(std::vector<CorrespondTid> &); 

    void getBCRate(unsigned &, unsigned &, unsigned &, unsigned &);
    void getMCRate(unsigned &, unsigned &, unsigned &, unsigned &);
    void getWDRate(unsigned &, unsigned &, unsigned &, unsigned &);
    void getRaceRate();
  };

  class AddressSpaceUtil {
    public: 
      static bool evaluateQueryMustBeTrue(Executor &, ExecutionState &, ref<Expr> &, bool &, bool &);
      static bool evaluateQueryMustBeFalse(Executor &, ExecutionState &, ref<Expr> &, bool &, bool &);
      static bool isTwoInstIdentical(llvm::Instruction *inst1, llvm::Instruction *inst2); 
      static void constructTmpRWSet(Executor &, ExecutionState &, 
                                    MemoryAccessVec &, MemoryAccessVec &, 
                                    std::vector<CorrespondTid> &, 
                                    std::vector<InstAccessSet> &, 
                                    std::vector<RefDivRegionSetVec> &, 
                                    std::vector<SameInstVec> &, unsigned);
      static void updateBuiltInRelatedConstraint(ExecutionState &, 
                                                 ConstraintManager &, 
                                                 ref<Expr> &);
      static void updateMemoryAccess(ExecutionState &, ConstraintManager &, 
                                     MemoryAccess &);
      static ref<Expr> constructSameBlockExpr(ExecutionState &, Expr::Width); 
      static ref<Expr> constructSameThreadExpr(ExecutionState &, Expr::Width); 
      static ref<Expr> constructRealThreadNumConstraint(ExecutionState &, 
                                                        unsigned, Expr::Width); 
      static ref<Expr> threadSameWarpConstraint(ExecutionState &, unsigned); 
      static ref<Expr> threadSameBlockDiffWarpConstraint(ExecutionState &, 
                                                         unsigned); 
      static ref<Expr> threadDiffBlockConstraint(ExecutionState &); 
   };

} // End klee namespace

#endif
