//===-- Executor.h ----------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class to perform actual execution, hides implementation details from external
// interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_EXECUTOR_H
#define KLEE_EXECUTOR_H

#include "klee/ExecutionState.h"
#include "klee/Interpreter.h"
#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "llvm/Support/CallSite.h"
#include <vector>
#include <string>
#include <map>
#include <set>

#include "CUDA.h"
#include "PathReduction.h"

struct KTest;

namespace llvm {
  class BasicBlock;
  class BranchInst;
  class CallInst;
  class Constant;
  class ConstantExpr;
  class Function;
  class GlobalValue;
  class Instruction;
  class TargetData;
  class Twine;
  class Value;
}

namespace klee {
  class Array;
  struct Cell;
  class ExecutionState;
  class ExternalDispatcher;
  class Expr;
  class InstructionInfoTable;
  struct KFunction;
  struct KInstruction;
  class KInstIterator;
  class KModule;
  class MemoryManager;
  class MemoryObject;
  class ObjectState;
  class PTree;
  class Searcher;
  class SeedInfo;
  class SpecialFunctionHandler;
  struct StackFrame;
  class StatsTracker;
  class TimingSolver;
  class TreeStreamWriter;

  //  class StringHandler;
  template<class T> class ref;

  /// \todo Add a context object to keep track of data only live
  /// during an instruction step. Should contain addedStates,
  /// removedStates, and haltExecution, among others.

class SharedMemoryObject {
public:
  unsigned kernelNum;
  MemoryObject *sharedGLMO;
  Cell *glCell;
  
  explicit SharedMemoryObject(unsigned _kernelNum, 
                              MemoryObject *_sharedGLMO, 
                              Cell *_glCell) : kernelNum(_kernelNum), 
                                               sharedGLMO(_sharedGLMO), 
                                               glCell(_glCell) {}
  ~SharedMemoryObject() {
    delete glCell;
  }
};

class AccumMemoryMark {
public:
  klee::ref<Expr> soffset;
  klee::ref<Expr> eoffset;
  klee::ref<Expr> value;
  bool isAccum;
  
  explicit AccumMemoryMark(klee::ref<Expr> _soffset, 
                           klee::ref<Expr> _eoffset, 
                           klee::ref<Expr> _value, 
                           bool _isAccum) : soffset(_soffset), 
                                            eoffset(_eoffset),
                                            value(_value), 
                                            isAccum(_isAccum) {}
  ~AccumMemoryMark();
};

class ExternSharedVar {
public:
  unsigned kernelNum;
  unsigned tableNum; // the location in constantTable...
  MemoryObject *externSharedMO;

  explicit ExternSharedVar(unsigned _kernelNum, 
                           unsigned _tableNum, 
                           MemoryObject *_externSharedMO) : kernelNum(_kernelNum), 
                                                            tableNum(_tableNum), 
                                                            externSharedMO(_externSharedMO) {}
};

class Executor : public Interpreter {
  friend class BumpMergingSearcher;
  friend class MergingSearcher;
  friend class RandomPathSearcher;
  friend class OwningSearcher;
  friend class WeightedRandomSearcher;
  friend class SpecialFunctionHandler;
  friend class StatsTracker;

public:
  class Timer {
  public:
    Timer();
    virtual ~Timer();

    /// The event callback.
    virtual void run() = 0;
  };

  typedef std::pair<ExecutionState*,ExecutionState*> StatePair;

  typedef std::vector<ExternSharedVar> ExternSharedVarVec;

  // Make the following private members / functions public so that 
  // the string handler can access them
  TimingSolver *solver;
  llvm::PostDominatorTree *postDominator;
  PRInfo PR_info;
  BCCoverage bc_cov_monitor;
  clock_t concreteStart, concreteEnd;
  clock_t symStart, symEnd;
  double concreteTotalTime;
  double symTotalTime;
  std::vector<ExternSharedVarVec> externSharedSet;

  /// Return a unique constant value for the given expression in the
  /// given state, if it has one (i.e. it provably only has a single
  /// value). Otherwise return the original expression.
  klee::ref<Expr> toUnique(ExecutionState &state, klee::ref<Expr> &e);

  void bindLocal(KInstruction *target, 
                 ExecutionState &state, 
                 klee::ref<Expr> value);

  ObjectState *bindObjectInState(ExecutionState &state, 
                                 const MemoryObject *mo,
                                 bool isLocal, const Array *array = 0);

  ObjectState *bindObjectInStateToShared(ExecutionState &state, 
                                         const MemoryObject *mo, unsigned bid);

  /// Resolve a pointer to the memory objects it could point to the
  /// start of, forking execution when necessary and generating errors
  /// for pointers to invalid locations (either out of bounds or
  /// address inside the middle of objects).
  ///
  /// \param results[out] A list of ((MemoryObject,ObjectState),
  /// state) pairs for each object the given address can point to the
  /// beginning of.
  typedef std::vector< std::pair<std::pair<const MemoryObject*, const ObjectState*>, 
                                 ExecutionState*> > ExactResolutionList;
  void resolveExact(ExecutionState &state,
                    klee::ref<Expr> p,
                    ExactResolutionList &results,
                    const std::string &name, 
		    GPUConfig::CTYPE ctype,  
		    unsigned b_t_index = 0);

  void terminateStateOnError(ExecutionState &state, 
                             const llvm::Twine &message,
                             const char *suffix,
                             const llvm::Twine &longMessage="");

  const Cell& evalSharedMemory(ExecutionState &state, klee::ref<Expr> &pointer, 
                               unsigned index);
  const Cell& eval(KInstruction *ki, unsigned index, 
                   ExecutionState &state);

  void evaluateConstraintAsNewFlow(ExecutionState &state, ParaTree &pTree, 
                                   klee::ref<Expr> &cond, bool flowCreated);

  void evaluateConstraintAsNewFlowUnderRacePrune(ExecutionState &state, ParaTree &pTree, 
                                                 klee::ref<Expr> &cond, bool flowCreated, 
                                                 llvm::BranchInst *bi);

  bool identifyConditionType(ExecutionState &state, klee::ref<Expr> &cond, 
                             bool &relatedToSym, bool &accum);
 
  void clearConfigRelatedConstraints(ExecutionState &state);
  // ***

private:
  class TimerInfo;

  KModule *kmodule;
  InterpreterHandler *interpreterHandler;
  Searcher *searcher;
  bool is_GPU_mode; // For convenience, some member functions 
                    // need this...
  bool accumStore;
  std::map<llvm::Instruction*, MemoryAccess> accumTaintSet; 
  klee::ref<Expr> atomicRes;
  std::set<std::string> kernelSet;     // global function set  
  std::set<std::string> builtInSet;    // builtIn variables set 
  llvm::Function *kernelFunc; 

  ExternalDispatcher *externalDispatcher;
  MemoryManager *memory;
  std::set<ExecutionState*> states;
  StatsTracker *statsTracker;
  TreeStreamWriter *pathWriter, *symPathWriter;
  SpecialFunctionHandler *specialFunctionHandler;
  std::vector<TimerInfo*> timers;
  PTree *processTree;
  std::string traceInfo;

  /// Used to track states that have been added during the current
  /// instructions step. 
  /// \invariant \ref addedStates is a subset of \ref states. 
  /// \invariant \ref addedStates and \ref removedStates are disjoint.
  std::set<ExecutionState*> addedStates;
  /// Used to track states that have been removed during the current
  /// instructions step. 
  /// \invariant \ref removedStates is a subset of \ref states. 
  /// \invariant \ref addedStates and \ref removedStates are disjoint.
  std::set<ExecutionState*> removedStates;

  /// When non-empty the Executor is running in "seed" mode. The
  /// states in this map will be executed in an arbitrary order
  /// (outside the normal search interface) until they terminate. When
  /// the states reach a symbolic branch then either direction that
  /// satisfies one or more seeds will be added to this map. What
  /// happens with other states (that don't satisfy the seeds) depends
  /// on as-yet-to-be-determined flags.
  std::map<ExecutionState*, std::vector<SeedInfo> > seedMap;
  
  /// Map of globals to their representative memory object.
  std::map<const llvm::GlobalValue*, MemoryObject*> globalObjects;

  /// Map of globals to their bound address. This also includes
  /// globals that have no representative object (i.e. functions).
  std::map<const llvm::GlobalValue*, klee::ref<ConstantExpr> > globalAddresses;

  /// Vector of global base address for shared memory handled as global variables 
  /// to their representative memory object.
  std::vector< std::pair<SharedMemoryObject*, MemoryObject*> > sharedObjects;

  /// The set of legal function addresses, used to validate function
  /// pointers. We use the actual Function* address as the function address.
  std::set<uint64_t> legalFunctions;

  /// When non-null the bindings that will be used for calls to
  /// klee_make_symbolic in order replay.
  const struct KTest *replayOut;
  /// When non-null a list of branch decisions to be used for replay.
  const std::vector<bool> *replayPath;
  /// The index into the current \ref replayOut or \ref replayPath
  /// object.
  unsigned replayPosition;

  /// When non-null a list of "seed" inputs which will be used to
  /// drive execution.
  const std::vector<struct KTest *> *usingSeeds;  

  /// Disables forking, instead a random path is chosen. Enabled as
  /// needed to control memory usage. \see fork()
  bool atMemoryLimit;

  /// Disables forking, set by client. \see setInhibitForking()
  bool inhibitForking;

  /// Signals the executor to halt execution at the next instruction
  /// step.
  bool haltExecution;  

  /// Whether implied-value concretization is enabled. Currently
  /// false, it is buggy (it needs to validate its writes).
  bool ivcEnabled;

  /// The maximum time to allow for a single stp query.
  double stpTimeout;  

  // The BI statistics
  unsigned bcBINum;
  unsigned mcBINum;
  unsigned wdBINum;
  unsigned bcBISum;
  unsigned mcBISum;
  unsigned wdBISum;
  // The Warp statistics
  unsigned bcWarpNum;
  unsigned mcWarpNum;
  unsigned wdWarpNum;
  unsigned bcWarpSum;
  unsigned mcWarpSum;
  unsigned wdWarpSum;

  bool symBC;
  bool symMC;
  bool symWD;
  bool symVMiss;
  bool symRace;

  llvm::Function* getTargetFunction(llvm::Value *calledVal,
                                    ExecutionState &state);
  
  void updateCType(ExecutionState &state, llvm::Value *value, 
                   klee::ref<Expr> &base, bool is_GPU_mode);

  void executeInstruction(ExecutionState &state, KInstruction *ki);

  void printFileLine(ExecutionState &state, KInstruction *ki);

  void updateParaTreeSet(ExecutionState &state);

  void updateParaTreeSetUnderRacePrune(ExecutionState &state);

  void handleEnterGPUMode(ExecutionState &state);

  void updateConstantTable(unsigned kernelNum);

  void configurateGPUKernelSet();

  void run(ExecutionState &initialState);

  // Given a concrete object in our [klee's] address space, add it to 
  // objects checked code can reference.
  MemoryObject *addExternalObject(ExecutionState &state, void *addr, 
                                  unsigned size, bool isReadOnly);

  void handleBuiltInVariablesAsSymbolic(ExecutionState &state, MemoryObject* mo, 
			                std::string vname);

  void handleBuiltInVariables(ExecutionState &state, MemoryObject* mo, 
			      std::string vname);

  void initializeMissedBuiltInVariables(ExecutionState &state);

  void initializeGlobalObject(ExecutionState &state, ObjectState *os, 
			      const llvm::Constant *c,
			      unsigned offset);

  void initializeExternalSharedGlobals(ExecutionState &state);

  void initializeGlobals(ExecutionState &state);

  void stepInstruction(ExecutionState &state);
  void updateStates(ExecutionState *current);
  void contextSwitchToNextThread(ExecutionState &state);
  void transferToBasicBlock(llvm::BasicBlock *dst, 
			    llvm::BasicBlock *src,
			    ExecutionState &state);

  void callExternalFunction(ExecutionState &state,
                            KInstruction *target,
                            llvm::Function *function,
                            std::vector< klee::ref<Expr> > &arguments);

  // by Guodong
  // process the intrinsic functions generated while compiling C++ programs
  void callIntrinsicFunction(ExecutionState &state,
			     KInstruction *target,
			     llvm::Function *function,
			     std::vector< klee::ref<Expr> > &arguments);

  /// Allocate and bind a new object in a particular state. NOTE: This
  /// function may fork.
  ///
  /// \param isLocal Flag to indicate if the object should be
  /// automatically deallocated on function return (this also makes it
  /// illegal to free directly).
  ///
  /// \param target Value at which to bind the base address of the new
  /// object.
  ///
  /// \param reallocFrom If non-zero and the allocation succeeds,
  /// initialize the new object from the given one and unbind it when
  /// done (realloc semantics). The initialized bytes will be the
  /// minimum of the size of the old and new objects, with remaining
  /// bytes initialized as specified by zeroMemory.
  void executeAlloc(ExecutionState &state,
                    klee::ref<Expr> size,
                    bool isLocal,
                    KInstruction *target,
                    bool zeroMemory=false,
                    const ObjectState *reallocFrom=0);

  /// Free the given address with checking for errors. If target is
  /// given it will be bound to 0 in the resulting states (this is a
  /// convenience for realloc). Note that this function can cause the
  /// state to fork and that \ref state cannot be safely accessed
  /// afterwards.
  void executeFree(ExecutionState &state,
                   klee::ref<Expr> address,
                   KInstruction *target = 0);
  
  void executeCall(ExecutionState &state, 
                   KInstruction *ki,
                   llvm::Function *f,
                   std::vector< klee::ref<Expr> > &arguments, 
                   unsigned seqNum = 0);

  void executeRaceCondition(ExecutionState &state, klee::ref<Expr> &raceCond);

  void executeNoMemoryCoalescing(ExecutionState &state, klee::ref<Expr> &noMCCond);

  void executeBankConflict(ExecutionState &state, klee::ref<Expr> &bcCond);

  void executeVolatileMissing(ExecutionState &state, klee::ref<Expr> &mcCond);

  void updateBaseCType(ExecutionState &state, klee::ref<Expr> &baseAddr);  
                   
  void dumpTmpOutOfBoundConfig(ExecutionState &state, klee::ref<Expr> boundExpr, klee::ref<Expr> offset);
  // do address resolution / object binding / out of bounds checking
  // and perform the operation
  void executeMemoryOperation(ExecutionState &state,
                              bool isWrite,
                              klee::ref<Expr> address,
                              klee::ref<Expr> value /* undef if read */,
                              KInstruction *target /* undef if write */,
                              unsigned seqNum = 0,  
                              bool isAtomic = false);

  void executeMakeSymbolic(ExecutionState &state, 
                           const MemoryObject *mo,
                           const std::string &name);

  /// Create a new state where each input condition has been added as
  /// a constraint and return the results. The input state is included
  /// as one of the results. Note that the output vector may included
  /// NULL pointers for states which were unable to be created.
  void branch(ExecutionState &state, 
              const std::vector< klee::ref<Expr> > &conditions,
              std::vector<ExecutionState*> &result);

  //bool identifyAccumVariable(ExecutionState &state, const MemoryObject *mo, 
  //                           klee::ref<Expr> offset, unsigned bytes, klee::ref<Expr> value); 

  // Fork current and return states in which condition holds / does
  // not hold, respectively. One of the states is necessarily the
  // current state, and one of the states may be null.
  StatePair fork(ExecutionState &current, klee::ref<Expr> condition, bool isInternal);

  /// Add the given (boolean) condition as a constraint on state. This
  /// function is a wrapper around the state's addConstraint function
  /// which also manages manages propogation of implied values,
  /// validity checks, and seed patching.
  void addConstraint(ExecutionState &state, klee::ref<Expr> condition);

  // Called on [for now] concrete reads, replaces constant with a symbolic
  // Used for testing.
  klee::ref<Expr> replaceReadWithSymbolic(ExecutionState &state, klee::ref<Expr> e);

  // construct shared memory successfully. 
  void constructSharedMemory(ExecutionState &state, unsigned bid);

  // clear shared memory. 
  void clearSharedMemory(ExecutionState &state);

  Cell& getArgumentCell(ExecutionState &state,
                        KFunction *kf,
                        unsigned index) {
    return state.getCurStack().back().locals[kf->getArgRegister(index)];
  }

  Cell& getDestCell(ExecutionState &state,
                    KInstruction *target) {
    return state.getCurStack().back().locals[target->dest];
  }

  void bindArgument(KFunction *kf, 
                    unsigned index,
                    ExecutionState &state,
                    klee::ref<Expr> value);

  klee::ref<klee::ConstantExpr> evalConstantExpr(const llvm::ConstantExpr *ce);

  /// Return a constant value for the given expression, forcing it to
  /// be constant in the given state by adding a constraint if
  /// necessary. Note that this function breaks completeness and
  /// should generally be avoided.
  ///
  /// \param purpose An identify string to printed in case of concretization.
  klee::ref<klee::ConstantExpr> toConstant(ExecutionState &state, klee::ref<Expr> e, 
                                     const char *purpose);

  klee::ref<klee::ConstantExpr> toConstantArguments(ExecutionState &state, 
                                              klee::ref<Expr> e, 
                                              const char *reason);

  /// Bind a constant value for e to the given target. NOTE: This
  /// function may fork state if the state has multiple seeds.
  void executeGetValue(ExecutionState &state, klee::ref<Expr> e, KInstruction *target);

  /// Get textual information regarding a memory address.
  std::string getAddressInfo(ExecutionState &state, klee::ref<Expr> address) const;

  void concludeExploredTime(ExecutionState &state);
  // remove state from queue and delete
  void terminateState(ExecutionState &state);
  // conclude the entire rate information... 
  void concludeRateStatistics(ExecutionState &state); 
  // generate the test cases for non-mc, bc, vm...
  void processPerformDefectTestCase(ExecutionState &state);
  // call exit handler and terminate state
  void terminateStateEarly(ExecutionState &state, const llvm::Twine &message);
  // call exit handler and terminate state
  void terminateStateOnExit(ExecutionState &state);
  // the difference of this function from others is 
  // no removal of state from queue. 
  void terminateStateOnPerformanceError(ExecutionState &state);
  // call error handler and terminate state
//   void terminateStateOnError(ExecutionState &state, 
//                              const llvm::Twine &message,
//                              const char *suffix,
//                              const llvm::Twine &longMessage="");

  // call error handler and terminate state, for execution errors
  // (things that should not be possible, like illegal instruction or
  // unlowered instrinsic, or are unsupported, like inline assembly)
  void terminateStateOnExecError(ExecutionState &state, 
                                 const llvm::Twine &message,
                                 const llvm::Twine &info="") {
    terminateStateOnError(state, message, "exec.err", info);
  }

  /// bindModuleConstants - Initialize the module constant table.
  void bindModuleConstants();

  bool forkNewParametricFlow(ExecutionState &state, KInstruction *ki);

  bool forkNewParametricFlowUnderRacePrune(ExecutionState &state, KInstruction *ki);

  template <typename TypeIt>
  void computeOffsets(KGEPInstruction *kgepi, TypeIt ib, TypeIt ie);

  /// bindInstructionConstants - Initialize any necessary per instruction
  /// constant values.
  void bindInstructionConstants(KInstruction *KI);

  void handlePointsToObj(ExecutionState &state, 
                         KInstruction *target, 
                         const std::vector<klee::ref<Expr> > &arguments);

  void doImpliedValueConcretization(ExecutionState &state,
                                    klee::ref<Expr> e,
                                    klee::ref<ConstantExpr> value);

  /// Add a timer to be executed periodically.
  ///
  /// \param timer The timer object to run on firings.
  /// \param rate The approximate delay (in seconds) between firings.
  void addTimer(Timer *timer, double rate);

  void initTimers();
  void processTimers(ExecutionState *current,
                     double maxInstTime);
                
public:
  Executor(const InterpreterOptions &opts, InterpreterHandler *ie);
  virtual ~Executor();

  const InterpreterHandler& getHandler() {
    return *interpreterHandler;
  }

  // XXX should just be moved out to utility module
  klee::ref<klee::ConstantExpr> evalConstant(const llvm::Constant *c);
  
  bool getGPUMode() const {
    return is_GPU_mode;
  }

  klee::ref<Expr> getAtomicRes() const {
    return atomicRes;
  }

  void encounterBarrier(ExecutionState &state, 
                        KInstruction *target, 
                        bool is_end_GPU_barrier, 
                        bool &allThreadsBarrier); 

  void handleBarrier(ExecutionState &state, 
                     KInstruction *target); 

  void handleMemfence(ExecutionState &state, 
                      KInstruction *target); 

  void handleEndGPU(ExecutionState &state, 
                    KInstruction *target);

  void executeAtomicAdd(ExecutionState &state, 
                        KInstruction *target, std::string fName, 
                        std::vector< klee::ref<Expr> > &arguments, 
                        unsigned seqNum); 

  void executeAtomicExch(ExecutionState &state, KInstruction *target, 
                         std::vector< klee::ref<Expr> > &arguments, 
                         unsigned seqNum); 

  void compareValue(ExecutionState &state, 
                    KInstruction *target, std::string fName, 
                    std::vector< klee::ref<Expr> > &arguments,
                    unsigned seqNum, 
                    klee::ref<Expr> base, bool isMin); 

  void executeAtomicMin(ExecutionState &state, 
                        KInstruction *target, std::string fName, 
                        std::vector< klee::ref<Expr> > &arguments, 
                        unsigned seqNum);

  void executeAtomicMax(ExecutionState &state, 
                        KInstruction *target, std::string fName, 
                        std::vector< klee::ref<Expr> > &arguments, 
                        unsigned seqNum); 

  void executeAtomicInc(ExecutionState &state, 
                        KInstruction *target, 
                        std::vector< klee::ref<Expr> > &arguments, 
                        unsigned seqNum); 

  void executeAtomicDec(ExecutionState &state, 
                        KInstruction *target, 
                        std::vector< klee::ref<Expr> > &arguments, 
                        unsigned seqNum);

  void executeAtomicCAS(ExecutionState &state, KInstruction *target, 
                        std::vector< klee::ref<Expr> > &arguments, 
                        unsigned seqNum); 

  void executeAtomicBitWise(ExecutionState &state, 
                            KInstruction *target, std::string fName,
                            std::vector< klee::ref<Expr> > &arguments, 
                            unsigned seqNum); 

  bool executeCUDAAtomic(ExecutionState &state,
                         KInstruction *target, std::string fName,
                         std::vector< klee::ref<Expr> > &arguments, 
                         unsigned seqNum); 

  void executeCUDAIntrinsics(ExecutionState &state, 
                             KInstruction *target,
                             llvm::Function *f, 
                             std::vector< klee::ref<Expr> > &arguments, 
                             unsigned seqNum);

  klee::ref<ConstantExpr> toConstantPublic(ExecutionState &state, 
                                     klee::ref<Expr> e, 
                                     const char *reason);

  StatePair forkAsPublic(ExecutionState &current, 
                         klee::ref<Expr> cond, bool isInternal);

  void terminateStateOnExecErrorPublic(ExecutionState &state, 
                                       const llvm::Twine &message, 
                                       const llvm::Twine &info="") {
    terminateStateOnExecError(state, message, info);
  } 

  virtual void setPathWriter(TreeStreamWriter *tsw) {
    pathWriter = tsw;
  }
  virtual void setSymbolicPathWriter(TreeStreamWriter *tsw) {
    symPathWriter = tsw;
  }      

  virtual void setReplayOut(const struct KTest *out) {
    assert(!replayPath && "cannot replay both buffer and path");
    replayOut = out;
    replayPosition = 0;
  }

  virtual void setReplayPath(const std::vector<bool> *path) {
    assert(!replayOut && "cannot replay both buffer and path");
    replayPath = path;
    replayPosition = 0;
  }

  virtual const llvm::Module *
  setModule(llvm::Module *module, const ModuleOptions &opts);

  virtual void useSeeds(const std::vector<struct KTest *> *seeds) { 
    usingSeeds = seeds;
  }

  virtual void runFunctionAsMain(llvm::Function *f,
                                 int argc,
                                 char **argv,
                                 char **envp);

  /*** Runtime options ***/
  
  virtual void setHaltExecution(bool value) {
    haltExecution = value;
  }

  virtual void setInhibitForking(bool value) {
    inhibitForking = value;
  }

  /*** State accessor methods ***/

  virtual unsigned getPathStreamID(const ExecutionState &state);

  virtual unsigned getSymbolicPathStreamID(const ExecutionState &state);

  virtual void getConstraintLog(const ExecutionState &state,
                                std::string &res,
                                bool asCVC = false);

  void addConfigConstraint(ExecutionState &state, klee::ref<Expr> condition);

  bool getSymbolicConfig(ExecutionState &state, klee::ref<Expr> cond);

  // No need to set it 'virtual'
  bool getSymbolicConfigSolution(ExecutionState &state, 
                                 klee::ref<Expr> condition,
                                 std::vector< klee::ref<Expr> > offsetVec, 
                                 std::vector< klee::ref<Expr> > &cOffsetVec,
                                 klee::ref<Expr> val1, klee::ref<Expr> val2, bool &benign,
                                 std::vector<SymBlockID_t> &symBlockIDs, 
                                 std::vector<SymThreadID_t> &symThreadIDs, 
                                 SymBlockDim_t &symBlockDim);

  bool dumpOffsetValue(const ExecutionState &state, 
                       klee::ref<Expr> offset);

  bool getConditionSolution(const ExecutionState &state, 
                            klee::ref<Expr> cond,
                            std::vector< 
                            std::pair<std::string,
                            std::vector<unsigned char> > >
                            &res);

  virtual bool getSymbolicSolution(const ExecutionState &state, 
                                   std::vector< 
                                   std::pair<std::string,
                                   std::vector<unsigned char> > >
                                   &res);

  virtual void getCoveredLines(const ExecutionState &state,
                               std::map<const std::string*, std::set<unsigned> > &res);


  // By peng li, for path reduction
  bool fullyExplore(ExecutionState &current, klee::ref<Expr> condition, PRInfo& info);

  Expr::Width getWidthForLLVMType(LLVM_TYPE_Q llvm::Type *type) const;
};

class ExecutorUtil {
  public: 
    static bool isForkInstruction(llvm::Instruction *inst);
    static void copyOutConstraint(ExecutionState &state, bool ignoreCur = false); 
    static void copyBackConstraint(ExecutionState &state); 
    static void copyOutConstraintUnderSymbolic(ExecutionState &state, bool ignoreCur = false);
    static void copyBackConstraintUnderSymbolic(ExecutionState &state);
    static void constructSymConfigEncodedConstraint(ExecutionState &state);
    static void constructSymBlockDimPrecondition(ExecutionState &state);

    static void addConfigConstraint(ExecutionState &state, klee::ref<Expr> condition);
};
  
} // End klee namespace

#endif
