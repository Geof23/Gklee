//===-- Executor.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "Common.h"

#include "Executor.h"
#include "Context.h"
#include "CoreStats.h"
#include "ExternalDispatcher.h"
#include "ImpliedValue.h"
#include "Memory.h"
#include "MemoryManager.h"
#include "PTree.h"
#include "Searcher.h"
#include "SeedInfo.h"
#include "SpecialFunctionHandler.h"
#include "StatsTracker.h"
// #include "../FLA/StringSolver.h"
#include "TimingSolver.h"
#include "UserSearcher.h"
#include "../Solver/SolverStats.h"
#include "klee/ExecutionState.h"
#include "klee/Expr.h"
#include "klee/Interpreter.h"
#include "klee/TimerStatIncrementer.h"
#include "klee/CommandLine.h"
#include "klee/util/Assignment.h"
#include "klee/util/ExprPPrinter.h"
#include "klee/util/ExprUtil.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "klee/Config/Version.h"
#include "klee/Internal/ADT/KTest.h"
#include "klee/Internal/ADT/RNG.h"
#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "klee/Internal/Support/FloatEvaluation.h"
#include "klee/Internal/System/Time.h"

#include "klee/logging.h"

#include "llvm/Attributes.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#if LLVM_VERSION_CODE >= LLVM_VERSION(2, 7)
#include "llvm/LLVMContext.h"
#endif
#include "llvm/Module.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 9)
#include "llvm/System/Process.h"
#else
#include "llvm/Support/Process.h"
#endif
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif
#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
#include "llvm/Analysis/DebugInfo.h"
#else
#include "llvm/DebugInfo.h"
#endif
#include "llvm/Support/raw_os_ostream.h"

#include <cassert>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <dirent.h>

#include <sys/mman.h>

#include <errno.h>
#include <cxxabi.h>
#include <time.h>

using namespace llvm;
using namespace klee;
using namespace Gklee;

namespace runtime {
  cl::opt<bool>
  DumpStatesOnHalt("dump-states-on-halt",
                   cl::init(true));

  cl::opt<bool>
  NoPreferCex("no-prefer-cex",
              cl::init(false));

  cl::opt<bool>
  RandomizeFork("randomize-fork",
                cl::init(false));
 
  cl::opt<bool>
  AllowExternalSymCalls("allow-external-sym-calls",
                        cl::init(false));

  cl::opt<bool>
  DebugPrintInstructions("debug-print-instructions", 
                         cl::desc("Print instructions during execution."));

  cl::opt<bool>
  DebugCheckForImpliedValues("debug-check-for-implied-values");

  cl::opt<bool>
  DebugValidateSolver("debug-validate-solver",
		      cl::init(false));

  cl::opt<bool>
  SuppressExternalWarnings("suppress-external-warnings");

  cl::opt<bool>
  AllExternalWarnings("all-external-warnings");

  cl::opt<bool>
  OnlyOutputStatesCoveringNew("reduce-tests",
                              cl::init(false));

  cl::opt<bool>
  AlwaysOutputSeeds("always-output-seeds",
                              cl::init(true));

  cl::opt<bool>
  UseIndependentSolver("use-independent-solver",
                       cl::init(true),
		       cl::desc("Use constraint independence"));

  cl::opt<bool>
  EmitAllErrors("emit-all-errors",
                cl::init(false),
                cl::desc("Generate tests cases for all errors "
                         "(default=one per (error,instruction) pair)"));

  cl::opt<bool>
  UseCexCache("use-cex-cache",
              cl::init(true),
	      cl::desc("Use counterexample caching"));

  cl::opt<bool>
  UseQueryPCLog("use-query-pc-log",
                cl::init(false));
  
  cl::opt<bool>
  UseSTPQueryPCLog("use-stp-query-pc-log",
                   cl::init(false));

  cl::opt<bool>
  NoExternals("no-externals", 
           cl::desc("Do not allow external functin calls"));

  cl::opt<bool>
  UseCache("use-cache",
	   cl::init(true),
	   cl::desc("Use validity caching"));

  cl::opt<bool>
  OnlyReplaySeeds("only-replay-seeds", 
                  cl::desc("Discard states that do not have a seed."));
 
  cl::opt<bool>
  OnlySeed("only-seed", 
           cl::desc("Stop execution after seeding is done without doing regular search."));
 
  cl::opt<bool>
  AllowSeedExtension("allow-seed-extension", 
                     cl::desc("Allow extra (unbound) values to become symbolic during seeding."));
 
  cl::opt<bool>
  ZeroSeedExtension("zero-seed-extension");
 
  cl::opt<bool>
  AllowSeedTruncation("allow-seed-truncation", 
                      cl::desc("Allow smaller buffers than in seeds."));
 
  cl::opt<bool>
  NamedSeedMatching("named-seed-matching",
                    cl::desc("Use names to match symbolic objects to inputs."));

  cl::opt<double>
  MaxStaticForkPct("max-static-fork-pct", cl::init(1.));
  cl::opt<double>
  MaxStaticSolvePct("max-static-solve-pct", cl::init(1.));
  cl::opt<double>
  MaxStaticCPForkPct("max-static-cpfork-pct", cl::init(1.));
  cl::opt<double>
  MaxStaticCPSolvePct("max-static-cpsolve-pct", cl::init(1.));

  cl::opt<double>
  MaxInstructionTime("max-instruction-time",
                     cl::desc("Only allow a single instruction to take this much time (default=0 (off))"),
                     cl::init(0));
  
  cl::opt<double>
  SeedTime("seed-time",
           cl::desc("Amount of time to dedicate to seeds, before normal search (default=0 (off))"),
           cl::init(0));
  
  cl::opt<double>
  MaxSTPTime("max-stp-time",
             cl::desc("Maximum amount of time for a single query (default=120s)"),
             cl::init(120.0));

  cl::opt<unsigned int>
  StopAfterNInstructions("stop-after-n-instructions",
                         cl::desc("Stop execution after specified number of instructions (0=off)"),
                         cl::init(0));

  cl::opt<unsigned>
  MaxForks("max-forks",
           cl::desc("Only fork this many times (-1=off)"),
           cl::init(~0u));
  
  cl::opt<unsigned>
  MaxDepth("max-depth",
           cl::desc("Only allow this many symbolic branches (0=off)"),
           cl::init(0));
  
  cl::opt<unsigned>
  MaxMemory("max-memory",
            cl::desc("Refuse to fork when more above this about of memory (in MB, 0=off)"),
            cl::init(0));

  cl::opt<bool>
  MaxMemoryInhibit("max-memory-inhibit",
            cl::desc("Inhibit forking at memory cap (vs. random terminate)"),
            cl::init(true));

  cl::opt<bool>
  UseForkedSTP("use-forked-stp", 
                 cl::desc("Run STP in forked process"));

  cl::opt<bool>
  STPOptimizeDivides("stp-optimize-divides", 
                 cl::desc("Optimize constant divides into add/shift/multiplies before passing to STP"),
                 cl::init(true));

  cl::opt<bool>
  PrintCondition("print-cond",
		 cl::desc("Print out the path condition for debugging"),
                 cl::init(false));

  cl::opt<bool>
  UseStats("use-stats",
	   cl::desc("Print out statistics information"),
	   cl::init(true));

  cl::opt<std::string>
  ReducePath("reduce-path", 
	     cl::desc("path reduction"),
	     cl::init(""));

  cl::opt<bool>
  PRUseDep("PR-use-dep", 
	   cl::desc("use dependency analysis when applying path reduction"),
	   cl::init(true));

  cl::opt<bool>
  BCCoverageLevel("bc-cov", 
		  cl::desc("calculate bytecode coverage for the threads"),
		  cl::init(false));
  
  cl::opt<bool>
  PerformTest("generate-perform-tests",
              cl::desc("Generate the performance-defect test cases"),
              cl::init(false));

  cl::opt<bool>
  Trace("trace", 
	cl::desc("Emit a trace file for each path"),
	cl::init(false));

  cl::opt<bool>
  Emacs("emacs",
	cl::desc("Configure output for use with Emacs front end"),
	cl::init(false));

  cl::opt<bool>
  RacePrune("race-prune", 
            cl::desc("Prune the paths not leading to races"), 
            cl::init(false));

  extern cl::opt<bool> ReuseCov;
  extern cl::opt<bool> IgnoreConcurBug;
  extern cl::opt<bool> CheckBC;
  extern cl::opt<bool> CheckMC;
  extern cl::opt<bool> CheckWD;
  extern cl::opt<bool> SimdSchedule;
  extern cl::opt<bool> UnboundConfig;
  extern cl::opt<bool> CheckBarrierRedundant;
  extern cl::opt<bool> UseSymbolicConfig;  
}

static void *theMMap = 0;
static unsigned theMMapSize = 0;

namespace klee {
  RNG theRNG;
}  

//#define CLOCKS_PER_SEC ((clock_t)1000) 

//a little logging helper
std::string getValidityString( klee::Solver::Validity const& v){
  switch( v ){
  case Solver::Unknown:
    return "Solver::Unknown";
    break;
  case Solver::True:
    return "Solver::True";
    break;
  case Solver::False:
    return "Solver::False";
    break;
  }
}

Solver *constructSolverChain(STPSolver *stpSolver,
                             std::string querySMT2LogPath,
                             std::string baseSolverQuerySMT2LogPath,
                             std::string queryPCLogPath,
                             std::string baseSolverQueryPCLogPath) {

  Gklee::Logging::enterFunc( std::string( "Constructing solver" ) , __PRETTY_FUNCTION__ ); 
  Solver *solver = stpSolver;

  if (optionIsSet(queryLoggingOptions,SOLVER_PC))
  {
    solver = createPCLoggingSolver(solver,
                                   baseSolverQueryPCLogPath,
                                   MinQueryTimeToLog);
    klee_message("Logging queries that reach solver in .pc format to %s",baseSolverQueryPCLogPath.c_str());
  }

  if (optionIsSet(queryLoggingOptions,SOLVER_SMTLIB))
  {
    solver = createSMTLIBLoggingSolver(solver,baseSolverQuerySMT2LogPath,
                                       MinQueryTimeToLog);
    klee_message("Logging queries that reach solver in .smt2 format to %s",baseSolverQuerySMT2LogPath.c_str());
  }

  if (UseFastCexSolver)
    solver = createFastCexSolver(solver);

  if (UseCexCache)
    solver = createCexCachingSolver(solver);

  if (UseCache)
    solver = createCachingSolver(solver);

  if (UseIndependentSolver)
    solver = createIndependentSolver(solver);

  if (DebugValidateSolver)
    solver = createValidatingSolver(solver, stpSolver);

  if (optionIsSet(queryLoggingOptions,ALL_PC))
  {
    solver = createPCLoggingSolver(solver,
                                   queryPCLogPath,
                                   MinQueryTimeToLog);
    klee_message("Logging all queries in .pc format to %s",queryPCLogPath.c_str());
  }

  if (optionIsSet(queryLoggingOptions,ALL_SMTLIB))
  {
    solver = createSMTLIBLoggingSolver(solver,querySMT2LogPath,
                                       MinQueryTimeToLog);
    klee_message("Logging all queries in .smt2 format to %s",querySMT2LogPath.c_str());
  }
  Gklee::Logging::exitFunc();
  return solver;
}


Executor::Executor(const InterpreterOptions &opts,
                   InterpreterHandler *ih) 
  : Interpreter(opts),
    kmodule(0),
    interpreterHandler(ih),
    searcher(0),
    is_GPU_mode(false),
    accumStore(false),
    atomicRes(0),
    kernelFunc(0),
    externalDispatcher(new ExternalDispatcher()),
    statsTracker(0),
    pathWriter(0),
    symPathWriter(0),
    specialFunctionHandler(0),
    processTree(0),
    replayOut(0),
    replayPath(0),    
    usingSeeds(0),
    atMemoryLimit(false),
    inhibitForking(false),
    haltExecution(false),
    ivcEnabled(false),
    stpTimeout(MaxSTPTime != 0 && MaxInstructionTime != 0
      ? std::min(MaxSTPTime,MaxInstructionTime)
      : std::max(MaxSTPTime,MaxInstructionTime)), 
    bcBINum(0),
    mcBINum(0),
    wdBINum(0),
    bcBISum(0),
    mcBISum(0),
    wdBISum(0),
    bcWarpNum(0),
    mcWarpNum(0), 
    wdWarpNum(0),
    bcWarpSum(0),
    mcWarpSum(0), 
    wdWarpSum(0), 
    symBC(false),
    symMC(true),
    symWD(false),
    symVMiss(false),
    symRace(false)
{

  Gklee::Logging::enterFunc( std::string( "Create STPSolver, postDomtree, memManager" ), __PRETTY_FUNCTION__ );
  concreteTotalTime = symTotalTime = 0.0f;
  STPSolver *stpSolver = new STPSolver(UseForkedSTP, STPOptimizeDivides);
  Solver *solver =
    constructSolverChain(stpSolver,
                         interpreterHandler->getOutputFilename(ALL_QUERIES_SMT2_FILE_NAME),
                         interpreterHandler->getOutputFilename(SOLVER_QUERIES_SMT2_FILE_NAME),
                         interpreterHandler->getOutputFilename(ALL_QUERIES_PC_FILE_NAME),
                         interpreterHandler->getOutputFilename(SOLVER_QUERIES_PC_FILE_NAME));
  this->solver = new TimingSolver(solver, stpSolver);
  postDominator = (llvm::PostDominatorTree*)llvm::createPostDomTree();
  memory = new MemoryManager();
  Gklee::Logging::exitFunc();
}


const Module *Executor::setModule(llvm::Module *module, 
                                  const ModuleOptions &opts) {
  Gklee::Logging::enterFunc( module->getModuleIdentifier(), __PRETTY_FUNCTION__ ); //*module , __PRETTY_FUNCTION__ );
  assert(!kmodule && module && "can only register one module"); // XXX gross
 if (GPUConfig::verbose > 0) {
   std::cout << "Entered setModule with " << module->getModuleIdentifier() << std::endl;
 }
  
  kmodule = new KModule(module);

  // Initialize the context.
  DataLayout *TD = kmodule->targetData;
  Context::initialize(TD->isLittleEndian(),
                      (Expr::Width) TD->getPointerSizeInBits());

  specialFunctionHandler = new SpecialFunctionHandler(*this);

  specialFunctionHandler->prepare();
  kmodule->prepare(opts, interpreterHandler);
  specialFunctionHandler->bind();

  if (StatsTracker::useStatistics()) {
    statsTracker = 
    new StatsTracker(*this,
                     interpreterHandler->getOutputFilename("assembly.ll"),
                     userSearcherRequiresMD2U());
  }
  Gklee::Logging::exitFunc();
  return module;
}

Executor::~Executor() {
  Gklee::Logging::enterFunc( std::string( "Deleting Executor" ), __PRETTY_FUNCTION__ );
  delete memory;
  delete externalDispatcher;
  if (processTree)
    delete processTree;
  if (specialFunctionHandler)
    delete specialFunctionHandler;
  if (statsTracker)
    delete statsTracker;
  // by Guodong: don't delete your solver twice!
  //delete solver;
  if (postDominator) 
    delete postDominator;
  delete kmodule;
  Gklee::Logging::exitFunc();
}

/***/

// enum dumpType {full, readSet, writeSet, memoryState};

// void
// //dumpInfo(dumpType dt, std::string stepInfo, ExecutionState &state){
//   return; //fix me
//   std::cerr << "########################################" << std::endl;
//   std::cerr << stepInfo << std::endl;
//   std::cerr << "new PC, file: " << state.getPC()->info->file << 
//     " line: " << state.getPC()->info->line << std::endl;
//   switch( dt ){
//   case memoryState:
//     std::cerr << "current memory state:" << std::endl;
//     state.addressSpace.dump( 0xF ); //masked: 1xxx = cpumem x1xx = device xx1x = shared xxx1 = local (see HierAddressSpace.cpp)
//     break;
//   case writeSet:
//   case readSet:
//     std::cerr << "current r/w sets: " << std::endl;
//     state.addressSpace.dumpInstAccessSet();
//     break;
//   case full:
//     std::cerr << "full state info: " << std::endl;
//     std::cerr << "current r/w sets: " << std::endl;
//     state.addressSpace.dumpInstAccessSet();
//     std::cerr << "current memory state:" << std::endl;
//     state.addressSpace.dump( 0xF ); //masked: 1xxx = cpumem x1xx = device xx1x = shared (see HierAddressSpace.cpp)
//     if( state.paraTreeSets.size() > 0 ){
//       auto currTree = state.getCurrentParaTree();
//       std::cerr << "current PFT: " << std::endl;
//       currTree.dumpParaTree();
//       auto currNode = currTree.getCurrentNode();
//       if( currNode != NULL ){
// 	std::cerr << "current Flow: " << std::endl;
// 	currNode->dumpParaTreeNode();
// 	std::cerr << "current flow condition: " << std::endl;
// 	currNode->inheritCond->dump();
//       }
//     }
//     break;
//   }
//  Gklee::Logging::exitFunc();
//  }


///
/// This function adds a memory object to the state
///
MemoryObject * Executor::addExternalObject(ExecutionState &state, 
                                           void *addr, unsigned size, 
                                           bool isReadOnly) {
  Gklee::Logging::enterFunc< std::string >( std::string( "adding mem obj to state" ) , __PRETTY_FUNCTION__ ); // std::string("obj: ") + 
			     // std::to_string( (size_t) addr) +
			     // ":" + std::to_string(size) + "RO?:" + 
			     // std::to_string( isReadOnly ));
 // if (GPUConfig::verbose > 0) {
 //   std::cout << "addExeternalObject, addr: " << hex << addr << " of size: " << 
 // 	     size << std::endl;
 // }
  MemoryObject *mo = memory->allocateFixed((uint64_t) (unsigned long) addr, 
                                           size, state.tinfo.is_GPU_mode, 0);
  ObjectState *os = bindObjectInState(state, mo, false);
  for(unsigned i = 0; i < size; i++)
    os->write8(i, ((uint8_t*)addr)[i]);
  if(isReadOnly)
    os->setReadOnly(true);
  //  Gklee::Logging::outItem("
  Gklee::Logging::exitFunc();
  return mo;
}
///
/// This instruction advances the state's PC
///
void Executor::stepInstruction(ExecutionState &state) {
  Gklee::Logging::enterFunc( std::string("current PC:") +
			     state.getPC()->info->file + ":" +
    			     std::to_string(state.getPC()->info->line),
			     __PRETTY_FUNCTION__ ); 
  if (DebugPrintInstructions) {
    printFileLine(state, state.getPC());
    std::cerr << std::setw(10) << stats::instructions << " ";
    llvm::errs() << *(state.getPC()->inst) << "\n";
  }

  if (!RacePrune && statsTracker)
    statsTracker->stepInstruction(state);

  ++stats::instructions;

  state.setPrevPC(state.getPC());
  state.incPC();

  if (stats::instructions==StopAfterNInstructions)
    haltExecution = true;
  // Gklee::Logging::outItem( std::string( "New PC" ), std::string("") + state.getPC()->info->file  + ":" +
  // 			   std::to_string( state.getPC()->info->line )); 
  Gklee::Logging::exitFunc();
}

///
/// This switches state from source to dst basic block, setting
/// the incomingBBIndex correctly for PHI node instructions
void Executor::transferToBasicBlock(BasicBlock *dst, BasicBlock *src, 
                                    ExecutionState &state) {
  // Note that in general phi nodes can reuse phi values from the same
  // block but the incoming value is the eval() result *before* the
  // execution of any phi nodes. this is pathological and doesn't
  // really seem to occur, but just in case we run the PhiCleanerPass
  // which makes sure this cannot happen and so it is safe to just
  // eval things in order. The PhiCleanerPass also makes sure that all
  // incoming blocks have the same order for each PHINode so we only
  // have to compute the index once.
  //
  // With that done we simply set an index in the state so that PHI
  // instructions know which argument to eval, set the pc, and continue.
  
  // if (GPUConfig::verbose > 0) {
  //   std::cout << "Entered " << __FUNCTION__ << 
  // }
  // XXX this lookup has to go ?
  Gklee::Logging::enterFunc( std::string("dst name:") +
			     dst->getName().str() + ", src: " + src->getName().str(),
			     __PRETTY_FUNCTION__ );
  KFunction *kf = state.getCurStack().back().kf;
  unsigned entry = kf->basicBlockEntry[dst];
  state.setPC(&kf->instructions[entry]);

  if (state.getPC()->inst->getOpcode() == Instruction::PHI) {
    PHINode *first = static_cast<PHINode*>(state.getPC()->inst);
    auto newBBI = first->getBasicBlockIndex(src);
    Gklee::Logging::outItem( std::to_string( newBBI ), "phi node, incomingBBIndex set" );
    state.incomingBBIndex[state.tinfo.get_cur_tid()]  = newBBI;
  }
  Gklee::Logging::exitFunc();
}

///
///  
///
void Executor::branch(ExecutionState &state, 
                      const std::vector< klee::ref<Expr> > &conditions,
                      std::vector<ExecutionState*> &result) {
  // Gklee::
  Gklee::Logging::enterFunc( conditions[0] , __PRETTY_FUNCTION__ );
  TimerStatIncrementer timer(stats::forkTime);
  unsigned N = conditions.size();
  assert(N);

  stats::forks += N-1;

  // XXX do proper balance or keep random?
  result.push_back(&state);
  for (unsigned i=1; i<N; ++i) {
    ExecutionState *es = result[theRNG.getInt32() % i];
    ExecutionState *ns = es->branch();
    addedStates.insert(ns);
    result.push_back(ns);
    es->ptreeNode->data = 0;
    std::pair<PTree::Node*,PTree::Node*> res = 
      processTree->split(es->ptreeNode, ns, es);
    ns->ptreeNode = res.first;
    es->ptreeNode = res.second;
  }

  // If necessary redistribute seeds to match conditions, killing
  // states if necessary due to OnlyReplaySeeds (inefficient but
  // simple).
  
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    std::vector<SeedInfo> seeds = it->second;
    seedMap.erase(it);

    // Assume each seed only satisfies one condition (necessarily true
    // when conditions are mutually exclusive and their conjunction is
    // a tautology).
    for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
           siie = seeds.end(); siit != siie; ++siit) {
      unsigned i;
      ExecutorUtil::copyOutConstraintUnderSymbolic(state);
      for (i=0; i<N; ++i) {
        klee::ref<ConstantExpr> res;
        bool success = 
          solver->getValue(state, siit->assignment.evaluate(conditions[i]), 
                           res);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue())
          break;
      }
      ExecutorUtil::copyBackConstraintUnderSymbolic(state);
      
      // If we didn't find a satisfying condition randomly pick one
      // (the seed will be patched).
      if (i==N)
        i = theRNG.getInt32() % N;

      seedMap[result[i]].push_back(*siit);
    }

    if (OnlyReplaySeeds) {
      for (unsigned i=0; i<N; ++i) {
        if (!seedMap.count(result[i])) {
          terminateState(*result[i]);
          result[i] = NULL;
        }
      } 
    }
  }

  for (unsigned i=0; i<N; ++i)
    if (result[i]){
      addConstraint(*result[i], conditions[i]);
      Gklee::Logging::outItem( conditions[ i ],
			       "state " + std::to_string( i ) + " cond" );
			       
    }
  Gklee::Logging::exitFunc();
  // Gklee::
} 

// True: condition is totally or partially related to built-in variables
// False: condition is not related to built-in variables  
bool Executor::identifyConditionType(ExecutionState &state, klee::ref<Expr> &cond, 
                                     bool &relatedToSym, bool &accum) {
  //std::cout << "condition in identifyConditionType: " << std::endl;
  //cond->dump();

  // Accumulative
  Gklee::Logging::enterFunc( cond , __PRETTY_FUNCTION__ );
  if (cond->accum)
    accum = true;

  std::vector<const Array*> arrayVec;
  findSymbolicObjects(cond, arrayVec);
  if (arrayVec.begin() == arrayVec.end()){
    Gklee::Logging::exitFunc();
    return false;
  }
  bool relatedBuiltin = false;
  std::vector<const Array*>::iterator vi = arrayVec.begin();
  for (; vi != arrayVec.end(); vi++) {
    size_t foundB;
    size_t foundT;
    std::string bidArray = "bid_arr";
    std::string tidArray = "tid_arr";
    foundB = ((*vi)->name).find(bidArray);
    foundT = ((*vi)->name).find(tidArray);

    // TDC 
    if (foundB != string::npos || foundT != string::npos) {
      relatedBuiltin = true;
    }
    // SYM 
    for (unsigned i = 0; i < state.symInputVec.size(); i++) {
      size_t foundSym = ((*vi)->name).find(state.symInputVec[i]);
      if (foundSym != string::npos)
        relatedToSym = true;
    }
  }

  // Roughtly check if the express contains the "const_arr"
  if (!relatedToSym) {
    std::stringstream ss;
    cond->print(ss);
    if (ss.str().find("const_arr") != std::string::npos)
      relatedToSym = true;
  }
  Gklee::Logging::outItem( std::string("relatedBuiltin:") +
			   std::to_string( relatedBuiltin ) + " relatedToSym:" +
			   std::to_string( relatedToSym ) + " accum:" +
			   std::to_string( accum ),
			   "types identified" );
  Gklee::Logging::exitFunc();
  return relatedBuiltin; 
}
 
static unsigned findUnusedThreadSlot(std::vector<CorrespondTid> &cTidSets) {
  Gklee::Logging::enterFunc( std::string( "|cTidSets|:") +
			     std::to_string( cTidSets.size()),
			     __PRETTY_FUNCTION__ );
  bool findUnused = false;
  unsigned i = 0;
  for (; i < cTidSets.size(); i++) {
    if (!cTidSets[i].slotUsed) {
      findUnused = true;
      break;
    }
  }   
  assert(findUnused && "Unused thread slot not found!\n");
  Gklee::Logging::outItem( std::to_string( i ), "returnResult" );
  Gklee::Logging::exitFunc();
  return i;
}

void Executor::evaluateConstraintAsNewFlow(ExecutionState &state, ParaTree &pTree,
                                           klee::ref<Expr> &cond, bool flowCreated) {
  
  Gklee::Logging::enterFunc( cond , __PRETTY_FUNCTION__ );  
  unsigned cur_bid = state.tinfo.get_cur_bid();
  unsigned cur_tid = state.tinfo.get_cur_tid();
  Gklee::Logging::outItem( std::to_string( cur_bid ) + ":" +
			   std::to_string( cur_tid ),
			   "bid:tid" );
  if (flowCreated) {
    unsigned idle_tid = findUnusedThreadSlot(state.cTidSets);
    GKLEE_INFO << "create new parametric flow: " << idle_tid 
               << std::endl;
    Gklee::Logging::outItem( std::to_string( idle_tid ), "new pflow" );
    state.tinfo.symExecuteSet.push_back(idle_tid);
    ParaConfig config(cur_bid, idle_tid, cond, 0, 0);
    pTree.updateCurrentNodeOnNewConfig(config, TDC);
    state.cTidSets[idle_tid].slotUsed = true;
  } else {
    // Only explore the 'false' flow ... 
    GKLEE_INFO << "keep using the current flow: " 
               << cur_tid << std::endl;
    Gklee::Logging::outItem( std::string( "" ) , "keeping current flow" );
    ParaConfig config(cur_bid, cur_tid, cond, 0, 0);
    pTree.updateCurrentNodeOnNewConfig(config, TDC);
    state.tinfo.sym_tdc_eval = 2;
  }
  Gklee::Logging::exitFunc();
  
}

void Executor::evaluateConstraintAsNewFlowUnderRacePrune(ExecutionState &state, ParaTree &pTree,
                                                         klee::ref<Expr> &cond, bool flowCreated,
                                                         BranchInst *bi) {
  
  Gklee::Logging::enterFunc( cond , __PRETTY_FUNCTION__ );
  unsigned cur_bid = state.tinfo.get_cur_bid();
  unsigned cur_tid = state.tinfo.get_cur_tid();

  if (flowCreated) {
    unsigned idle_tid = findUnusedThreadSlot(state.cTidSets);
    GKLEE_INFO << "create new parametric flow: " << idle_tid 
               << std::endl;
    Gklee::Logging::outItem( std::to_string( idle_tid ), "new pflow" );
    state.tinfo.symExecuteSet.push_back(idle_tid);
    ParaConfig config(cur_bid, idle_tid, cond, 0, 0);
    pTree.updateCurrentNodeOnNewConfig(config, TDC);
    state.cTidSets[idle_tid].slotUsed = true;
    if (bi->getMetadata("br-S-G")
         || bi->getMetadata("br-G-G")) {
      // will contribute to the race detection across BIs
      if (bi->getMetadata("br-G-G"))
        std::cout << "br-G-G" << std::endl;
      else 
        std::cout << "br-S-G" << std::endl;

      state.cTidSets[idle_tid].keep = true;
      Logging::outItem< std::string >( std::to_string( idle_tid ),
				       "new flow" );
    } else if (bi->getMetadata("br-S-S")) {
      if (state.cTidSets[cur_tid].keep){
        state.cTidSets[idle_tid].keep = true;
	Logging::outItem< std::string >( std::to_string( idle_tid ),
				       "new flow" );
      }
    }
  } else {
    // Only explore the 'false' flow ... 
    GKLEE_INFO << "keep using the current flow: " 
               << cur_tid << std::endl;
    Gklee::Logging::outItem( std::to_string( cur_tid ), 
			     "keeping current flow, PRUNED ID" );
    ParaConfig config(cur_bid, cur_tid, cond, 0, 0);
    pTree.updateCurrentNodeOnNewConfig(config, TDC);
    state.tinfo.sym_tdc_eval = 2;
  }
  Gklee::Logging::exitFunc();
  
}

Executor::StatePair 
Executor::fork(ExecutionState &current, klee::ref<Expr> condition, bool isInternal) {
   //TODO flow experiment
  
  Gklee::Logging::enterFunc( condition , __PRETTY_FUNCTION__ ); 
  Solver::Validity res;
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&current);
  bool isSeeding = it != seedMap.end();

  if (!isSeeding && !isa<ConstantExpr>(condition) && 
      (MaxStaticForkPct!=1. || MaxStaticSolvePct != 1. ||
       MaxStaticCPForkPct!=1. || MaxStaticCPSolvePct != 1.) &&
      statsTracker->elapsed() > 60.) {

    StatisticManager &sm = *theStatisticManager;
    CallPathNode *cpn = current.getCurStack().back().callPathNode;
    //I don't understand the statistics at this point, but 
    //this solves the current constraint and adds the solution
    //to the constraint
    if ((MaxStaticForkPct<1. &&
         sm.getIndexedValue(stats::forks, sm.getIndex()) > 
         stats::forks*MaxStaticForkPct) ||
        (MaxStaticCPForkPct<1. &&
         cpn && (cpn->statistics.getValue(stats::forks) > 
                 stats::forks*MaxStaticCPForkPct)) ||
        (MaxStaticSolvePct<1 &&
         sm.getIndexedValue(stats::solverTime, sm.getIndex()) > 
         stats::solverTime*MaxStaticSolvePct) ||
        (MaxStaticCPForkPct<1. &&
         cpn && (cpn->statistics.getValue(stats::solverTime) > 
                 stats::solverTime*MaxStaticCPSolvePct))) {
      klee::ref<ConstantExpr> value; 
      bool success = solver->getValue(current, condition, value);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      addConstraint(current, EqExpr::create(value, condition));
      Gklee::Logging::outItem< klee::ref<klee::Expr> >( value , "adding constraint to current state" );
      condition = value;
    }
  }

  double timeout = stpTimeout;
  if (isSeeding)
    timeout *= it->second.size();
  solver->setTimeout(timeout);

  bool success = false;
  bool isSymCond = false;
  bool isAccumCond = false;
  bool isTDCCond = false;
  if (UseSymbolicConfig) {
    isTDCCond = identifyConditionType(current, condition, 
                                      isSymCond, isAccumCond);
    Gklee::Logging::outItem( std::to_string( isTDCCond ), "isTDCCond" );
    Gklee::Logging::outItem( condition, "condition" );
    if (current.tinfo.is_GPU_mode) {
      if (isSymCond) {
        // SYM, Accumulative or other kinds of conditionals
        ExecutorUtil::copyOutConstraint(current, !isInternal);
        success = solver->evaluate(current, condition, res);
	Logging::outItem( getValidityString( res ), "condition evaluation" );
        ExecutorUtil::copyBackConstraint(current);
        if (RacePrune) { 
          if (!isInternal
	      && success 
	      && res == Solver::Unknown) {
            if (current.brMeta.meta == GG) {
              res = Solver::Unknown;
              current.cTidSets[current.tinfo.get_cur_tid()].keep = true;
            } else if (current.brMeta.meta == SS) {
              res = Solver::True;
            } else if (current.brMeta.meta == GS) {
              res = Solver::True;
              current.cTidSets[current.tinfo.get_cur_tid()].keep = true;
            } else if (current.brMeta.meta == SG) { 
              res = Solver::False;
              current.cTidSets[current.tinfo.get_cur_tid()].keep = true;
            }else{
	    }
            current.brMeta.meta = NA;
          } else {
            if (current.tinfo.is_Atomic_op > 0) {
              if (current.tinfo.is_Atomic_op == 1)
                res = Solver::True;
              else 
                res = Solver::False;
            }
          }
	  if( current.cTidSets[current.tinfo.get_cur_tid()].keep == false )
	    Logging::outItem( std::to_string( current.tinfo.get_cur_tid() ),
			      "keep == false" );
	  else
	    Logging::outItem( std::to_string( current.tinfo.get_cur_tid() ),
			      "keep == true" );
        }
      } else { 
        if (isTDCCond) {
          // TDC conditionals 
          success = true;
          res = Solver::Unknown;
	  Logging::outItem< std::string >( "is tdc conditional", "" );
        } else if (isAccumCond) {
	  Logging::outItem< std::string >( "accum condition", "" );
          bool ignoreCurrent = !isInternal;
          ExecutorUtil::copyOutConstraint(current, ignoreCurrent);
          success = solver->evaluate(current, condition, res);
          ExecutorUtil::copyBackConstraint(current);
        } else {
          success = solver->evaluate(current, condition, res);
        }
      }
    } else {
      success = solver->evaluate(current, condition, res);
    }
  } else {
    success = solver->evaluate(current, condition, res);
  }

  Logging::outItem( getValidityString( res ), "condition evaluation" );
  solver->setTimeout(0);
  if (!success) {
    current.setPC(current.getPrevPC());
    terminateStateEarly(current, "query timed out");
    Gklee::Logging::exitFunc();
    
     //TODO flow experiment
    return StatePair(0, 0);
  }

  if (!isSeeding) {
    if (replayPath && !isInternal) {

      // std::cerr << "choice 10\n";
      assert(replayPosition<replayPath->size() &&
             "ran out of branches in replay path mode");
      bool branch = (*replayPath)[replayPosition++];
      
      if (res==Solver::True) {
        assert(branch && "hit invalid branch in replay path mode");
      } else if (res==Solver::False) {
        assert(!branch && "hit invalid branch in replay path mode");
      } else {
        // add constraints
        if(branch) {
          res = Solver::True;
          addConstraint(current, condition);
        } else  {
          res = Solver::False;
          addConstraint(current, Expr::createIsZero(condition));
        }
      }
    } else if (res==Solver::Unknown) {
      assert(!replayOut && "in replay mode, only one branch can be true.");
     
      // std::cerr << "condition with unknown value \n";
 
      if ((MaxMemoryInhibit && atMemoryLimit) || 
          current.forkDisabled ||
          inhibitForking || 
          (MaxForks!=~0u && stats::forks >= MaxForks)) {

	if (MaxMemoryInhibit && atMemoryLimit)
	  klee_warning_once(0, "skipping fork (memory cap exceeded)");
	else if (current.forkDisabled)
	  klee_warning_once(0, "skipping fork (fork disabled on current path)");
	else if (inhibitForking)
	  klee_warning_once(0, "skipping fork (fork disabled globally)");
	else 
	  klee_warning_once(0, "skipping fork (max-forks reached)");

        TimerStatIncrementer timer(stats::forkTime);
        if (theRNG.getBool()) {
          addConstraint(current, condition);
          res = Solver::True;
        } else {
          addConstraint(current, Expr::createIsZero(condition));
          res = Solver::False;
        }
      }
    }
  }

  // Fix branch in only-replay-seed mode, if we don't have both true
  // and false seeds.
  if (isSeeding && 
      (current.forkDisabled || OnlyReplaySeeds) && 
      res == Solver::Unknown) {

    bool trueSeed=false, falseSeed=false;
    // Is seed extension still ok here?
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      klee::ref<ConstantExpr> res;
      bool success = 
        solver->getValue(current, siit->assignment.evaluate(condition), res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res->isTrue()) {
        trueSeed = true;
      } else {
        falseSeed = true;
      }
      if (trueSeed && falseSeed)
        break;
    }
    if (!(trueSeed && falseSeed)) {
      assert(trueSeed || falseSeed);
      
      res = trueSeed ? Solver::True : Solver::False;
      addConstraint(current, trueSeed ? condition : Expr::createIsZero(condition));
    }
  }

  // XXX - even if the constraint is provable one way or the other we
  // can probably benefit by adding this constraint and allowing it to
  // reduce the other constraints. For example, if we do a binary
  // search on a particular value, and then see a comparison against
  // the value it has been fixed at, we should take this as a nice
  // hint to just use the single constraint instead of all the binary
  // search ones. If that makes sense.
  //std::cout << "test total: " << std::endl;
  //condition->dump();
  if (res==Solver::True) {
    //std::cerr << "true\n";
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "1";
      }
    }

    if (UseSymbolicConfig
         && current.tinfo.is_GPU_mode
           && !isInternal
             && isSymCond) {
      // If the current node's cond type is non-TDC, and 
      // it's evaluated to be TRUE, then reset the condition
      // of this node to TRUE 
      current.getCurrentParaTree().resetNonTDCNodeCond();
    }
    Gklee::Logging::exitFunc();
    
     //TODO flow experiment
    return StatePair(&current, 0);
  } else if (res==Solver::False) {
    //std::cerr << "false\n";
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "0";
      }
    }

    if (UseSymbolicConfig
         && current.tinfo.is_GPU_mode
           && !isInternal
             && isSymCond) {
      // If the current node's cond type is non-TDC, and 
      // it's evaluated to be FALSE, then reset the condition
      // of this node to FALSE 
      Logging::outItem< std::string >( "resetting node's condition to false", "" );
      current.getCurrentParaTree().resetNonTDCNodeCond();
    }
    Gklee::Logging::exitFunc();
    
     //TODO flow experiment
    return StatePair(0, &current);
  } else {  // both branches are possible
    //std::cout << "condition: " << std::endl;
    //condition->dump();
    //std::cout << "unknown" << std::endl;
    // TDC is evaluated here 
    if (UseSymbolicConfig 
         && current.tinfo.is_GPU_mode 
           && current.tinfo.sym_tdc_eval) {
      if (current.tinfo.sym_tdc_eval == 1) {
	Logging::outItem< std::string >( "returning only true branch", "" );
	Gklee::Logging::exitFunc();
	
	 //TODO flow experiment
        return StatePair(&current, 0); 
      } else {
	Logging::outItem< std::string >( "returning only false branch", "" );
	Gklee::Logging::exitFunc();
	
	 //TODO flow experiment
        return StatePair(0, &current); 
      }
    }

    if (UseSymbolicConfig) {
      if (current.tinfo.is_GPU_mode
           && !PR_info.symFullyExplore(current, bc_cov_monitor.getCovInfo(current.getKernelNum()))) {
        std::cout << "Explore only one branch (symbolic)!" << std::endl;
        // only explore the left (true) branch
	Logging::outItem< std::string >( "only exploring true branch", "" );
	Gklee::Logging::exitFunc();
	
	 //TODO flow experiment
        return StatePair(&current, 0);
      }
    } else {
      // by Guodong: path reduction
      if (current.tinfo.is_GPU_mode && 
          !PR_info.fullyExplore(current, bc_cov_monitor.getCovInfo(current.getKernelNum()))) {
        // only explore the left (true) branch
        addConstraint(current, condition);
	Logging::outItem< std::string >( "only true branch being explored", "" );
	Gklee::Logging::exitFunc();
	
	 //TODO flow experiment
        return StatePair(&current, 0);
      }
    }
    
    TimerStatIncrementer timer(stats::forkTime);
    ExecutionState *falseState, *trueState = &current;

    ++stats::forks;

    falseState = trueState->branch();
    falseState->forkStateBINum = falseState->BINum;  

    addedStates.insert(falseState);

    if (RandomizeFork && theRNG.getBool())
      std::swap(trueState, falseState);

    if (it != seedMap.end()) {
      std::vector<SeedInfo> seeds = it->second;
      it->second.clear();
      std::vector<SeedInfo> &trueSeeds = seedMap[trueState];
      std::vector<SeedInfo> &falseSeeds = seedMap[falseState];
      for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
             siie = seeds.end(); siit != siie; ++siit) {
        klee::ref<ConstantExpr> res;
        bool success = 
          solver->getValue(current, siit->assignment.evaluate(condition), res);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue()) {
          trueSeeds.push_back(*siit);
        } else {
          falseSeeds.push_back(*siit);
        }
      }
      
      bool swapInfo = false;
      if (trueSeeds.empty()) {
        if (&current == trueState) swapInfo = true;
        seedMap.erase(trueState);
      }
      if (falseSeeds.empty()) {
        if (&current == falseState) swapInfo = true;
        seedMap.erase(falseState);
      }
      if (swapInfo) {
        std::swap(trueState->coveredNew, falseState->coveredNew);
        std::swap(trueState->coveredLines, falseState->coveredLines);
      }
    }

    current.ptreeNode->data = 0;
    std::pair<PTree::Node*, PTree::Node*> res =
      processTree->split(current.ptreeNode, falseState, trueState);
    falseState->ptreeNode = res.first;
    trueState->ptreeNode = res.second;

    if (!isInternal) {
      if (pathWriter) {
        falseState->pathOS = pathWriter->open(current.pathOS);
        trueState->pathOS << "1";
        falseState->pathOS << "0";
      }      
      if (symPathWriter) {
        falseState->symPathOS = symPathWriter->open(current.symPathOS);
        trueState->symPathOS << "1";
        falseState->symPathOS << "0";
      }
    }
  
    if (UseSymbolicConfig) {
      if (falseState->tinfo.is_GPU_mode) {
        // update the Parametric flow tree 
        // for newly generated state *falseState
        if (!isInternal) {
          ParaTree &paraTree = falseState->getCurrentParaTree();
          paraTree.negateNonTDCNodeCond();
        } 
      }
    } else {
      addConstraint(*trueState, condition);
      addConstraint(*falseState, Expr::createIsZero(condition));
    }

    // Kinda gross, do we even really still want this option?
    if (MaxDepth && MaxDepth<=trueState->depth) {
      terminateStateEarly(*trueState, "max-depth exceeded");
      terminateStateEarly(*falseState, "max-depth exceeded");
      Gklee::Logging::exitFunc();
      
       //TODO flow experiment
      return StatePair(0, 0);
    }
    Gklee::Logging::exitFunc();
    
     //TODO flow experiment
    return StatePair(trueState, falseState);
  }
  Gklee::Logging::exitFunc();
  
   //TODO flow experiment
}

void Executor::addConstraint(ExecutionState &state, klee::ref<Expr> condition) {
  Gklee::Logging::enterFunc( condition , __PRETTY_FUNCTION__ ); 
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(condition)) {
    assert(CE->isTrue() && "attempt to add invalid constraint");
    Gklee::Logging::exitFunc();
    return;
  }

  // Check to see if this constraint violates seeds.
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    bool warn = false;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      bool res;
      bool success = 
        solver->mustBeFalse(state, siit->assignment.evaluate(condition), res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        siit->patchSeed(state, condition, solver);
        warn = true;
      }
    }
    if (warn)
      klee_warning("seeds patched for violating constraint"); 
  }

  state.addConstraint(condition);
  if (ivcEnabled)
    doImpliedValueConcretization(state, condition, 
                                 ConstantExpr::alloc(1, Expr::Bool));
  Gklee::Logging::exitFunc();
}

const Cell& Executor::evalSharedMemory(ExecutionState &state, klee::ref<Expr> &pointer, 
                                       unsigned index) {
  Gklee::Logging::enterFunc( pointer , __PRETTY_FUNCTION__ );  
  bool findShared = false;
  std::pair<SharedMemoryObject*, MemoryObject*> tmpPair;
      
  if (state.tinfo.is_GPU_mode) {
    if (UseSymbolicConfig)
      ExecutorUtil::copyOutConstraint(state);

    std::vector< std::pair<SharedMemoryObject*, MemoryObject*> >::const_iterator vit;
    for (vit = sharedObjects.begin(); vit != sharedObjects.end(); ++vit) {
      if (vit->first->kernelNum == state.getKernelNum()
           && vit->second->getBTId() == state.tinfo.get_cur_bid()) {
        Solver::Validity res;
        Expr::Width width = vit->first->sharedGLMO->getBaseExpr()->getWidth(); 
        klee::ref<Expr> pointerZExt = ZExtExpr::create(pointer, width);
        klee::ref<Expr> geCond = UgeExpr::create(pointerZExt, vit->first->sharedGLMO->getBaseExpr());
        
        // ensure that (address) >= (base addr)
        solver->evaluate(state, geCond, res);
        if (res == Solver::True) {
          klee::ref<Expr> boundCheckCond = vit->first->sharedGLMO->getBoundsCheckPointer(pointerZExt);
          // ensure that (address - base addr) < size
          solver->evaluate(state, boundCheckCond, res);
          if (res == Solver::True) {
            tmpPair = *vit;
            findShared = true;
            break;
          }
        }
      }
    }

    if (UseSymbolicConfig)
      ExecutorUtil::copyBackConstraint(state);
  }

  if (!findShared) {
    // the corresponding base addr not found
    Gklee::Logging::outItem( kmodule->constantTable[index].value , "shared not found, returning" );
    Gklee::Logging::exitFunc();
    return kmodule->constantTable[index];
  } else {
    // base addr found, the corresponding share address is found too..
    klee::ref<Expr> shareAddr = AddExpr::create(tmpPair.second->getBaseExpr(), 
                                          tmpPair.first->sharedGLMO->getOffsetExpr(pointer));
    shareAddr->ctype = GPUConfig::SHARED;
    tmpPair.first->glCell->value = shareAddr; 
    Gklee::Logging::outItem( shareAddr , "found shared, addr:" );
    Gklee::Logging::exitFunc();
    return *(tmpPair.first->glCell);
  }
}
 
const Cell& Executor::eval(KInstruction *ki, unsigned index, 
                           ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "Evaluating operand" , __PRETTY_FUNCTION__ );  
  assert(index < ki->inst->getNumOperands());
  int vnumber = ki->operands[index];

  assert(vnumber != -1 &&
         "Invalid operand to eval(), not a value or constant!");

  // Determine if this is a constant or not.
  if (vnumber < 0) {
    unsigned index = -vnumber - 2;
    klee::ref<Expr> addr = (kmodule->constantTable[index]).value;
    const Cell& esm = evalSharedMemory(state, addr, index);
    Gklee::Logging::outItem( esm.value , "result" );
    //Gklee::Logging::outItem( index).value , "result", evalSharedMemory(state, addr );
    Gklee::Logging::exitFunc();
    return esm;
    //return evalSharedMemory(state, addr, index);
  } else {
    unsigned index = vnumber;
    StackFrame &sf = state.getCurStack().back();
    Gklee::Logging::outItem( sf.locals[index].value , "result" );
    Gklee::Logging::exitFunc();
    return sf.locals[index];
  }
}

void Executor::bindLocal(KInstruction *target, ExecutionState &state, 
                         klee::ref<Expr> value) {
  Gklee::Logging::enterFunc( value , __PRETTY_FUNCTION__ );  
  getDestCell(state, target).value = value;
  Gklee::Logging::exitFunc();
}

void Executor::bindArgument(KFunction *kf, unsigned index, 
                            ExecutionState &state, klee::ref<Expr> value) {
  Gklee::Logging::enterFunc( value , __PRETTY_FUNCTION__ );  
  getArgumentCell(state, kf, index).value = value;
  Gklee::Logging::exitFunc();
}

klee::ref<Expr> Executor::toUnique(ExecutionState &state, 
                             klee::ref<Expr> &e) {
  Gklee::Logging::enterFunc( e , __PRETTY_FUNCTION__ );  
  klee::ref<Expr> result = e;

  if (!isa<ConstantExpr>(e)) {
    klee::ref<ConstantExpr> value;
    bool isTrue = false;

    ExecutorUtil::copyOutConstraintUnderSymbolic(state);

    solver->setTimeout(stpTimeout);      
    if (solver->getValue(state, e, value) &&
        solver->mustBeTrue(state, EqExpr::create(e, value), isTrue) &&
        isTrue)
      result = value;
    solver->setTimeout(0);

    ExecutorUtil::copyBackConstraintUnderSymbolic(state);
    // concretize the arguments 
  }
  Gklee::Logging::outItem( result , "result" );
  Gklee::Logging::exitFunc();
  return result;
}


/* Concretize the given expression, and return a possible constant value. 
   'reason' is just a documentation string stating the reason for concretization. */
klee::ref<klee::ConstantExpr> 
Executor::toConstant(ExecutionState &state, 
                     klee::ref<Expr> e,
                     const char *reason) {
  Gklee::Logging::enterFunc( e , __PRETTY_FUNCTION__ );  
  e = state.constraints.simplifyExpr(e);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(e)) {
    CE->ctype = e->ctype;
    if (UseSymbolicConfig)
      CE->accum = e->accum;
    Gklee::Logging::exitFunc();
    return CE;
  }

  ExecutorUtil::copyOutConstraintUnderSymbolic(state);
  klee::ref<ConstantExpr> value;
  bool success = solver->getValue(state, e, value);
  assert(success && "FIXME: Unhandled solver failure");
  (void) success;
  ExecutorUtil::copyBackConstraintUnderSymbolic(state);
    
  std::ostringstream os;
  os << "silently concretizing (reason: " << reason << ") expression " << e 
     << " to value " << value 
     << " (" << (*(state.getPC())).info->file << ":" << (*(state.getPC())).info->line << ")";
      
  if (AllExternalWarnings)
    klee_warning(reason, os.str().c_str());
  else
    klee_warning_once(reason, "%s", os.str().c_str());

  if (!UseSymbolicConfig)
    addConstraint(state, EqExpr::create(e, value));
   
  value->ctype = e->ctype; 
  if (UseSymbolicConfig)
    value->accum = e->accum; 
  Gklee::Logging::outItem< klee::ref< klee::Expr > >( value , "returning" );
  Gklee::Logging::exitFunc();
  return value;
}

klee::ref<klee::ConstantExpr>
Executor::toConstantArguments(ExecutionState &state, 
                              klee::ref<Expr> e, 
                              const char *reason) {
  Gklee::Logging::enterFunc( e , __PRETTY_FUNCTION__ );  
  e = state.constraints.simplifyExpr(e);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(e)) {
    CE->ctype = e->ctype;
    if (UseSymbolicConfig)
      CE->accum = e->accum;
    Gklee::Logging::outItem< klee::ref< klee::Expr >>( CE , "returning" );
    Gklee::Logging::exitFunc();
    return CE;
  }

  ExecutorUtil::copyOutConstraintUnderSymbolic(state);
  klee::ref<ConstantExpr> value;
  bool success = solver->getValue(state, e, value);
  assert(success && "FIXME: Unhandled solver failure");
  (void) success;
  ExecutorUtil::copyBackConstraintUnderSymbolic(state);
    
  std::ostringstream os;
  os << "arguments silently concretizing (reason: " << reason << ") expression " << e 
     << " to value " << value 
     << " (" << (*(state.getPC())).info->file << ":" << (*(state.getPC())).info->line << ")";
      
  if (AllExternalWarnings)
    klee_warning(reason, os.str().c_str());
  else
    klee_warning_once(reason, "%s", os.str().c_str());

  value->ctype = e->ctype; 
  if (UseSymbolicConfig)
    value->accum = e->accum; 
  Gklee::Logging::outItem< klee::ref< klee::Expr >>( value , "returning" );
  Gklee::Logging::exitFunc();
  return value;
}

klee::ref<klee::ConstantExpr> 
Executor::toConstantPublic(ExecutionState &state, 
                           klee::ref<Expr> e,
                           const char *reason) {
  Gklee::Logging::enterFunc( e , __PRETTY_FUNCTION__ );  
  auto cons = toConstant(state, e, reason);
  Gklee::Logging::outItem< klee::ref< klee::Expr >>( cons , "returning" );
  Gklee::Logging::exitFunc();
  return cons;
}

Executor::StatePair 
Executor::forkAsPublic(ExecutionState &current, klee::ref<Expr> cond, bool isInternal) {
  Gklee::Logging::enterFunc( cond , __PRETTY_FUNCTION__ );  
  auto f = fork(current, cond, isInternal); 
  Gklee::Logging::exitFunc();
  return f;
}  

void Executor::executeGetValue(ExecutionState &state,
                               klee::ref<Expr> e,
                               KInstruction *target) {
  Gklee::Logging::enterFunc( e , __PRETTY_FUNCTION__ );  
  e = state.constraints.simplifyExpr(e);
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it==seedMap.end() || isa<ConstantExpr>(e)) {
    klee::ref<ConstantExpr> value;
    bool success = solver->getValue(state, e, value);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    bindLocal(target, state, value);
  } else {
    std::set< klee::ref<Expr> > values;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      klee::ref<ConstantExpr> value;
      bool success = 
        solver->getValue(state, siit->assignment.evaluate(e), value);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      values.insert(value);
    }
    
    std::vector< klee::ref<Expr> > conditions;
    for (std::set< klee::ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit)
      conditions.push_back(EqExpr::create(e, *vit));

    std::vector<ExecutionState*> branches;
    branch(state, conditions, branches);
    
    std::vector<ExecutionState*>::iterator bit = branches.begin();
    for (std::set< klee::ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit) {
      ExecutionState *es = *bit;
      if (es)
        bindLocal(target, *es, *vit);
      ++bit;
    }
  }
  Gklee::Logging::exitFunc();
}

/// Compute the true target of a function call, resolving LLVM and KLEE aliases
/// and bitcasts.
Function* Executor::getTargetFunction(Value *calledVal, ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  SmallPtrSet<const GlobalValue*, 3> Visited;

  Constant *c = dyn_cast<Constant>(calledVal);
  if (!c){
    Gklee::Logging::exitFunc();
    return 0;
  }
  while (true) {
    if (GlobalValue *gv = dyn_cast<GlobalValue>(c)) {
      if (!Visited.insert(gv)){
	Gklee::Logging::outItem< std::string >( "0" , "returning" );
	Gklee::Logging::exitFunc();
        return 0;
      }
      std::string alias = state.getFnAlias(gv->getName());
      if (alias != "") {
        llvm::Module* currModule = kmodule->module;
        GlobalValue *old_gv = gv;
        gv = currModule->getNamedValue(alias);
        if (!gv) {
          llvm::errs() << "Function " << alias << "(), alias for " 
                       << old_gv->getName() << " not found!\n";
          assert(0 && "function alias not found");
        }
      }
     
      if (Function *f = dyn_cast<Function>(gv)){
	Gklee::Logging::outItem( f->getName().str() , "returning" );
	Gklee::Logging::exitFunc();
        return f;
      }else if (GlobalAlias *ga = dyn_cast<GlobalAlias>(gv)){
        c = ga->getAliasee();
      }else{
	Gklee::Logging::outItem< std::string >( "0" , "returning" );
	Gklee::Logging::exitFunc();
        return 0;
      }
    } else if (llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(c)) {
      if (ce->getOpcode()==Instruction::BitCast){
        c = ce->getOperand(0);
      }else{
	Gklee::Logging::outItem< std::string >( "0" , "returning" );
	Gklee::Logging::exitFunc();
        return 0;
      }
    } else{
      Gklee::Logging::outItem< std::string >( "0" , "returning" );
      Gklee::Logging::exitFunc();
      return 0;
    }
    //    Gklee::Logging::exitFunc();
  }
  //Gklee::Logging::exitFunc();
}

static bool isDebugIntrinsic(const Function *f, KModule *KM) {
  Gklee::Logging::enterFunc( f->getName().str() , __PRETTY_FUNCTION__ );  
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 7)
  // Fast path, getIntrinsicID is slow.
  if (f == KM->dbgStopPointFn){
    Gklee::Logging::outItem( "true" , "returning" );
    Gklee::Logging::exitFunc();
    return true;
  }

  switch (f->getIntrinsicID()) {
  case Intrinsic::dbg_stoppoint:
  case Intrinsic::dbg_region_start:
  case Intrinsic::dbg_region_end:
  case Intrinsic::dbg_func_start:
  case Intrinsic::dbg_declare:
    Gklee::Logging::outItem( "true" , "returning" );
    Gklee::Logging::exitFunc();
    return true;

  default:
    Gklee::Logging::outItem( "false" , "returning" );
    Gklee::Logging::exitFunc();
    return false;
  }
#else
  Gklee::Logging::outItem< std::string >( "false" , "returning" );
  Gklee::Logging::exitFunc();
  return false;
#endif
}

static inline const llvm::fltSemantics * fpWidthToSemantics(unsigned width) {
  switch(width) {
  case Expr::Int16:
    return &llvm::APFloat::IEEEhalf;
  case Expr::Int32:
    return &llvm::APFloat::IEEEsingle;
  case Expr::Int64:
    return &llvm::APFloat::IEEEdouble;
  case Expr::Fl80:
    return &llvm::APFloat::x87DoubleExtended;
  default:
    return 0;
  }
}

bool ExecutorUtil::isForkInstruction(Instruction *inst) {
  Gklee::Logging::enterFunc( *inst, __PRETTY_FUNCTION__ );  
  if (inst->getOpcode() == Instruction::Br) {
    if (BranchInst *bi = cast<BranchInst>(inst)) {
      if (bi->isConditional()){
	Gklee::Logging::outItem< std::string >( "true" , "returning" );
	Gklee::Logging::exitFunc();
	return true;
      }
    }
  } else if (inst->getOpcode() == Instruction::Switch) {
    Gklee::Logging::outItem< std::string >( "true" , "returning" );
    Gklee::Logging::exitFunc();  
    return true;
  }
  Gklee::Logging::outItem< std::string >( "false" , "returning" );
  Gklee::Logging::exitFunc();  
  return false;
}

void Executor::updateBaseCType(ExecutionState &state, klee::ref<Expr> &baseAddr) {
  Gklee::Logging::enterFunc( baseAddr , __PRETTY_FUNCTION__ );  
  ExecutorUtil::copyOutConstraintUnderSymbolic(state);
  // First look up the host memory ...
  MemoryMap &hostObj = state.addressSpace.cpuMemory.objects;  

  for (MemoryMap::iterator oi = hostObj.begin(); oi != hostObj.end(); ++oi) {
    const MemoryObject *mo = oi->first;
    Solver::Validity res;
    Expr::Width width = mo->getBaseExpr()->getWidth();
    klee::ref<Expr> baseExpr = ZExtExpr::create(baseAddr, width);
    klee::ref<Expr> ugeExpr = UgeExpr::create(baseExpr, mo->getBaseExpr());

    // ensure that (adress >= base addr)
    solver->evaluate(state, ugeExpr, res);
    if (res == Solver::True) {
      klee::ref<Expr> boundCheckCond = mo->getBoundsCheckPointer(baseExpr);

      // ensure that (address - base addr) < size
      solver->evaluate(state, boundCheckCond, res);
      if (res != Solver::False) {
        baseAddr->ctype = mo->ctype;
        assert(baseAddr->ctype == GPUConfig::HOST && "The ctype mismatches");
	Gklee::Logging::outItem< std::string >( "HOST" , "type" );
        break;
      }
    }
  }

  if (baseAddr->ctype == GPUConfig::UNKNOWN) {
    // Then look up the global memory ...
    MemoryMap &deviceObj = state.addressSpace.deviceMemory.objects;  

    for (MemoryMap::iterator oi = deviceObj.begin(); oi != deviceObj.end(); ++oi) {
      const MemoryObject *mo = oi->first;
      Solver::Validity res;
      Expr::Width width = mo->getBaseExpr()->getWidth();
      klee::ref<Expr> baseExpr = ZExtExpr::create(baseAddr, width);
      klee::ref<Expr> ugeExpr = UgeExpr::create(baseExpr, mo->getBaseExpr());

      // ensure that (adress >= base addr)
      solver->evaluate(state, ugeExpr, res);
      if (res == Solver::True) {
        klee::ref<Expr> boundCheckCond = mo->getBoundsCheckPointer(baseExpr);
        // ensure that (address - base addr) < size
        solver->evaluate(state, boundCheckCond, res);
        if (res == Solver::True || res == Solver::Unknown) {
          baseAddr->ctype = mo->ctype;
          assert(baseAddr->ctype == GPUConfig::DEVICE && "The ctype mismatches");
	  Gklee::Logging::outItem< std::string >( "DEVICE" , "type" );
          break;
        }
      }
    }
  }
  
  if (baseAddr->ctype == GPUConfig::UNKNOWN) {
    // Then look up the shared memory ... 
    unsigned cur_bid = state.tinfo.get_cur_bid();
    MemoryMap &sharedObj = state.addressSpace.sharedMemories[cur_bid].objects;  
    for (MemoryMap::iterator oi = sharedObj.begin(); oi != sharedObj.end(); ++oi) {
      const MemoryObject *mo = oi->first;
      Solver::Validity res;
      Expr::Width width = mo->getBaseExpr()->getWidth();
      klee::ref<Expr> baseExpr = ZExtExpr::create(baseAddr, width);
      klee::ref<Expr> ugeExpr = UgeExpr::create(baseExpr, mo->getBaseExpr());

      // ensure that (adress >= base addr)
      solver->evaluate(state, ugeExpr, res);
      if (res == Solver::True) {
        klee::ref<Expr> boundCheckCond = mo->getBoundsCheckPointer(baseExpr);
        // ensure that (address - base addr) < size
        solver->evaluate(state, boundCheckCond, res);
        if (res == Solver::True || res == Solver::Unknown) {
          baseAddr->ctype = mo->ctype;
          assert(baseAddr->ctype == GPUConfig::SHARED && "The ctype mismatches");
	  Gklee::Logging::outItem< std::string >( "SHARED" , "type" );
          break;
        }
      }
    }
  }

  if (baseAddr->ctype == GPUConfig::UNKNOWN) {
    // Then look up the local memory ... 
    unsigned cur_tid = state.tinfo.get_cur_tid();
    MemoryMap &localObj = state.addressSpace.localMemories[cur_tid].objects;  
    for (MemoryMap::iterator oi = localObj.begin(); oi != localObj.end(); ++oi) {
      const MemoryObject *mo = oi->first;
      Solver::Validity res;
      Expr::Width width = mo->getBaseExpr()->getWidth();
      klee::ref<Expr> baseExpr = ZExtExpr::create(baseAddr, width);
      klee::ref<Expr> ugeExpr = UgeExpr::create(baseExpr, mo->getBaseExpr());

      // ensure that (adress >= base addr)
      solver->evaluate(state, ugeExpr, res);
      if (res == Solver::True) {
        klee::ref<Expr> boundCheckCond = mo->getBoundsCheckPointer(baseExpr);
        // ensure that (address - base addr) < size
        solver->evaluate(state, boundCheckCond, res);
        if (res == Solver::True || res == Solver::Unknown) {
          baseAddr->ctype = mo->ctype;
          assert(baseAddr->ctype == GPUConfig::LOCAL && "The ctype mismatches");
	  Gklee::Logging::outItem< std::string >( "LOCAL" , "type" );
          break;
        }
      }
    }
  }
  ExecutorUtil::copyBackConstraintUnderSymbolic(state); 
  Gklee::Logging::exitFunc();
}

// useRealGrid argument means the GridSize or SymGridSize will be used
void ExecutorUtil::constructSymConfigEncodedConstraint(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  

  // Two symbolic blocks
  klee::ref<Expr> blockConstraint;
  state.constructBlockEncodedConstraint(blockConstraint, 0);
  Gklee::Logging::outItem( blockConstraint , "blockConstraint" );
  addConfigConstraint(state, blockConstraint);
  state.constructBlockEncodedConstraint(blockConstraint, 1);
  Gklee::Logging::outItem( blockConstraint , "blockConstraint" );
  addConfigConstraint(state, blockConstraint);
    
  // Two symbolic threads
  klee::ref<Expr> threadConstraint;
  state.constructThreadEncodedConstraint(threadConstraint, 0);
  Gklee::Logging::outItem( threadConstraint , "threadConstraint" );
  addConfigConstraint(state, threadConstraint);
  state.constructThreadEncodedConstraint(threadConstraint, 1);
  Gklee::Logging::outItem( threadConstraint , "threadConstraint" );
  addConfigConstraint(state, threadConstraint);
  Gklee::Logging::exitFunc();
}

void ExecutorUtil::constructSymBlockDimPrecondition(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  // Two unbounded symbolic blocks
  state.constructUnboundedBlockEncodedConstraint(0);
  state.constructUnboundedBlockEncodedConstraint(1);

  // Two unbounded symbolic threads
  state.constructUnboundedThreadEncodedConstraint(0);
  state.constructUnboundedThreadEncodedConstraint(1);
  Gklee::Logging::exitFunc();
}

void ExecutorUtil::copyOutConstraint(ExecutionState &state, bool ignoreCur) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  state.paraConstraints = state.constraints; 
  constructSymConfigEncodedConstraint(state);
  klee::ref<Expr> cond = state.getTDCCondition(ignoreCur);
  Gklee::Logging::outItem( cond , "TDCCondition" );
  addConfigConstraint(state, cond);
  Gklee::Logging::exitFunc();
}

void ExecutorUtil::copyBackConstraint(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  state.constraints = state.paraConstraints; 
  Gklee::Logging::exitFunc();
}

void ExecutorUtil::copyOutConstraintUnderSymbolic(ExecutionState &state, bool ignoreCur) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (UseSymbolicConfig
       && state.tinfo.is_GPU_mode)
      copyOutConstraint(state, ignoreCur);
  Gklee::Logging::exitFunc();
}

void ExecutorUtil::copyBackConstraintUnderSymbolic(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (UseSymbolicConfig)
    if (state.tinfo.is_GPU_mode) 
      copyBackConstraint(state);
  Gklee::Logging::exitFunc();
}

void ExecutorUtil::addConfigConstraint(ExecutionState &state, klee::ref<Expr> condition) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (isa<ConstantExpr>(condition)){
    Gklee::Logging::exitFunc();
    return;
  }
  state.addConstraint(condition);
  Gklee::Logging::exitFunc();
}

void Executor::updateCType(ExecutionState &state, llvm::Value* value, 
                           klee::ref<Expr> &base, bool is_GPU_mode) { 
  Gklee::Logging::enterFunc( base , __PRETTY_FUNCTION__ );  
  if (base->ctype == GPUConfig::UNKNOWN) {
    if (value) // value != NULL
      base->ctype = CUDAUtil::getUpdatedCType(value, is_GPU_mode);

    if (base->ctype == GPUConfig::UNKNOWN) {
      updateBaseCType(state, base);

      if (base->ctype == GPUConfig::UNKNOWN) {
        if (state.tinfo.is_GPU_mode)
          base->ctype = GPUConfig::LOCAL;
        else 
          base->ctype = GPUConfig::HOST;
      }
    }
  }
  Gklee::Logging::exitFunc();
}

void Executor::executeInstruction(ExecutionState &state, KInstruction *ki) {
  
  Instruction *i = ki->inst;
  
  Gklee::Logging::enterFunc( *i , __PRETTY_FUNCTION__ );  

  unsigned seqNum = 0;

  bool isFork = false;
  if (Trace) {
    ostringstream os;
    os << state.tinfo.get_cur_bid() 
	      << ":" << state.tinfo.get_cur_tid() << std::endl;
    if(MDNode *N = i->getMetadata("dbg")) {
      DILocation Loc(N);
      StringRef File = Loc.getFilename();
      StringRef Dir = Loc.getDirectory();
      os << Loc.getLineNumber() << ":" << Dir.str() << "/" << File.str() <<
	":";
      raw_os_ostream ross(os);
      raw_ostream *ros = dynamic_cast<raw_ostream*>(&ross);
      i->print(*ros, (AssemblyAnnotationWriter*)NULL);
      os << std::endl;
    }
    //os << i << endl;;
    os << "*****" << std::endl;
    state.traceInfo += os.str();
  }
  if (GPUConfig::verbose > 0) {
    if (state.tinfo.is_GPU_mode) {
      std::cout << "bid: " << state.tinfo.get_cur_bid() 
                << ", tid: " << state.tinfo.get_cur_tid() << std::endl;
      std::cout << "inst: " << std::endl;
      i->dump();
    }
  }

  if (ExecutorUtil::isForkInstruction(i)) isFork = true;

  if (!UseSymbolicConfig)
    state.addressSpace.insertInst(state.tinfo.is_GPU_mode, 
                                  state.tinfo.get_cur_bid(), 
                                  state.tinfo.get_cur_tid(), i, 
                                  isFork, seqNum);
  
  switch (i->getOpcode()) {
    // Control flow
  case Instruction::Ret: {
    ReturnInst *ri = cast<ReturnInst>(i);
    KInstIterator kcaller = state.getCurStack().back().caller;
    Instruction *caller = kcaller ? kcaller->inst : 0;
    bool isVoidReturn = (ri->getNumOperands() == 0);
    klee::ref<Expr> result = ConstantExpr::alloc(0, Expr::Bool);
    if (!isVoidReturn) {
      result = eval(ki, 0, state).value;
    }
    Gklee::Logging::outItem( result , "return result" );
    
    if (state.getCurStack().size() <= 1) {
      assert(!caller && "caller set on initial stack frame");
      GKLEE_INFO2 << "Finishing the program!\n ";
      terminateStateOnExit(state);
    } else {
      state.popFrame();

      if (statsTracker)
        statsTracker->framePopped(state);

      if (InvokeInst *ii = dyn_cast<InvokeInst>(caller)) {
        transferToBasicBlock(ii->getNormalDest(), caller->getParent(), state);
      } else {
        state.setPC(kcaller);
        state.incPC();
      }

      if (!isVoidReturn) {
        LLVM_TYPE_Q Type *t = caller->getType();
        if (t != Type::getVoidTy(getGlobalContext())) {
          // may need to do coercion due to bitcasts
          Expr::Width from = result->getWidth();
          Expr::Width to = getWidthForLLVMType(t);
            
          if (from != to) {
            CallSite cs = (isa<InvokeInst>(caller) ? CallSite(cast<InvokeInst>(caller)) : 
                           CallSite(cast<CallInst>(caller)));

            // XXX need to check other param attrs ?
            if (cs.paramHasAttr(0, llvm::Attributes::SExt)) {
              result = SExtExpr::create(result, to);
            } else {
              result = ZExtExpr::create(result, to);
            }
          }
	  Gklee::Logging::outItem( result , "binding result" );
          bindLocal(kcaller, state, result);
        }
      } else {
        // We check that the return value has no users instead of
        // checking the type, since C defaults to returning int for
        // undeclared functions.
        if (!caller->use_empty()) {
          terminateStateOnExecError(state, "return void when caller expected a result");
        }
      }
    }
  
    if (kernelFunc && (i->getParent()->getParent() == kernelFunc)) 
      handleEndGPU(state, ki); 
    break;
  }
#if LLVM_VERSION_CODE < LLVM_VERSION(3, 1)
  case Instruction::Unwind: {
    for (;;) {
      KInstruction *kcaller = state.getCurStack().back().caller;
      state.popFrame();

      if (statsTracker)
        statsTracker->framePopped(state);

      if (state.getCurStack().empty()) {
        terminateStateOnExecError(state, "unwind from initial stack frame");
        break;
      } else {
        Instruction *caller = kcaller->inst;
        if (InvokeInst *ii = dyn_cast<InvokeInst>(caller)) {
          transferToBasicBlock(ii->getUnwindDest(), caller->getParent(), state);
          break;
        }
      }
    }
    break;
  }
#endif
  case Instruction::Br: {
    BranchInst *bi = cast<BranchInst>(i);
    if (bi->isUnconditional()) {
      transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), state);
    } else {
      // Gklee::
      // Gklee::Logging::enterFunc( *i, "handling branch case in execute" );
      // FIXME: Find a way that we don't have this hidden dependency.
      assert(bi->getCondition() == bi->getOperand(0) &&
             "Wrong operand index!");
      klee::ref<Expr> cond = eval(ki, 0, state).value;
      Gklee::Logging::outItem( cond , "branch condition" );

      if (RacePrune) { 
        if (bi->hasMetadata()) {
          if (bi->getMetadata("br-G-G"))
            state.brMeta.meta = GG;
          else if (bi->getMetadata("br-S-S"))
            state.brMeta.meta = SS;
          else if (bi->getMetadata("br-G-S"))
            state.brMeta.meta = GS;
          else if (bi->getMetadata("br-S-G"))
            state.brMeta.meta = SG;
        }
      }
      Executor::StatePair branches = fork(state, cond, false);

      // NOTE: There is a hidden dependency here, markBranchVisited
      // requires that we still be in the context of the branch
      // instruction (it reuses its statistic id). Should be cleaned
      // up with convenient instruction specific data.
      if (statsTracker && state.getCurStack().back().kf->trackCoverage)
        statsTracker->markBranchVisited(branches.first, branches.second);

      // per thread coverage
      if (branches.first) {
	Gklee::Logging::outItem< std::string >( "first" , "branching" );
        if (!RacePrune)
          bc_cov_monitor.markTakenBranch(&state, true);
        transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
      }
      if (branches.second) {
	Gklee::Logging::outItem< std::string >( "second" , "branching" );
        if (!RacePrune)
	  bc_cov_monitor.markTakenBranch(&state, false);
        transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
      }
    }
    // Gklee::Logging::exitFunc();
    // Gklee::
    break;
  }
  case Instruction::Switch: {
    SwitchInst *si = cast<SwitchInst>(i);
    klee::ref<Expr> cond = eval(ki, 0, state).value;
    Gklee::Logging::outItem( cond , "switch condition" );
    BasicBlock *bb = si->getParent();

    cond = toUnique(state, cond);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(cond)) {
      // Somewhat gross to create these all the time, but fine till we
      // switch to an internal rep.
      LLVM_TYPE_Q llvm::IntegerType *Ty = 
        cast<IntegerType>(si->getCondition()->getType());
      ConstantInt *ci = ConstantInt::get(Ty, CE->getZExtValue());
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
      unsigned index = si->findCaseValue(ci).getSuccessorIndex();
#else
      unsigned index = si->findCaseValue(ci);
#endif
      transferToBasicBlock(si->getSuccessor(index), si->getParent(), state);
    } else {
      // Preserve ... 
      ExecutorUtil::copyOutConstraintUnderSymbolic(state);
      std::map<BasicBlock*, klee::ref<Expr> > targets;
      klee::ref<Expr> isDefault = ConstantExpr::alloc(1, Expr::Bool);
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)      
      for (SwitchInst::CaseIt i = si->case_begin(), e = si->case_end();
           i != e; ++i) {
        klee::ref<Expr> value = evalConstant(i.getCaseValue());
	Gklee::Logging::outItem( value , "case value" );
#else
      for (unsigned i=1, cases = si->getNumCases(); i<cases; ++i) {
        klee::ref<Expr> value = evalConstant(si->getCaseValue(i));
#endif
        klee::ref<Expr> match = EqExpr::create(cond, value);
        isDefault = AndExpr::create(isDefault, Expr::createIsZero(match));
        bool result;
        bool success = solver->mayBeTrue(state, match, result);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (result) {
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
          BasicBlock *caseSuccessor = i.getCaseSuccessor();
#else
          BasicBlock *caseSuccessor = si->getSuccessor(i);
#endif
          std::map<BasicBlock*, klee::ref<Expr> >::iterator it =
            targets.insert(std::make_pair(caseSuccessor,
                           ConstantExpr::alloc(0, Expr::Bool))).first;
          it->second = OrExpr::create(match, it->second);
        }
      }
      bool res;
      bool success = solver->mayBeTrue(state, isDefault, res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      // Restore ... 
      ExecutorUtil::copyBackConstraintUnderSymbolic(state);
      if (res)
        targets.insert(std::make_pair(si->getDefaultDest(), isDefault));
      
      std::vector< klee::ref<Expr> > conditions;
      for (std::map<BasicBlock*, klee::ref<Expr> >::iterator it = 
             targets.begin(), ie = targets.end();
           it != ie; ++it){
	Gklee::Logging::outItem( it->second , "adding condition" );
        conditions.push_back(it->second);
      }
      
      std::vector<ExecutionState*> branches;
      branch(state, conditions, branches);
       
      std::vector<ExecutionState*>::iterator bit = branches.begin();
      for (std::map<BasicBlock*, klee::ref<Expr> >::iterator it = 
             targets.begin(), ie = targets.end();
           it != ie; ++it) {
        ExecutionState *es = *bit;
        if (es)
          transferToBasicBlock(it->first, bb, *es);
        ++bit;
      }
    break;
      }
    }
  case Instruction::Unreachable:
    // Note that this is not necessarily an internal bug, llvm will
    // generate unreachable instructions in cases where it knows the
    // program will crash. So it is effectively a SEGV or internal
    // error.
    terminateStateOnExecError(state, "reached \"unreachable\" instruction");
    break;

  case Instruction::Invoke:
  case Instruction::Call: {
    CallSite cs(i);

    unsigned numArgs = cs.arg_size();
    Value *fp = cs.getCalledValue();
    Function *f = getTargetFunction(fp, state);
    Gklee::Logging::outItem( f->getName().str(), "target function" );
    // Skip debug intrinsics, we can't evaluate their metadata arguments.
    if (f && isDebugIntrinsic(f, kmodule))
      break;

    if (isa<InlineAsm>(fp)) {
      terminateStateOnExecError(state, "inline assembly is unsupported");
      break;
    }
    // evaluate arguments
    std::vector< klee::ref<Expr> > arguments;
    arguments.reserve(numArgs);

    for (unsigned j=0; j<numArgs; ++j){
      auto arg = eval(ki, j+1, state).value;
      Gklee::Logging::outItem( arg , "arg" );
      arguments.push_back( arg );
    }
     
    if (f) {
      const FunctionType *fType =
        dyn_cast<FunctionType>(cast<PointerType>(f->getType())->getElementType());
      const FunctionType *fpType =
        dyn_cast<FunctionType>(cast<PointerType>(fp->getType())->getElementType());

      // special case the call with a bitcast case
      if (fType != fpType) {
        assert(fType && fpType && "unable to get function type");

        // XXX check result coercion

        // XXX this really needs thought and validation
        unsigned i=0;
        for (std::vector< klee::ref<Expr> >::iterator
               ai = arguments.begin(), ie = arguments.end();
             ai != ie; ++ai) {
          Expr::Width to, from = (*ai)->getWidth();

          if (i<fType->getNumParams()) {
            to = getWidthForLLVMType(fType->getParamType(i));

            if (from != to) {
              // XXX need to check other param attrs ?
              if (cs.paramHasAttr(i+1, llvm::Attributes::SExt)) {
                arguments[i] = SExtExpr::create(arguments[i], to);
              } else {
                arguments[i] = ZExtExpr::create(arguments[i], to);
              }
            }
          }

          i++;
        }
      }

      executeCall(state, ki, f, arguments, seqNum);
    } else {
      klee::ref<Expr> v = eval(ki, 0, state).value;

      Gklee::Logging::outItem( v , "op 0 value" );

      ExecutionState *free = &state;
      bool hasInvalid = false, first = true;
      /* XXX This is wasteful, no need to do a full evaluate since we
         have already got a value. But in the end the caches should
         handle it for us, albeit with some overhead. */
      do {
        klee::ref<ConstantExpr> value;
        bool success = solver->getValue(*free, v, value);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        StatePair res = fork(*free, EqExpr::create(v, value), true);
        if (res.first) {
          uint64_t addr = value->getZExtValue();
          if (legalFunctions.count(addr)) {
            f = (Function*) addr;
            // Don't give warning on unique resolution
            if (res.second || !first)
              klee_warning_once((void*) (unsigned long) addr,
                                "resolved symbolic function pointer to: %s",
                                f->getName().data());

            executeCall(*res.first, ki, f, arguments, seqNum);
          } else {
            if (!hasInvalid) {
              terminateStateOnExecError(state, "invalid function pointer");
              hasInvalid = true;
            }
          }
        }

        first = false;
        free = res.second;
      } while (free);
    }
    break;
  }
  case Instruction::PHI: {
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 0)
    klee::ref<Expr> result = eval(ki, state.incomingBBIndex[state.tinfo.get_cur_tid()], state).value;
#else
    klee::ref<Expr> result = eval(ki, state.incomingBBIndex[state.tinfo.get_cur_tid()] * 2, state).value;
#endif
    Gklee::Logging::outItem( result , "phi result" );
    bindLocal(ki, state, result);

    break;
  }

    // Special instructions
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(ki->inst);
    assert(SI->getCondition() == SI->getOperand(0) &&
           "Wrong operand index!");
    klee::ref<Expr> cond = eval(ki, 0, state).value;
    klee::ref<Expr> tExpr = eval(ki, 1, state).value;
    klee::ref<Expr> fExpr = eval(ki, 2, state).value;
    klee::ref<Expr> result = SelectExpr::create(cond, tExpr, fExpr);
    Gklee::Logging::outItem( result , "select result" );
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::VAArg:
    terminateStateOnExecError(state, "unexpected VAArg instruction");
    break;

  // Arithmetic / logical

  case Instruction::Add: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = AddExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Sub: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = SubExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }
 
  case Instruction::Mul: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = MulExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::UDiv: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = UDivExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::SDiv: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = SDivExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::URem: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = URemExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }
 
  case Instruction::SRem: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = SRemExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::And: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = AndExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Or: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = OrExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Xor: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = XorExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Shl: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = ShlExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::LShr: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = LShrExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::AShr: {
    klee::ref<Expr> left = eval(ki, 0, state).value;
    klee::ref<Expr> right = eval(ki, 1, state).value;
    klee::ref<Expr> result = AShrExpr::create(left, right);
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

    // Compare

  case Instruction::ICmp: {
    CmpInst *ci = cast<CmpInst>(i);
    ICmpInst *ii = cast<ICmpInst>(ci);
 
    switch(ii->getPredicate()) {
    case ICmpInst::ICMP_EQ: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = EqExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_NE: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = NeExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_UGT: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = UgtExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state,result);
      break;
    }

    case ICmpInst::ICMP_UGE: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = UgeExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULT: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = UltExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULE: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = UleExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGT: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = SgtExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGE: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = SgeExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLT: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = SltExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLE: {
      klee::ref<Expr> left = eval(ki, 0, state).value;
      klee::ref<Expr> right = eval(ki, 1, state).value;
      klee::ref<Expr> result = SleExpr::create(left, right);
      if (UseSymbolicConfig && (left->accum || right->accum))
        result->accum = true;
      bindLocal(ki, state, result);
      break;
    }

    default:
      terminateStateOnExecError(state, "invalid ICmp predicate");
    }
    break;
  }
 
    // Memory instructions...
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 7)
  case Instruction::Malloc:
  case Instruction::Alloca: {
    AllocationInst *ai = cast<AllocationInst>(i);
#else
  case Instruction::Alloca: {
    AllocaInst *ai = cast<AllocaInst>(i);
#endif
    unsigned elementSize = 
      kmodule->targetData->getTypeStoreSize(ai->getAllocatedType());
    klee::ref<Expr> size = Expr::createPointer(elementSize);
    if (ai->isArrayAllocation()) {
      klee::ref<Expr> count = eval(ki, 0, state).value;
      count = Expr::createZExtToPointerWidth(count);
      size = MulExpr::create(size, count);
      Gklee::Logging::outItem( size , "array alloc size" );
    }
    bool isLocal = i->getOpcode()==Instruction::Alloca;
    executeAlloc(state, size, isLocal, ki);
    //    if(GPUConfig::verbose > 0){
      //dumpInfo(memoryState, "Completed alloc", state);
      //    }
    break;
  }
#if LLVM_VERSION_CODE < LLVM_VERSION(2, 7)
  case Instruction::Free: {
    klee::ref<Expr> base = eval(ki, 0, state).value;
    Gklee::Logging::outItem( base , "free addr" );
    executeFree(state, base);
    break;
  }
#endif

  case Instruction::Load: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    klee::ref<Expr> base = eval(ki, 0, state).value;
    Gklee::Logging::outItem( base , "load addr" );
    updateCType(state, kgepi->inst->getOperand(0), 
                base, state.tinfo.is_GPU_mode);

    if (UseSymbolicConfig) {
      if (accumTaintSet.find(i) != accumTaintSet.end()) {
	Gklee::Logging::outItem< std::string >( "true" , "in taint set" );
        accumStore = true;
      }
      Gklee::Logging::outItem< std::string >( "false" , "in taint set" );
    }
      
    executeMemoryOperation(state, false, base, 0, ki, seqNum);
    //    if(GPUConfig::verbose > 0){
      //dumpInfo(readSet, "Completed load", state);
      //    }
    break;
  }
  case Instruction::Store: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    klee::ref<Expr> base = eval(ki, 1, state).value;
    klee::ref<Expr> value = eval(ki, 0, state).value;
    updateCType(state, kgepi->inst->getOperand(1), 
                base, state.tinfo.is_GPU_mode);

    Gklee::Logging::outItem( base , "store addr" );
    executeMemoryOperation(state, true, base, value, ki, seqNum);
    //    if(GPUConfig::verbose > 0){
      //dumpInfo(writeSet, "Completed store", state);
      //    }
    break;
  }

  case Instruction::GetElementPtr: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    klee::ref<Expr> base = eval(ki, 0, state).value;
    updateCType(state, kgepi->inst->getOperand(0), 
                base, state.tinfo.is_GPU_mode);
    GPUConfig::CTYPE ctype = base->ctype;
    
    for (std::vector< std::pair<unsigned, uint64_t> >::iterator 
           it = kgepi->indices.begin(), ie = kgepi->indices.end(); 
         it != ie; ++it) {
      uint64_t elementSize = it->second;
      klee::ref<Expr> index = eval(ki, it->first, state).value;
      base = AddExpr::create(base,
                             MulExpr::create(Expr::createSExtToPointerWidth(index),
                                             Expr::createPointer(elementSize)));
    }
    if (kgepi->offset)
      base = AddExpr::create(base, Expr::createPointer(kgepi->offset));

    base->ctype = ctype;
    bindLocal(ki, state, base);
    //    if(GPUConfig::verbose > 0){
      //dumpInfo(memoryState, "Completed getElementPtr", state);
      //  }
    break;
  }

  // Conversion
  case Instruction::Trunc: {
    CastInst *ci = cast<CastInst>(i);
    klee::ref<Expr> tmp = eval(ki, 0, state).value; 
    klee::ref<Expr> result = ExtractExpr::create(tmp,
                                           0,
                                           getWidthForLLVMType(ci->getType()));
    if (UseSymbolicConfig && tmp->accum)
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ZExt: {
    CastInst *ci = cast<CastInst>(i);
    klee::ref<Expr> tmp = eval(ki, 0, state).value;
    klee::ref<Expr> result = ZExtExpr::create(tmp,
                                        getWidthForLLVMType(ci->getType()));
    if (UseSymbolicConfig && tmp->accum) 
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::SExt: {
    CastInst *ci = cast<CastInst>(i);
    klee::ref<Expr> tmp = eval(ki, 0, state).value;
    klee::ref<Expr> result = SExtExpr::create(tmp,
                                        getWidthForLLVMType(ci->getType()));
    if (UseSymbolicConfig && tmp->accum)
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::IntToPtr: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width pType = getWidthForLLVMType(ci->getType());
    klee::ref<Expr> arg = eval(ki, 0, state).value;
    klee::ref<Expr> tmp = ZExtExpr::create(arg, pType);
    if (UseSymbolicConfig && arg->accum)
      tmp->accum = true; 
    bindLocal(ki, state, tmp);
    break;
  } 
  case Instruction::PtrToInt: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width iType = getWidthForLLVMType(ci->getType());
    klee::ref<Expr> arg = eval(ki, 0, state).value;
    klee::ref<Expr> tmp = ZExtExpr::create(arg, iType);
    if (UseSymbolicConfig && arg->accum)
      tmp->accum = true;
    bindLocal(ki, state, tmp);
    break;
  }

  case Instruction::BitCast: {
    klee::ref<Expr> result = eval(ki, 0, state).value;
    bindLocal(ki, state, result);
    break;
  }

    // Floating point instructions

  case Instruction::FAdd: {
    klee::ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    klee::ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth())){
      terminateStateOnExecError(state, "Unsupported FAdd operation");
    }

    llvm::APFloat Res(left->getAPValue());
    Res.add(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
    klee::ref<Expr> result = ConstantExpr::alloc(Res.bitcastToAPInt()); 
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true; 
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::FSub: {
    klee::ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    klee::ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth())){
      terminateStateOnExecError(state, "Unsupported FSub operation");
    }

    llvm::APFloat Res(left->getAPValue());
    Res.subtract(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
    klee::ref<Expr> result = ConstantExpr::alloc(Res.bitcastToAPInt()); 
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true; 
    bindLocal(ki, state, result);
    break;
  }
 
  case Instruction::FMul: {
    klee::ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    klee::ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth())){
      terminateStateOnExecError(state, "Unsupported FMul operation");
    }

    llvm::APFloat Res(left->getAPValue());
    Res.multiply(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
    klee::ref<Expr> result = ConstantExpr::alloc(Res.bitcastToAPInt());
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::FDiv: {
    klee::ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    klee::ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth())){
      terminateStateOnExecError(state, "Unsupported FDiv operation");
    }

    llvm::APFloat Res(left->getAPValue());
    Res.divide(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
    klee::ref<Expr> result = ConstantExpr::alloc(Res.bitcastToAPInt());
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::FRem: {
    klee::ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    klee::ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth())){
      terminateStateOnExecError(state, "Unsupported FRem operation");
    }

    llvm::APFloat Res(left->getAPValue());
    Res.mod(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
    klee::ref<Expr> result = ConstantExpr::alloc(Res.bitcastToAPInt());
    if (UseSymbolicConfig && (left->accum || right->accum))
      result->accum = true;
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::FPTrunc: {
    FPTruncInst *fi = cast<FPTruncInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    klee::ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > arg->getWidth()){
      terminateStateOnExecError(state, "Unsupported FPTrunc operation");
    }

    llvm::APFloat Res(arg->getAPValue());
    bool losesInfo = false;
    Res.convert(*fpWidthToSemantics(resultType),
                llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
    bindLocal(ki, state, ConstantExpr::alloc(Res));
    break;
  }

  case Instruction::FPExt: {
    FPExtInst *fi = cast<FPExtInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    klee::ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || arg->getWidth() > resultType){
      terminateStateOnExecError(state, "Unsupported FPExt operation");
    }

    llvm::APFloat Res(arg->getAPValue());
    bool losesInfo = false;
    Res.convert(*fpWidthToSemantics(resultType),
                llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
    bindLocal(ki, state, ConstantExpr::alloc(Res));
    break;
  }

  case Instruction::FPToUI: {
    FPToUIInst *fi = cast<FPToUIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    klee::ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64){
      terminateStateOnExecError(state, "Unsupported FPToUI operation");
    }

    llvm::APFloat Arg(arg->getAPValue());
    uint64_t value = 0;
    bool isExact = true;
    Arg.convertToInteger(&value, resultType, false,
                         llvm::APFloat::rmTowardZero, &isExact);
    bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    break;
  }

  case Instruction::FPToSI: {
    FPToSIInst *fi = cast<FPToSIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    klee::ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64){
      terminateStateOnExecError(state, "Unsupported FPToSI operation");
    }

    llvm::APFloat Arg(arg->getAPValue());
    uint64_t value = 0;
    bool isExact = true;
    Arg.convertToInteger(&value, resultType, true,
                         llvm::APFloat::rmTowardZero, &isExact);
    bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    break;
  }

  case Instruction::UIToFP: {
    UIToFPInst *fi = cast<UIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    klee::ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics){
      terminateStateOnExecError(state, "Unsupported UIToFP operation");
    }
    llvm::APFloat f(*semantics, 0);
    f.convertFromAPInt(arg->getAPValue(), false,
                       llvm::APFloat::rmNearestTiesToEven);

    bindLocal(ki, state, ConstantExpr::alloc(f));
    break;
  }

  case Instruction::SIToFP: {
    SIToFPInst *fi = cast<SIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    klee::ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics){
      terminateStateOnExecError(state, "Unsupported SIToFP operation");
    }
    llvm::APFloat f(*semantics, 0);
    f.convertFromAPInt(arg->getAPValue(), true,
                       llvm::APFloat::rmNearestTiesToEven);

    bindLocal(ki, state, ConstantExpr::alloc(f));
    break;
  }

  case Instruction::FCmp: {
    FCmpInst *fi = cast<FCmpInst>(i);
    klee::ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    klee::ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth())){
      terminateStateOnExecError(state, "Unsupported FCmp operation");
    }

    APFloat LHS(left->getAPValue());
    APFloat RHS(right->getAPValue());
    APFloat::cmpResult CmpRes = LHS.compare(RHS);

    bool Result = false;
    switch( fi->getPredicate() ) {
      // Predicates which only care about whether or not the operands are NaNs.
    case FCmpInst::FCMP_ORD:
      Result = CmpRes != APFloat::cmpUnordered;
      break;

    case FCmpInst::FCMP_UNO:
      Result = CmpRes == APFloat::cmpUnordered;
      break;

      // Ordered comparisons return false if either operand is NaN.  Unordered
      // comparisons return true if either operand is NaN.
    case FCmpInst::FCMP_UEQ:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OEQ:
      Result = CmpRes == APFloat::cmpEqual;
      break;

    case FCmpInst::FCMP_UGT:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OGT:
      Result = CmpRes == APFloat::cmpGreaterThan;
      break;

    case FCmpInst::FCMP_UGE:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OGE:
      Result = CmpRes == APFloat::cmpGreaterThan || CmpRes == APFloat::cmpEqual;
      break;

    case FCmpInst::FCMP_ULT:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OLT:
      Result = CmpRes == APFloat::cmpLessThan;
      break;

    case FCmpInst::FCMP_ULE:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OLE:
      Result = CmpRes == APFloat::cmpLessThan || CmpRes == APFloat::cmpEqual;
      break;

    case FCmpInst::FCMP_UNE:
      Result = CmpRes == APFloat::cmpUnordered || CmpRes != APFloat::cmpEqual;
      break;
    case FCmpInst::FCMP_ONE:
      Result = CmpRes != APFloat::cmpUnordered && CmpRes != APFloat::cmpEqual;
      break;

    default:
      assert(0 && "Invalid FCMP predicate!");
    case FCmpInst::FCMP_FALSE:
      Result = false;
      break;
    case FCmpInst::FCMP_TRUE:
      Result = true;
      break;
    }

    bindLocal(ki, state, ConstantExpr::alloc(Result, Expr::Bool));
    break;
  }
  case Instruction::InsertValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    klee::ref<Expr> agg = eval(ki, 0, state).value;
    klee::ref<Expr> val = eval(ki, 1, state).value;

    klee::ref<Expr> l = NULL, r = NULL;
    unsigned lOffset = kgepi->offset*8, rOffset = kgepi->offset*8 + val->getWidth();

    if (lOffset > 0)
      l = ExtractExpr::create(agg, 0, lOffset);
    if (rOffset < agg->getWidth())
      r = ExtractExpr::create(agg, rOffset, agg->getWidth() - rOffset);

    klee::ref<Expr> result;
    if (!l.isNull() && !r.isNull())
      result = ConcatExpr::create(r, ConcatExpr::create(val, l));
    else if (!l.isNull())
      result = ConcatExpr::create(val, l);
    else if (!r.isNull())
      result = ConcatExpr::create(r, val);
    else
      result = val;

    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ExtractValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    klee::ref<Expr> agg = eval(ki, 0, state).value;

    klee::ref<Expr> result = ExtractExpr::create(agg, kgepi->offset*8, getWidthForLLVMType(i->getType()));

    bindLocal(ki, state, result);
    break;
  }

    // Other instructions...
    // Unhandled
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
    terminateStateOnError(state, "XXX vector instructions unhandled",
                          "xxx.err");
    break;
 
  default:
    terminateStateOnExecError(state, "illegal instruction");
    break;
  }

  if (SimdSchedule && state.tinfo.is_GPU_mode) {
    if (UseSymbolicConfig) {
      unsigned cur_tid = state.tinfo.get_cur_tid();
      if (isFork) {
        // This branch has been added into the Parameterized Tree
        if (state.tinfo.builtInFork)
          state.getCurrentParaTree().initializeCurrentNodeRange(cur_tid, seqNum); 
        if (state.tinfo.warpInBranch)
          state.getCurrentParaTree().incrementCurrentNodeRange(cur_tid, seqNum);
      } else {
        if (state.tinfo.warpInBranch) {
          state.getCurrentParaTree().incrementCurrentNodeRange(cur_tid, seqNum);
          state.symEncounterPostDominator(i);
        }
      }
    } else {
      if (isFork) {
        if (i->getOpcode() == Instruction::Br) {
          state.addBranchDivRegionSet(postDominator, i, true, seqNum);
          // update the added states set
          for (std::set<ExecutionState*>::iterator si = addedStates.begin(); 
               si != addedStates.end(); si++) {
            (*si)->addBranchDivRegionSet(postDominator, i, true, seqNum);
          }
        } else {
          state.addBranchDivRegionSet(postDominator, i, false, seqNum);
          // update the added states set
          for (std::set<ExecutionState*>::iterator si = addedStates.begin(); 
               si != addedStates.end(); si++) {
            (*si)->addBranchDivRegionSet(postDominator, i, false, seqNum);
          }
        }
      } else {
        if (state.cTidSets[state.tinfo.get_cur_tid()].inBranch) {
          std::vector<BranchDivRegionSet> &branchDivRegionSets =
                                          state.addressSpace.branchDivRegionSets;
          unsigned size = branchDivRegionSets.size();
          if (size > 0 && !branchDivRegionSets[size-1].explored)
            branchDivRegionSets[size-1].explored = true;

          state.updateBranchDivRegionSet(i, seqNum);
        }
      }
    }
  }
  Gklee::Logging::exitFunc();
  
}

// by Peng
// construct the shared memory for each block 
// Currently variables declared with __shared__ are treated as global ones by llvm.
// Hence, each block's shared memory will be constructed memory through this "global"
// shared memory.
void Executor::constructSharedMemory(ExecutionState &state, unsigned bid) {
  // Fake that there is an Alloca instruction occurring here
  // The parameter is obtained by referring the global objects.
  Gklee::Logging::enterFunc< std::string >( std::string( "bid:") + 
					    std::to_string( bid ),
					    __PRETTY_FUNCTION__ );  
  std::map<const llvm::GlobalValue*, MemoryObject*>::iterator it;
  for (it = globalObjects.begin(); it != globalObjects.end(); it++) {
    MemoryObject* glmo = (*it).second;
    if (glmo->ctype == GPUConfig::SHARED && !glmo->is_builtin) {
      // The memoryobjects of shared type are only considered
      if (GPUConfig::verbose > 0) {
        std::cout << "The original \"global\" shared memory base addr: " << std::endl;
        std::cout << "name : " << glmo->name << std::endl;
        glmo->getBaseExpr()->dump();    
      }

      klee::ref<ConstantExpr> size = glmo->getSizeExpr();
      // Mimic KLEE's allocation scheme ..
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(size)) {
        MemoryObject *sharemo = NULL;
        sharemo = memory->allocateSharedMO(CE->getZExtValue());

        if (sharemo) {
          if (GPUConfig::verbose > 0) {
            std::cout << "The newly constructed shared memory base addr: " << std::endl;
            sharemo->getBaseExpr()->dump();   
          }
	  Gklee::Logging::outItem< klee::ref< klee::Expr >>( sharemo->getBaseExpr() , "base addr" );
          Cell *c = new Cell();
          SharedMemoryObject *shareMemObj = new SharedMemoryObject(state.getKernelNum(), glmo, c);
          std::string str;
          llvm::raw_string_ostream name(str);
          name << glmo->getName() << "_block_" << bid;
          sharemo->name = name.str();
          name.flush();       
          // set the CType as shared
          sharemo->ctype = GPUConfig::SHARED;
          sharemo->setBTId(bid);

          sharedObjects.push_back(std::make_pair(shareMemObj, sharemo));
          ObjectState *os = bindObjectInStateToShared(state, sharemo, bid);
          os->initializeToZero();
        }
      }
    }
  }
  Gklee::Logging::exitFunc();
}

// clear the shared memories ...
void Executor::clearSharedMemory(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  /*unsigned i = 0;
  std::cout << "size: " << sharedObjects.size() << std::endl;
  std::vector< std::pair<MemoryObject*, MemoryObject*> >::iterator vit;
  for (vit = sharedObjects.begin(); vit != sharedObjects.end(); ++vit, ++i) {
    state.addressSpace.unbindObject((*vit).second, (*vit).second->getBTId());
    (*vit).second = 0;
  }

  sharedObjects.clear();*/
  Gklee::Logging::exitFunc();
}

// by Guodong
// map known intrinsic functions to the library functions
void Executor::callIntrinsicFunction(ExecutionState &state, 
				     KInstruction *ki,
				     Function *f,
				     std::vector< klee::ref<Expr> > &arguments) {
  Gklee::Logging::enterFunc( f->getName().str(), __PRETTY_FUNCTION__ );  
  std::string f_name = f->getName();
  bool b = false;
  llvm::Module* currModule = kmodule->module;

  if (f_name.compare(0, 11, "llvm.memcpy") == 0) {
    f = currModule->getFunction("memcpy");
    b = true;
  }
  else if (f_name.compare(0, 12, "llvm.memmove") == 0) {
    f = currModule->getFunction("memmove");
    b = true;
  }
  else if (f_name.compare(0, 11, "llvm.memset") == 0) {
    f = currModule->getFunction("memset");
    b = true;
  }

  if (b) {
    arguments.pop_back();
    executeCall(state, ki, f, arguments);
    // Function* f1 = Function::Create(f->getFunctionType(), f->getLinkage(), lib_name);
    // callExternalFunction(state, ki, f1, arguments);
    // delete f1;
  }
  else {
    // skip the other cases for this moment
    klee_warning("ignore intrinsic function: %s", f->getName().data());
  }
  Gklee::Logging::exitFunc();
}

static bool enterRealGPUKernel(std::string kernelName, 
                               std::set<std::string> &kernelSet) {
  Gklee::Logging::enterFunc( std::string( "Searching kernelset.txt" ), __PRETTY_FUNCTION__ );  
  bool isReal = false;
  for (std::set<std::string>::iterator si = kernelSet.begin(); 
       si != kernelSet.end(); si++) {
    if ((*si).compare(kernelName) == 0) {
      isReal = true;
      break;
    }
  }
  Gklee::Logging::outItem( std::to_string( isReal ), "return" );
  Gklee::Logging::exitFunc();
  return isReal;
}

void Executor::executeCall(ExecutionState &state, 
                           KInstruction *ki,
                           Function *f,
                           std::vector< klee::ref<Expr> > &arguments, 
                           unsigned seqNum) {

  Gklee::Logging::enterFunc( f->getName().str(), __PRETTY_FUNCTION__ );  
  if (state.tinfo.kernel_call) {
    if (f) {
      std::string kernelName = f->getName().str();
      state.tinfo.just_enter_GPU_mode = enterRealGPUKernel(kernelName, kernelSet);
    }
  }

  if (f && f->isDeclaration()) {
    Gklee::Logging::outItem( std::string( "intrinsic?" ), 
			     std::string( "func is declaration" ));
    //std::cout << "execute declaration: " << f->getName().str() << std::endl;
    switch(f->getIntrinsicID()) {
    case Intrinsic::not_intrinsic:
      // state may be destroyed by this call, cannot touch
      executeCUDAIntrinsics(state, ki, f, arguments, seqNum);
      break;
        
      // va_arg is handled by caller and intrinsic lowering, see comment for
      // ExecutionState::varargs
    case Intrinsic::vastart:  {
      StackFrame &sf = state.getCurStack().back();
      
      assert(sf.varargs && 
             "vastart called in function with no vararg object");

      // FIXME: This is really specific to the architecture, not the pointer
      // size. This happens to work fir x86-32 and x86-64, however.
      Expr::Width WordSize = Context::get().getPointerWidth();
      if (WordSize == Expr::Int32) {
        executeMemoryOperation(state, true, arguments[0], 
                               sf.varargs->getBaseExpr(), 0);
      } else {
        assert(WordSize == Expr::Int64 && "Unknown word size!");

        // X86-64 has quite complicated calling convention. However,
        // instead of implementing it, we can do a simple hack: just
        // make a function believe that all varargs are on stack.
        executeMemoryOperation(state, true, arguments[0],
                               ConstantExpr::create(48, 32), 0); // gp_offset
        klee::ref<Expr> addr1 = AddExpr::create(arguments[0], 
                                          ConstantExpr::create(4, 64));
        addr1->ctype = arguments[0]->ctype; 
        executeMemoryOperation(state, true, addr1,
                               ConstantExpr::create(304, 32), 0); // fp_offset
        klee::ref<Expr> addr2 = AddExpr::create(arguments[0], 
                                          ConstantExpr::create(8, 64));
        addr2->ctype = arguments[0]->ctype; 
        executeMemoryOperation(state, true, addr2,
                               sf.varargs->getBaseExpr(), 0); // overflow_arg_area
        klee::ref<Expr> addr3 = AddExpr::create(arguments[0], 
                                          ConstantExpr::create(16, 64));
        addr3->ctype = arguments[0]->ctype; 
        executeMemoryOperation(state, true, addr3, 
                               ConstantExpr::create(0, 64), 0); // reg_save_area
      }
      break;
    }
    case Intrinsic::vaend:
      // va_end is a noop for the interpreter.
      //
      // FIXME: We should validate that the target didn't do something bad
      // with vaeend, however (like call it twice).
      break;
        
    case Intrinsic::vacopy:
      // va_copy should have been lowered.
      //
      // FIXME: It would be nice to check for errors in the usage of this as
      // well.
      break;
    default:
      // By Guodong: ingore unknown intrinsic functions
      //             C++ programs introduce extra LLVM intrinsic functions
      callIntrinsicFunction(state, ki, f, arguments);
      // klee_error("unknown intrinsic: %s", f->getName().data());
    }

    if (!state.tinfo.is_GPU_mode) {
      if (InvokeInst *ii = dyn_cast<InvokeInst>(ki->inst))
        transferToBasicBlock(ii->getNormalDest(), ki->inst->getParent(), state);
    }
  } else {
    // FIXME: I'm not really happy about this reliance on prevPC but it is ok, I
    // guess. This just done to avoid having to pass KInstIterator everywhere
    // instead of the actual instruction, since we can't make a KInstIterator
    // from just an instruction (unlike LLVM).
    //std::cout << "execute non-declaration: " << f->getName().str() << std::endl;
    KFunction *kf = kmodule->functionMap[f];
    state.pushFrame(state.getPrevPC(), kf);
    state.setPC(kf->instructions);
    Gklee::Logging::outItem< std::string >( "show caller and kf info" , "pushing new stack frame on current stack" );
     if (statsTracker)
      statsTracker->framePushed(state, &state.getCurStack()[state.getCurStack().size()-2]);
 
     // TODO: support "byval" parameter attribute
     // TODO: support zeroext, signext, sret attributes
        
    unsigned callingArgs = arguments.size();
    unsigned funcArgs = f->arg_size();
    if (!f->isVarArg()) {
      if (callingArgs > funcArgs) {
        klee_warning_once(f, "calling %s with extra arguments.", 
                          f->getName().data());
      } else if (callingArgs < funcArgs) {
        terminateStateOnError(state, "calling function with too few arguments", 
                              "user.err");
	Gklee::Logging::exitFunc();
        return;
      }
    } else {
      if (callingArgs < funcArgs) {
        terminateStateOnError(state, "calling function with too few arguments", 
                              "user.err");
	Gklee::Logging::exitFunc();
        return;
      }
            
      StackFrame &sf = state.getCurStack().back();
      unsigned size = 0;
      for (unsigned i = funcArgs; i < callingArgs; i++) {
        // FIXME: This is really specific to the architecture, not the pointer
        // size. This happens to work fir x86-32 and x86-64, however.
        Expr::Width WordSize = Context::get().getPointerWidth();
        if (WordSize == Expr::Int32) {
          size += Expr::getMinBytesForWidth(arguments[i]->getWidth());
        } else {
          size += llvm::RoundUpToAlignment(arguments[i]->getWidth(), 
                                           WordSize) / 8;
        }
      }

      MemoryObject *mo = sf.varargs = memory->allocate(size, true, false, false, 
                                                       state.tinfo.is_GPU_mode, state.getPrevPC()->inst);

      
      Gklee::Logging::outItem( *(mo->allocSite), "allocated args" );
      if (!mo) {
        terminateStateOnExecError(state, "out of memory (varargs)");
        symRace = true;
	Gklee::Logging::exitFunc();
        return;
      }
      ObjectState *os = bindObjectInState(state, mo, true);
      unsigned offset = 0;
      for (unsigned i = funcArgs; i < callingArgs; i++) {
        // FIXME: This is really specific to the architecture, not the pointer
        // size. This happens to work fir x86-32 and x86-64, however.
        Expr::Width WordSize = Context::get().getPointerWidth();
        if (WordSize == Expr::Int32) {
          os->write(offset, arguments[i]);
          offset += Expr::getMinBytesForWidth(arguments[i]->getWidth());
        } else {
          assert(WordSize == Expr::Int64 && "Unknown word size!");
          os->write(offset, arguments[i]);
          offset += llvm::RoundUpToAlignment(arguments[i]->getWidth(), 
                                             WordSize) / 8;
        }
      }
    }
    Gklee::Logging::outItem< std::string >( "list arg names" , "Binding arguments" );
    unsigned numFormals = f->arg_size();
    for (unsigned i=0; i<numFormals; ++i) 
      bindArgument(kf, i, state, arguments[i]);
  }
  Gklee::Logging::exitFunc();
}


void Executor::printFileLine(ExecutionState &state, KInstruction *ki) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  const InstructionInfo &ii = *ki->info;
  if (ii.file != "") 
    std::cerr << "     " << ii.file << ":" << ii.line << ":";
  else
    std::cerr << "     [no debug info]:";
  Gklee::Logging::exitFunc();
}


void Executor::updateStates(ExecutionState *current) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (searcher) {
    searcher->update(current, addedStates, removedStates);
    Gklee::Logging::outItem< std::string >( "updating with added and removed states" , "searcher" );
  }
  
  states.insert(addedStates.begin(), addedStates.end());
  addedStates.clear();
  
  for (std::set<ExecutionState*>::iterator
         it = removedStates.begin(), ie = removedStates.end();
       it != ie; ++it) {
    ExecutionState *es = *it;
    std::set<ExecutionState*>::iterator it2 = states.find(es);
    assert(it2!=states.end());
    states.erase(it2);
    std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(es);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    Gklee::Logging::outItem< std::string >( "" , "removing ptree node from current state" );
    processTree->remove(es->ptreeNode);
    delete es;
  }
  removedStates.clear();
  Gklee::Logging::exitFunc();
}

void Executor::contextSwitchToNextThread(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (!UseSymbolicConfig) {
    bool moveToNextWarp = false;
    bool hasDeadlock = false;
    bool newBI = false;
    state.tinfo.incTid(state.cTidSets, state.addressSpace.branchDivRegionSets, 
                       newBI, moveToNextWarp, hasDeadlock);
    if (hasDeadlock) {
      // This is an obvious deadlock occurring within the same warp ... 
      terminateStateOnExecError(state, "execution halts on a barrier mismatch, incurring a deadlock");
      if (Emacs) {
	const KInstruction *target = state.getPrevPC();
	if(MDNode *N = target->inst->getMetadata("dbg")){
	  DILocation Loc(N); 
	  std::cout << "emacs:dlbm:" << state.tinfo.get_cur_bid() << ":" <<
	    state.tinfo.get_cur_tid() << ":" << Loc.getDirectory().str() << 
	    "/" << Loc.getFilename().str() << ":" <<  Loc.getLineNumber() <<"::::"<< std::endl;
	}else{
	  std::cout << "problem with emacs:dlbm, inst->getMetaData didn't return location information" << std::endl;
	}
      }
    } else {
      if (moveToNextWarp) {
        // Denote that all threads all encounter the explicit __syncthread barrier 
        if (newBI) {
          concreteEnd = clock();
          double time = (double)(concreteEnd-concreteStart)/CLOCKS_PER_SEC;
          GKLEE_INFO << "In Barrier Interval " << state.BINum 
                     << ", Elapsed time for concrete configuration: " 
                     << time << std::endl;
          state.concreteTimeVec.push_back(time);
          concreteStart = clock();  
	  Gklee::Logging::outItem< std::string >( "" , "moving to next warp" );
          if (state.tinfo.allEndKernel) {
            kernelFunc = NULL;
            state.tinfo.is_GPU_mode = false;
            is_GPU_mode = false;
            state.addressSpace.clearAccessSet();
            state.addressSpace.clearInstAccessSet(true);
            state.clearCorrespondTidSets();
          } else {
            state.BINum++;
            state.moveToNextWarpAfterExplicitBarrier(true);
            state.restoreCorrespondTidSets();
          }
        } else {
          state.moveToNextWarpAfterExplicitBarrier(false);
        }
      }
    }
  } else { //symbolic config
    bool newBI = false;
    state.tinfo.incParametricFlow(state.cTidSets, state.getCurrentParaTree(), 
                                  newBI);
    Gklee::Logging::outItem< std::string >( "" , "inc parametric flow" );
    if (newBI) {
      symEnd = clock();
      double time = (double)(symEnd-symStart)/CLOCKS_PER_SEC;
      GKLEE_INFO << "In Barrier Interval " << state.BINum 
                 << ", Elapsed time for symbolic configuration (Parametric Flow): " 
                 << time << std::endl;
      state.symTimeVec.push_back(time);
      symStart = clock(); // initialize again...
      // All kernels finish ...
      if (state.tinfo.allEndKernel) {
        kernelFunc = NULL;
        state.tinfo.is_GPU_mode = false;
        is_GPU_mode = false;
        state.clearCorrespondTidSets();
      } else {
        state.BINum++;
        state.restoreCorrespondTidSets();
        if (!RacePrune)
          updateParaTreeSet(state);
        else{
	   //TODO flow experiment
          updateParaTreeSetUnderRacePrune(state);
	   //TODO flow experiment
	}
      }
    }
  }
  Gklee::Logging::exitFunc();
}

void Executor::bindModuleConstants() {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  for (std::vector<KFunction*>::iterator it = kmodule->functions.begin(),
         ie = kmodule->functions.end(); it != ie; ++it) {
    KFunction *kf = *it;
    for (unsigned i=0; i<kf->numInstructions; ++i)
      bindInstructionConstants(kf->instructions[i]);
  }

  kmodule->constantTable = new Cell[kmodule->constants.size()];
  for (unsigned i=0; i<kmodule->constants.size(); ++i) {
    Cell &c = kmodule->constantTable[i];
    c.value = evalConstant(kmodule->constants[i]);
  }
  Gklee::Logging::exitFunc();
}

static bool determineBranchType(Instruction *inst) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (inst->getOpcode() == Instruction::Br) {
    Gklee::Logging::exitFunc();
    return true;
  } else {
    Gklee::Logging::exitFunc();
    return false;
  } 
}

static klee::ref<Expr> constructInheritExpr(ExecutionState &state, 
                                      ParaTree &paraTree, klee::ref<Expr> &tdcCond) {

  Gklee::Logging::enterFunc( tdcCond , __PRETTY_FUNCTION__ );  
  ParaTreeNode *current = paraTree.getCurrentNode();
  klee::ref<Expr> expr;
  if (current == NULL) {
    unsigned tid = state.tinfo.get_cur_tid();
    expr = state.cTidSets[tid].inheritExpr; 
    tdcCond = state.cTidSets[tid].inheritExpr;
  } else {
    std::vector<ParaConfig> &configVec = current->successorConfigVec; 
    unsigned which = current->whichSuccessor;
    expr = AndExpr::create(current->inheritCond, configVec[which].cond);
    if (current->symBrType == TDC) {
      tdcCond = AndExpr::create(current->tdcCond, configVec[which].cond);
    } else {
      tdcCond = current->tdcCond;
    }
  }
  // Simplify the inherit condition 
  expr = state.constraints.simplifyExpr(expr);
  tdcCond = state.constraints.simplifyExpr(tdcCond);
  Gklee::Logging::outItem( expr , "return" );
  Gklee::Logging::exitFunc();
  return expr;
}

// Only used in parameterized implementation
// Fork a new parametric flow based on BDC (block-dependent conditional) 
// or TDC (thread-dependent conditional)
bool Executor::forkNewParametricFlow(ExecutionState &state, KInstruction *ki) {
  Instruction *i = ki->inst;
  Gklee::Logging::enterFunc( *i, __PRETTY_FUNCTION__ );  
  klee::ref<Expr> cond = eval(ki, 0, state).value;
  Gklee::Logging::outItem( cond , "new para flow cond" );
  //std::cout << "cond in forkNewSymbolicFlow: " << std::endl;
  //cond->dump();

  bool relatedToSym = false;
  bool accum = false;
  bool relatedToBuiltIn = identifyConditionType(state, cond, relatedToSym, accum);
  bool builtInFork = false;
 
  if (relatedToSym) {
    // Insert to the Parametric tree
    bool isCondBr = determineBranchType(i);
    llvm::BasicBlock *postDom = state.findNearestCommonPostDominator(postDominator, i, isCondBr); 
    ParaTree &pTree = state.getCurrentParaTree();
    klee::ref<Expr> tdcCond = 0;
    klee::ref<Expr> inheritCond = constructInheritExpr(state, pTree, tdcCond);
    ParaTreeNode *paraNode = new ParaTreeNode(i, postDom, SYM, isCondBr, 
                                              false, inheritCond, tdcCond);
    Gklee::Logging::outItem< std::string >( "related to sym" , "constructed new paraNode" );
    pTree.insertNodeIntoParaTree(paraNode);
    Logging::outItem< std::string >( std::to_string( state.tinfo.get_cur_bid() ) +
				     ":" + 
				     std::to_string( state.tinfo.get_cur_tid() ),
				     "paraConfig bid:tid" );
    ParaConfig config(state.tinfo.get_cur_bid(), 
                      state.tinfo.get_cur_tid(), 
                      cond, 0, 0);
    pTree.updateCurrentNodeOnNewConfig(config, SYM);
    builtInFork = true;
  } else {
    if (relatedToBuiltIn) { 
      // Pure TDC ...
      if (state.tinfo.sym_tdc_eval == 0) {
        ParaTree &pTree = state.getCurrentParaTree();
        bool isCondBr = determineBranchType(i);
        llvm::BasicBlock *postDom = state.findNearestCommonPostDominator(postDominator, i, isCondBr); 
        klee::ref<Expr> tdcCond = 0;
        klee::ref<Expr> inheritCond = constructInheritExpr(state, pTree, tdcCond);

        ParaTreeNode *paraNode = new ParaTreeNode(i, postDom, TDC, isCondBr, 
                                                  false, inheritCond, tdcCond);
	Gklee::Logging::outItem< std::string >( "related to builtin" , "constructed new paraNode" );
        pTree.insertNodeIntoParaTree(paraNode);
        state.tinfo.warpInBranch = true;
        // update two branches of BDC or TDC
        ParaTreeNode *curNode = pTree.getCurrentNode();
        state.paraConstraints = state.constraints;
        ExecutorUtil::constructSymConfigEncodedConstraint(state);
        ExecutorUtil::addConfigConstraint(state, tdcCond);
        bool result = false;
        bool success = solver->mustBeTrue(state, cond, result);
        klee::ref<Expr> trueExpr = ConstantExpr::create(1, Expr::Bool);

        if (success) {
	  Logging::outItem( cond, "new cond" );
          if (result) { // Only 'True' flow 
            GKLEE_INFO << "'True' path flow feasible !" << std::endl;
            state.tinfo.sym_tdc_eval = 1;
            ParaConfig config(state.tinfo.get_cur_bid(), 
                              state.tinfo.get_cur_tid(), 
                              trueExpr, 0, 0);
            pTree.updateCurrentNodeOnNewConfig(config, TDC);
            GKLEE_INFO << "'Else' path flow infeasible !" << std::endl;
          } else {
            success = solver->mayBeTrue(state, cond, result);
            if (result) { // Both 'True' and 'False' flows
              GKLEE_INFO << "'True' path flow feasible !" << std::endl;
              state.tinfo.sym_tdc_eval = 1;
              ParaConfig config(state.tinfo.get_cur_bid(), 
                                state.tinfo.get_cur_tid(), 
                                cond, 0, 0);
	      Logging::outItem< std::string >( std::to_string( state.tinfo.get_cur_bid() ) +
				     ":" + 
				     std::to_string( state.tinfo.get_cur_tid() ),
				     "paraConfig bid:tid" );
              pTree.updateCurrentNodeOnNewConfig(config, TDC);
              GKLEE_INFO << "'Else' path flow feasible !" << std::endl;
              klee::ref<Expr> negateExpr = Expr::createIsZero(cond);
              evaluateConstraintAsNewFlow(state, pTree, negateExpr, true);
            } else { // Only 'False' flow
              GKLEE_INFO << "'True' path flow infeasible !" << std::endl;
              GKLEE_INFO << "'Else' path flow feasible !" << std::endl;
              evaluateConstraintAsNewFlow(state, pTree, trueExpr, false);
            }
          }
        }

        // synchronize PCs when branch or switch instructions are encountered 
        state.tinfo.synchronizeBranchPCs(curNode);
        state.tinfo.synchronizeBarrierInfo(curNode);
        state.synchronizeBranchStacks(curNode);
        ExecutorUtil::copyBackConstraint(state);
      } else {
        state.tinfo.sym_tdc_eval++;
      }
      builtInFork = true;
    } else if (accum) {
      GKLEE_INFO << "Accumulative condition encountered!" << std::endl;
      cond->dump();
      bool isCondBr = determineBranchType(i);
      llvm::BasicBlock *postDom = state.findNearestCommonPostDominator(postDominator, i, isCondBr); 
      ParaTree &pTree = state.getCurrentParaTree();
      klee::ref<Expr> tdcCond = 0;
      klee::ref<Expr> inheritCond = constructInheritExpr(state, pTree, tdcCond);
      ParaTreeNode *paraNode = new ParaTreeNode(i, postDom, ACCUM, isCondBr, 
                                                false, inheritCond, tdcCond);
      pTree.insertNodeIntoParaTree(paraNode);
      ParaConfig config(state.tinfo.get_cur_bid(), 
                        state.tinfo.get_cur_tid(), 
                        cond, 0, 0);
      pTree.updateCurrentNodeOnNewConfig(config, ACCUM);
      builtInFork = true;
    }
  }
  Gklee::Logging::outItem( std::to_string( builtInFork ), "return" );
  Gklee::Logging::exitFunc();
  return builtInFork; 
}

bool Executor::forkNewParametricFlowUnderRacePrune(ExecutionState &state,
                                                   KInstruction *ki) {
  Instruction *i = ki->inst;
  Gklee::Logging::enterFunc( *i, __PRETTY_FUNCTION__ );  
  klee::ref<Expr> cond = eval(ki, 0, state).value;
  Logging::outItem( cond, "evaluating branch condition" );
  //std::cout << "cond in forkNewSymbolicFlow in [RacePrune]: " << std::endl;
  //cond->dump();

  bool relatedToSym = false;
  bool accum = false;
  bool relatedToBuiltIn = identifyConditionType(state, cond, relatedToSym, accum);
  bool builtInFork = false;
 
  if (relatedToSym) {
    // Insert to the Parametric tree
    bool isCondBr = determineBranchType(i);
    llvm::BasicBlock *postDom = state.findNearestCommonPostDominator(postDominator, i, isCondBr); 
    ParaTree &pTree = state.getCurrentParaTree();
    klee::ref<Expr> tdcCond = 0;
    klee::ref<Expr> inheritCond = constructInheritExpr(state, pTree, tdcCond);
    ParaTreeNode *paraNode = new ParaTreeNode(i, postDom, SYM, isCondBr, 
                                              false, inheritCond, tdcCond);
    pTree.insertNodeIntoParaTree(paraNode);
    ParaConfig config(state.tinfo.get_cur_bid(), 
                      state.tinfo.get_cur_tid(), 
                      cond, 0, 0);
    pTree.updateCurrentNodeOnNewConfig(config, SYM);
    builtInFork = true;
  } else {
    if (relatedToBuiltIn) { 
      // Pure TDC ...
      if (state.tinfo.sym_tdc_eval == 0) {
        ParaTree &pTree = state.getCurrentParaTree();
        bool isCondBr = determineBranchType(i);
        llvm::BasicBlock *postDom = state.findNearestCommonPostDominator(postDominator, i, isCondBr); 
        klee::ref<Expr> tdcCond = 0;
        klee::ref<Expr> inheritCond = constructInheritExpr(state, pTree, tdcCond);

        ParaTreeNode *paraNode = new ParaTreeNode(i, postDom, TDC, isCondBr, 
                                                  false, inheritCond, tdcCond);
        pTree.insertNodeIntoParaTree(paraNode);
        state.tinfo.warpInBranch = true;
        // update two branches of BDC or TDC
        ParaTreeNode *curNode = pTree.getCurrentNode();
        state.paraConstraints = state.constraints;
        ExecutorUtil::constructSymConfigEncodedConstraint(state);
        ExecutorUtil::addConfigConstraint(state, tdcCond);
        bool result = false;
        bool success = solver->mustBeTrue(state, cond, result);
        klee::ref<Expr> trueExpr = ConstantExpr::create(1, Expr::Bool);
	Logging::outItem< std::string >( "condition is pure TDC", "" );
        if (success) {
          if (result) { // Only 'True' flow 
	    Logging::outItem< std::string >( "condition evaluated to True", "");
            GKLEE_INFO << "'True' path flow feasible in [RacePrune] mode!" 
                       << std::endl;
            state.tinfo.sym_tdc_eval = 1;
            ParaConfig config(state.tinfo.get_cur_bid(), 
                              state.tinfo.get_cur_tid(), 
                              trueExpr, 0, 0);
            pTree.updateCurrentNodeOnNewConfig(config, TDC);
            GKLEE_INFO << "'Else' path flow infeasible in [RacePrune] mode!" 
                       << std::endl;
          } else {
            success = solver->mayBeTrue(state, cond, result);
            if (result) { // Both 'True' and 'False' flows
	      Logging::outItem< std::string >( "condition can be true/false", "");
              BranchInst *bi = cast<BranchInst>(i);
              GKLEE_INFO << "'True' path flow feasible in [RacePrune] mode!" 
                         << std::endl;
              state.tinfo.sym_tdc_eval = 1;
              ParaConfig config(state.tinfo.get_cur_bid(), 
                                state.tinfo.get_cur_tid(), 
                                cond, 0, 0);
              pTree.updateCurrentNodeOnNewConfig(config, TDC);
              if (bi->getMetadata("br-G-G")
                   || bi->getMetadata("br-G-S")) {
                // will contribute to the race detection across BIs
                state.cTidSets[state.tinfo.get_cur_tid()].keep = true;
		Logging::outItem( std::to_string( state.tinfo.get_cur_tid() ),
				  "branch G-G or G-S" );
              }
              GKLEE_INFO << "'Else' path flow feasible in [RacePrune] mode!" 
                         << std::endl;
              klee::ref<Expr> negateExpr = Expr::createIsZero(cond);
	      //	       //TODO flow experiement
              evaluateConstraintAsNewFlowUnderRacePrune(state, pTree, negateExpr, true, bi);
	      //	       //TODO flow experiment
            } else { // Only 'False' flow
              GKLEE_INFO << "'True' path flow infeasible in RacePrune mode!" 
                         << std::endl;
              GKLEE_INFO << "'Else' path flow feasible in RacePrune mode!" 
                         << std::endl;
              evaluateConstraintAsNewFlow(state, pTree, trueExpr, false);
            }
          }
        }

        // synchronize PCs when branch or switch instructions are encountered 
        state.tinfo.synchronizeBranchPCs(curNode);
        state.tinfo.synchronizeBarrierInfo(curNode);
        state.synchronizeBranchStacks(curNode);
        ExecutorUtil::copyBackConstraint(state);
      } else {
        state.tinfo.sym_tdc_eval++;
	Logging::outItem( std::to_string( state.tinfo.sym_tdc_eval ),
			  "sym_tdc_eval incremented" );
      }
      builtInFork = true;
    } else if (accum) {
      Logging::outItem< std::string >( "condition accum type", "" );
      cond->dump();
      bool isCondBr = determineBranchType(i);
      llvm::BasicBlock *postDom = state.findNearestCommonPostDominator(postDominator, i, isCondBr); 
      ParaTree &pTree = state.getCurrentParaTree();
      klee::ref<Expr> tdcCond = 0;
      klee::ref<Expr> inheritCond = constructInheritExpr(state, pTree, tdcCond);
      Logging::outItem( inheritCond, "accum inheritCond" );
      ParaTreeNode *paraNode = new ParaTreeNode(i, postDom, ACCUM, isCondBr, 
                                                false, inheritCond, tdcCond);
      pTree.insertNodeIntoParaTree(paraNode);
      ParaConfig config(state.tinfo.get_cur_bid(), 
                        state.tinfo.get_cur_tid(), 
                        cond, 0, 0);
      pTree.updateCurrentNodeOnNewConfig(config, ACCUM);
      builtInFork = true;
    }
  }
  Gklee::Logging::exitFunc();
  return builtInFork; 
}

static void constructSymInputVec(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  std::string bidArray = "bid_arr";
  std::string tidArray = "tid_arr";

  if (!state.symInputVec.empty()) state.symInputVec.clear();

  for (unsigned i = 0; i < state.symbolics.size(); i++) {
    const Array *tmpArray = state.symbolics[i].second; 
    std::string tmpName = tmpArray->name; 
    if (tmpName.find(bidArray) == string::npos && 
        tmpName.find(tidArray) == string::npos) {
      // Symbolics except for bid and tid ... 
      state.symInputVec.push_back(tmpName);
    }
  }
  Gklee::Logging::exitFunc();
}

void Executor::updateParaTreeSet(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  ParaTreeVec paraTreeVec;
  for (unsigned i = 0; i < state.cTidSets.size(); i++) {
    if (state.cTidSets[i].slotUsed) {
      if (i != 1) {
        paraTreeVec.push_back(ParaTree());
        state.tinfo.symParaTreeVec.push_back(i);
      }
    } else break;
  }
  state.getCurrentParaTreeSet().push_back(paraTreeVec);
  Gklee::Logging::exitFunc();
} 

void Executor::updateParaTreeSetUnderRacePrune(ExecutionState &state) {
  
  Gklee::Logging::enterFunc< std::string >( std::string("BI num:") +
					    std::to_string( state.BINum ), 
					    __PRETTY_FUNCTION__ );  
  ParaTreeVec paraTreeVec;
  bool firstNonKeep = false;
  unsigned nonKeep = 0;
  klee::ref<Expr> orExpr = ConstantExpr::create(1, Expr::Bool);
  for (unsigned i = 0; i < state.cTidSets.size(); i++) {
    if (i != 1) {
      if (state.cTidSets[i].slotUsed) {
        //std::cout << "slotUsed flow : " << i << std::endl;
        if (state.cTidSets[i].keep) {
	  Logging::outItem( std::to_string( i ),
			      "flow slotUsed keep true" );
          //std::cout << "keep flow : " << i << std::endl;
          paraTreeVec.push_back(ParaTree());
          state.tinfo.symParaTreeVec.push_back(i);
        } else {
	  Logging::outItem( std::to_string( i ),
			      "flow slotUsed keep false" );
          if (!firstNonKeep) {
            firstNonKeep = true;
            nonKeep = i;
            state.cTidSets[i].slotUsed = true;
	    Logging::outItem< std::string >( std::to_string( i ),
					     "marking slotUsed true in non-keep" );
            paraTreeVec.push_back(ParaTree());
            state.tinfo.symParaTreeVec.push_back(i);
            orExpr = state.cTidSets[i].inheritExpr;
          } else {
	    Logging::outItem( std::to_string( i ),
			      "flow slotUsed marked false -- Prune?" );
            state.cTidSets[i].slotUsed = false;
            orExpr = OrExpr::create(orExpr, state.cTidSets[i].inheritExpr);
          }
	} 
      } else break;
    }
  }

  if (firstNonKeep) { 
    orExpr = state.constraints.simplifyExpr(orExpr); 
    state.cTidSets[nonKeep].inheritExpr = orExpr; 
    Logging::outItem( orExpr, "flow merge or expression" );
  }

  if (paraTreeVec.size() == 0) {
    paraTreeVec.push_back(ParaTree());
    state.tinfo.symParaTreeVec.push_back(0);
    klee::ref<Expr> cond = ConstantExpr::create(1, Expr::Bool);
    state.cTidSets[0].inheritExpr = cond;
    Logging::outItem< std::string >( "Empty paraTreeVect, creating new", "" );
  }
  state.getCurrentParaTreeSet().push_back(paraTreeVec);
  Gklee::Logging::exitFunc();
  
}

void Executor::handleEnterGPUMode(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  GKLEE_INFO2 << "Start executing a GPU kernel \n\n";
  state.tinfo.kernel_call = false;
  state.tinfo.is_GPU_mode = true;
  state.tinfo.allEndKernel = false;
  state.tinfo.warpInBranch = false;
  is_GPU_mode = true;
  if (CheckBarrierRedundant)
    UseSymbolicConfig = true;
  
  state.incKernelNum();
  state.BINum = 1;
  Gklee::Logging::outItem( std::string( "BI#:1, Kern#:" ) +
			   std::to_string( state.kernelNum ),
			   "inc kernel# and set BINum" );
  // handle 'extern __shared__' case ...
  if (state.maxKernelSharedSize > 0) {
    initializeExternalSharedGlobals(state); 
  }
  // now synchronize the PCs of all the threads
  // the stacks of each thread should be equal to that of thread 0
  state.tinfo.synchronizePCs();
  Gklee::Logging::outItem< std::string >( "PCs and stacks" , "Synch threads" );
  Gklee::Logging::outItem< std::string >( "Creating stack for each thread" , 
					  std::to_string( state.tinfo.get_num_threads() ) +
					  " threads" );
  for (unsigned i = 1; i < state.tinfo.get_num_threads(); i++)
    state.stacks[i] = state.stacks[0];

  // now set up the per thread coverage information
  bc_cov_monitor.initPerThreadCov();

  // set the corresponding tid sets...
  state.setCorrespondTidSets();

  Gklee::Logging::outItem( std::string( "num blocks: " ) + 
			   std::to_string( GPUConfig::num_blocks ),
			   std::string( "constructing shared memory" ));
  // Then construct the shared memory region for each block... 
  for (unsigned bid = 0; bid < GPUConfig::num_blocks; bid++)
    constructSharedMemory(state, bid);

  concreteStart = clock();

  if (UseSymbolicConfig) {
    state.tinfo.symExecuteSet.clear();
    state.tinfo.symParaTreeVec.clear();
    ParaTreeSet set;
    state.paraTreeSets.push_back(set);

    state.cTidSets[0].slotUsed = true;
    state.cTidSets[1].slotUsed = true;
    Logging::outItem< std::string >( "0:1", "initializing two flows (cTidSets)" );
    // create a para tree vec;
    state.tinfo.symExecuteSet.push_back(0); 
    updateParaTreeSet(state);
    constructSymInputVec(state);
    symStart = clock();
    Gklee::Logging::outItem(std::string( "symExecuteSet, ") +
			    "symParaTreeVec, push new paraTreeSet, " +
			    "push empty to state.tinfo.symExecuteSet",
			    "Clearing state" );
  }
  Gklee::Logging::exitFunc();
}

void Executor::updateConstantTable(unsigned kernelNum) {
  // update the constant table according to the externSharedSet 
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  unsigned i = 0;
  for (; i < externSharedSet.size(); i++) {
    ExternSharedVar &var = externSharedSet[i][0];
    if (var.kernelNum == kernelNum)
      break;
  }
 
  if (i != externSharedSet.size()) {
    ExternSharedVarVec &vec = externSharedSet[i];
    for (unsigned j = 0; j < vec.size(); j++) {
      unsigned tableNum = vec[j].tableNum;
      Cell &c = kmodule->constantTable[tableNum];
      c.value = vec[j].externSharedMO->getBaseExpr(); 
    }
  }
  Gklee::Logging::exitFunc();
}

static std::string strip(std::string &in) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  unsigned len = in.size();
  unsigned lead = 0, trail = len;
  while (lead<len && isspace(in[lead]))
    ++lead;
  while (trail>lead && isspace(in[trail-1]))
    --trail;
  auto subs = in.substr(lead, trail-lead);
  Gklee::Logging::exitFunc();
  return subs;
}

 void Executor::configurateGPUKernelSet() {     
   //  const char* c_file = "kernelSet.txt";                                                              
   Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
   DIR* dir = opendir(".");
   while( dir ){
     struct dirent* di = readdir( dir );
     if(!di) break;
     std::string dname(di->d_name);
     if( dname.find( "kernelSet.txt" ) != std::string::npos ){
       std::ifstream f( dname.c_str() );                                                          
       assert(f.is_open() && std::string("unable to open " + dname + " file" ).c_str());                      
       while (!f.eof()) {
	 std::string line;
	 std::getline(f, line);
	 line = strip(line);                                                           
	 if (!line.empty())
	   Gklee::Logging::outItem( line, dname + " item" );
	   kernelSet.insert(line);                                         
       } 
       f.close();
     }
   }
   Gklee::Logging::exitFunc();
 }

void Executor::run(ExecutionState &initialState) {

  Gklee::Logging::enterFunc( initialState.getPC()->info->file , __PRETTY_FUNCTION__ );
   
  bindModuleConstants();

  configurateGPUKernelSet();

  // Delay init till now so that ticks don't accrue during
  // optimization and such.
  initTimers();

  if (ReducePath.size() > 0)
    PR_info.init(ReducePath, PRUseDep);

  states.insert(&initialState);

  if (usingSeeds) {
    std::vector<SeedInfo> &v = seedMap[&initialState];
    
    for (std::vector<KTest*>::const_iterator it = usingSeeds->begin(), 
           ie = usingSeeds->end(); it != ie; ++it)
      v.push_back(SeedInfo(*it));

    int lastNumSeeds = usingSeeds->size()+10;
    double lastTime, startTime = lastTime = util::getWallTime();
    ExecutionState *lastState = 0;
    while (!seedMap.empty()) {
      if (haltExecution) goto dump;

      std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it = 
        seedMap.upper_bound(lastState);
      if (it == seedMap.end())
        it = seedMap.begin();
      lastState = it->first;
      unsigned numSeeds = it->second.size();

      ExecutionState &state = *lastState;
      KInstruction *ki = state.getPC();
      stepInstruction(state);

      executeInstruction(state, ki);
      processTimers(&state, MaxInstructionTime * numSeeds);
      updateStates(&state);

      if ((stats::instructions % 1000) == 0) {
        int numSeeds = 0, numStates = 0;
        for (std::map<ExecutionState*, std::vector<SeedInfo> >::iterator
               it = seedMap.begin(), ie = seedMap.end();
             it != ie; ++it) {
          numSeeds += it->second.size();
          numStates++;
        }
        double time = util::getWallTime();
        if (SeedTime>0. && time > startTime + SeedTime) {
          klee_warning("seed time expired, %d seeds remain over %d states",
                       numSeeds, numStates);
          break;
        } else if (numSeeds<=lastNumSeeds-10 ||
                   time >= lastTime+10) {
          lastTime = time;
          lastNumSeeds = numSeeds;          
          klee_message("%d seeds remaining over: %d states", 
                       numSeeds, numStates);
        }
      }
    }

    klee_message("seeding done (%d states remain)", (int) states.size());

    // XXX total hack, just because I like non uniform better but want
    // seed results to be equally weighted.
    for (std::set<ExecutionState*>::iterator
           it = states.begin(), ie = states.end();
         it != ie; ++it) {
      (*it)->weight = 1.;
    }

    if (OnlySeed)
      goto dump;
  }

  searcher = constructUserSearcher(*this);

  searcher->update(0, states, std::set<ExecutionState*>());
  while (!states.empty() && !haltExecution) {
  ExecutionState &state = searcher->selectState();
    // update the constant table 
    if (state.tinfo.is_GPU_mode 
         && externSharedSet.size() > 0) {
      unsigned size = externSharedSet.size();
      ExternSharedVar &var = externSharedSet[size-1][0]; 
      if (var.kernelNum > state.kernelNum)
        updateConstantTable(state.kernelNum);  
    }

    KInstruction *ki = state.getPC();
    if (UseSymbolicConfig && state.tinfo.is_GPU_mode) {
      if (ExecutorUtil::isForkInstruction(ki->inst)) {
	 //TODO this is for flow study
	
        if (!RacePrune)
          state.tinfo.builtInFork = forkNewParametricFlow(state, ki);
        else 
          state.tinfo.builtInFork = forkNewParametricFlowUnderRacePrune(state, ki); 
	 //TODO this is for flow study
	
      }
    }
    stepInstruction(state);
    executeInstruction(state, ki);
    if (state.tinfo.just_enter_GPU_mode)
      handleEnterGPUMode(state);

    if (state.tinfo.is_GPU_mode) {
      if (!UseSymbolicConfig) {
        if (SimdSchedule) {
          if (!state.tinfo.just_enter_GPU_mode) {
            if (!kernelFunc)
              kernelFunc = ki->inst->getParent()->getParent(); 
            // Context switch to next thread 
            contextSwitchToNextThread(state);
            for (std::set<ExecutionState*>::iterator si = addedStates.begin(); 
                 si != addedStates.end(); si++) {
              contextSwitchToNextThread(**si);
            }
          } else state.tinfo.just_enter_GPU_mode = false;
        } else { // Pure Canonical Schedule
          if (!state.tinfo.just_enter_GPU_mode) {
            if (!kernelFunc)
              kernelFunc = ki->inst->getParent()->getParent();
            // If all threads end
            if (state.tinfo.allEndKernel) {
              kernelFunc = NULL;
              state.tinfo.is_GPU_mode = false;
              is_GPU_mode = false;
              state.addressSpace.clearAccessSet();
              state.addressSpace.clearInstAccessSet(true);
              state.clearCorrespondTidSets();
            }
          } else state.tinfo.just_enter_GPU_mode = false;
        }
      } else { //SYMBOLIC CONFIG!
        if (!state.tinfo.just_enter_GPU_mode) {
          if (!kernelFunc) {
            kernelFunc = ki->inst->getParent()->getParent(); 
          }
          // Context switch to next thread 
          contextSwitchToNextThread(state);
          for (std::set<ExecutionState*>::iterator si = addedStates.begin(); 
               si != addedStates.end(); si++) {
            contextSwitchToNextThread(**si);
          }
        } else state.tinfo.just_enter_GPU_mode = false;
      }
    }

    processTimers(&state, MaxInstructionTime);

    if (MaxMemory) {
      if ((stats::instructions & 0xFFFF) == 0) {
        // We need to avoid calling GetMallocUsage() often because it
        // is O(elts on freelist). This is really bad since we start
        // to pummel the freelist once we hit the memory cap.
        unsigned mbs = sys::Process::GetTotalMemoryUsage() >> 20;
        
        if (mbs > MaxMemory) {
          if (mbs > MaxMemory + 100) {
            // just guess at how many to kill
            unsigned numStates = states.size();
            unsigned toKill = std::max(1U, numStates - numStates*MaxMemory/mbs);

            if (MaxMemoryInhibit)
              klee_warning("killing %d states (over memory cap)",
                           toKill);

            std::vector<ExecutionState*> arr(states.begin(), states.end());
            for (unsigned i=0,N=arr.size(); N && i<toKill; ++i,--N) {
              unsigned idx = rand() % N;

              // Make two pulls to try and not hit a state that
              // covered new code.
              if (arr[idx]->coveredNew)
                idx = rand() % N;

              std::swap(arr[idx], arr[N-1]);
              terminateStateEarly(*arr[N-1], "memory limit");
            }
          }
          atMemoryLimit = true;
        } else {
          atMemoryLimit = false;
        }
      }
    }

    updateStates(&state);
  }

  delete searcher;
  searcher = 0;
  
 dump:
  if (DumpStatesOnHalt && !states.empty()) {
    std::cerr << "KLEE: halting execution, dumping remaining states\n";
    for (std::set<ExecutionState*>::iterator
           it = states.begin(), ie = states.end();
         it != ie; ++it) {
      ExecutionState &state = **it;
      stepInstruction(state); // keep stats rolling
      terminateStateEarly(state, "execution halting");
    }
    updateStates(0);
  }
  Gklee::Logging::exitFunc();
}

std::string Executor::getAddressInfo(ExecutionState &state, 
                                     klee::ref<Expr> address) const{
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  std::ostringstream info;
  info << "\taddress: " << address << "\n";
  uint64_t example;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(address)) {
    example = CE->getZExtValue();
  } else {
    klee::ref<ConstantExpr> value;
    ExecutorUtil::copyOutConstraintUnderSymbolic(state);
    bool success = solver->getValue(state, address, value);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    ExecutorUtil::copyBackConstraintUnderSymbolic(state);
    example = value->getZExtValue();
    info << "\texample: " << example << "\n";
    std::pair< klee::ref<Expr>, klee::ref<Expr> > res = solver->getRange(state, address);
    info << "\trange: [" << res.first << ", " << res.second <<"]\n";
  }
  
  MemoryObject hack((unsigned) example);    
  // by Guodong
  MemoryMap::iterator lower = state.addressSpace.getAddressSpace().objects.upper_bound(&hack);
  info << "\tnext: ";
  if (lower==state.addressSpace.getAddressSpace().objects.end()) {
    info << "none\n";
  } else {
    const MemoryObject *mo = lower->first;
    std::string alloc_info;
    mo->getAllocInfo(alloc_info);
    info << "object at " << mo->address
         << " of size " << mo->size << "\n"
         << "\t\t" << alloc_info << "\n";
  }
  if (lower!=state.addressSpace.getAddressSpace().objects.begin()) {
    --lower;
    info << "\tprev: ";
    if (lower==state.addressSpace.getAddressSpace().objects.end()) {
      info << "none\n";
    } else {
      const MemoryObject *mo = lower->first;
      std::string alloc_info;
      mo->getAllocInfo(alloc_info);
      info << "object at " << mo->address 
           << " of size " << mo->size << "\n"
           << "\t\t" << alloc_info << "\n";
    }
  }
  auto st = info.str();
  Gklee::Logging::exitFunc();
  return st;
}

void Executor::terminateState(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (replayOut && replayPosition!=replayOut->numObjects) {
    klee_warning_once(replayOut, 
                      "replay did not consume all objects in test input.");
  }

  traceInfo.empty();
  interpreterHandler->incPathsExplored();
  
  std::set<ExecutionState*>::iterator it = addedStates.find(&state);
  if (it==addedStates.end()) {
    state.setPC(state.getPrevPC());

    removedStates.insert(&state);
  } else {
    // never reached searcher, just delete immediately
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(&state);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    addedStates.erase(it);
    processTree->remove(state.ptreeNode);
    delete &state;
  }
  Gklee::Logging::exitFunc();
}

void Executor::concludeExploredTime(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  double totalTime = 0.0;
  if (UseSymbolicConfig) {
    unsigned pathNum = interpreterHandler->getNumPathsExplored();
    pathNum++;
    std::cout << "path num explored here (symbolic config): " 
              << pathNum << std::endl; 

    unsigned i = (state.forkStateBINum > 0) ? state.forkStateBINum-1 : 0;
    for (; i < state.symTimeVec.size(); i++) {
      std::cout << "<BI: " << i+1 << ", Time: " << state.symTimeVec[i] << ">"
                << std::endl;
      totalTime += state.symTimeVec[i];
    }
    std::cout << "Under symbolic configuration, Across " 
              << i << " BIs, Total Time: " << totalTime << std::endl;
    symTotalTime += totalTime;
    std::cout << "After exploring " << pathNum 
              << " paths, the average exploration time (symbolic) is " 
              << symTotalTime / pathNum << "s"
              << ", the total execution time: " 
              << symTotalTime << "s" << std::endl;
     
    if (symBC) {
      std::cout << "********** Bank Conflict found **********" << std::endl;
    } else {
      std::cout << "********** No Bank Conflict found **********" << std::endl;
    } 

    if (symMC) {
      std::cout << "********** Memory Coalescing found **********" << std::endl;
    } else {
      std::cout << "********** No Memory Coalescing found **********" << std::endl;
    }

    if (symWD) {
      std::cout << "********** Warp Divergence found **********" << std::endl;
    } else {
      std::cout << "********** No Warp Divergence found **********" << std::endl;
    }

    if (symVMiss) {
      std::cout << "********** Volatile Missed found **********" << std::endl;
    } else {
      std::cout << "********** No Volatile Missed found *********" << std::endl;
    }
  
    if (symRace) {
      std::cout << "********** Race found **********" << std::endl;
    } else {
      std::cout << "********** No Race found **********" << std::endl;
    }
  } else {
    unsigned pathNum = interpreterHandler->getNumPathsExplored();
    pathNum++;
    unsigned i = 0;
    for (; i < state.concreteTimeVec.size(); i++) {
      std::cout << "<BI: " << i+1 << ", Time: " << state.concreteTimeVec[i] << ">"
                << std::endl;
      totalTime += state.concreteTimeVec[i];
    }
    std::cout << "Under concrete configuration, Acorss " 
              << i << " BIs, Total Time: " << totalTime << std::endl;
    concreteTotalTime += totalTime;
    std::cout << "After exploring " << pathNum 
              << " paths, the average exploration time (concrete) is " 
              << concreteTotalTime / pathNum << std::endl; 
  }
  Gklee::Logging::exitFunc();
}

void Executor::concludeRateStatistics(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  unsigned pathNum = interpreterHandler->getNumPathsExplored();
  pathNum++;
  std::cout << "path num explored here: " << pathNum << std::endl; 
  if (CheckBC) {
    if (!Emacs) std::cout << "+++++++++++++++++ Bank Conflict Rate: +++++++++++++++++" << std::endl;
    state.addressSpace.getBCRate(bcWarpNum, bcWarpSum, bcBINum, bcBISum); 
    
    unsigned avgWarpNum = bcWarpNum / pathNum; 
    unsigned avgWarpSum = bcWarpSum / pathNum; 
    
    unsigned avgBINum = bcBINum / pathNum; 
    unsigned avgBISum = bcBISum / pathNum; 

    unsigned warpRate = bcWarpSum?(bcWarpNum * 100)/bcWarpSum:0;  
    unsigned biRate = bcBISum?(bcBINum * 100)/bcBISum:0;
    if(Emacs){
      std::cout << "BC:" << pathNum << ":" << 
	warpRate << ":" << avgWarpNum << ":" << avgWarpSum << ":" << 
	biRate << ":" << avgBINum << ":" << avgBISum << std::endl;
    }else{
      if (bcWarpSum) {
	GKLEE_INFO2 << "The Average 'Warp' Bank Conflict Rate for all shared memories at path " 
		    << pathNum << " : " << warpRate << "%" << ", <avgBCWarp, avgWarp> : " << "<" 
		    << avgWarpNum << ", " << avgWarpSum << ">\n";
      }
      if (bcBISum) {
	GKLEE_INFO2 << "The Average 'BI' Bank Conflict Rate for all shared memories at path " 
		    << pathNum << " : " << biRate << "%" << ", <avgBCBI, avgBI> : " << "<" 
		    << avgBINum << ", " << avgBISum << ">\n";
      }
      std::cout << "+++++++++++++++++ end +++++++++++++++++" << std::endl;
    }
  }
  if (CheckMC) {
    // mcRate means the non-coalescing rate...
    if(!Emacs) std::cout << "+++++++++++++++++ Memory Coalescing Rate: +++++++++++++++++" << std::endl;
    state.addressSpace.getMCRate(mcWarpNum, mcWarpSum, mcBINum, mcBISum); 

    unsigned avgWarpNum = mcWarpNum / pathNum; 
    unsigned avgWarpSum = mcWarpSum / pathNum; 
    
    unsigned avgBINum = mcBINum / pathNum; 
    unsigned avgBISum = mcBISum / pathNum; 

    unsigned warpRate = mcWarpSum?(mcWarpNum * 100)/mcWarpSum:0;  
    unsigned biRate = mcBISum?(mcBINum * 100)/mcBISum:0; 
    if(Emacs){
      std::cout << "MC:" << pathNum << ":" << 
	warpRate << ":" << avgWarpNum << ":" << avgWarpSum << ":" <<
	biRate << ":" << avgBINum << ":" << avgBISum << std::endl;
    }else{
      if (mcWarpSum) {
	GKLEE_INFO2 << "The Average 'Warp' Memory Coalescing Rate at path " 
		    << pathNum << " : " << warpRate << "%" << ", <avgMCWarp, avgWarp> : " << "<" 
		    << avgWarpNum << ", " << avgWarpSum << ">\n";
      }
      if (mcBISum) {
	GKLEE_INFO2 << "The Average 'BI' Memory Coalescing Rate at path " 
		    << pathNum << " : " << biRate << "%" << ", <avgMCBI, avgBI> : " << "<" 
		    << avgBINum << ", " << avgBISum << ">\n";
      }
      std::cout << "+++++++++++++++++ end +++++++++++++++++" << std::endl;
    }
  }
  if (CheckWD) {
    if(!Emacs) std::cout << "+++++++++++++++++ Warp Divergence Rate: +++++++++++++++++" << std::endl;
    state.addressSpace.getWDRate(wdWarpNum, wdWarpSum, wdBINum, wdBISum); 

    unsigned avgWarpNum = wdWarpNum / pathNum; 
    unsigned avgWarpSum = wdWarpSum / pathNum; 
    
    unsigned avgBINum = wdBINum / pathNum; 
    unsigned avgBISum = wdBISum / pathNum; 

    unsigned warpRate = wdWarpSum?(wdWarpNum * 100)/wdWarpSum:0;  
    unsigned biRate = wdBISum?(wdBINum * 100)/wdBISum:0; 
    if(Emacs){
      std::cout << "WD:" << pathNum << ":" <<
	warpRate << ":" << avgWarpNum << ":" << avgWarpSum << ":" <<
	biRate << ":" << avgBINum << ":" << avgBISum << std::endl;
    }else{
      if (wdWarpSum) {
	GKLEE_INFO2 << "The Average 'Warp' Warp Divergence Rate at path " 
		    << pathNum << " : " << warpRate << "%" << ", <avgWDWarp, avgWarp> : " << "<" 
		    << avgWarpNum << ", " << avgWarpSum << ">\n";
      }
      if (wdBISum) {
	GKLEE_INFO2 << "The Average 'BI' Warp Divergence Rate at path " 
		    << pathNum << " : " << biRate << "%" << ", <avgWDBI, avgBI> : " << "<" 
		    << avgBINum << ", " << avgBISum << ">\n";
      }
      std::cout << "+++++++++++++++++ end +++++++++++++++++" << std::endl;
    }
  }
  //if (CheckRace) {
  //  state.addressSpace.getRaceRate();
  //}
  Gklee::Logging::exitFunc();
}

void Executor::processPerformDefectTestCase(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (!PerformTest){
    Gklee::Logging::exitFunc();
    return;
  }
  if (state.addressSpace.hasBC)
    executeBankConflict(state, state.addressSpace.bcCondComb);

  if (state.addressSpace.hasNoMC)
    executeNoMemoryCoalescing(state, state.addressSpace.nonMCCondComb);

  if (state.addressSpace.hasVM)
    executeVolatileMissing(state, state.addressSpace.vmCondComb);
  Gklee::Logging::exitFunc();
}

void Executor::terminateStateEarly(ExecutionState &state, 
                                   const Twine &message) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (!OnlyOutputStatesCoveringNew || state.coveredNew ||
      (AlwaysOutputSeeds && seedMap.count(&state))) {
    interpreterHandler->processTestCase(state, (message + "\n").str().c_str(),
                                          "early");
    processPerformDefectTestCase(state);
  }
  if (!UseSymbolicConfig)
    concludeRateStatistics(state);
  concludeExploredTime(state);
  terminateState(state);
  Gklee::Logging::exitFunc();
}

void Executor::terminateStateOnExit(ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (!OnlyOutputStatesCoveringNew || state.coveredNew || 
      (AlwaysOutputSeeds && seedMap.count(&state))) {
    interpreterHandler->processTestCase(state, 0, 0);
    processPerformDefectTestCase(state);

    // by Guodong; print out the path condition
    static int test_index = 1;
    if (PrintCondition) {
      std::cout << "Test " << test_index++ << "'s path condition: \n";
      for (ConstraintManager::const_iterator ii = state.constraints.begin();
	   ii != state.constraints.end(); ii++)
	(*ii)->dump();
    }
  }
  if (!UseSymbolicConfig)
    concludeRateStatistics(state);
  concludeExploredTime(state);
  terminateState(state);
  Gklee::Logging::exitFunc();
}

void Executor::terminateStateOnError(ExecutionState &state,
                                     const llvm::Twine &messaget,
                                     const char *suffix,
                                     const llvm::Twine &info) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  std::string message = messaget.str();
  static std::set< std::pair<Instruction*, std::string> > emittedErrors;
  const InstructionInfo &ii = *state.getPrevPC()->info;
  
  if (EmitAllErrors ||
      emittedErrors.insert(std::make_pair(state.getPrevPC()->inst, message)).second) {
    if (ii.file != "") {
      klee_message("ERROR: %s:%d: %s", ii.file.c_str(), ii.line, message.c_str());
    } else {
      klee_message("ERROR: %s", message.c_str());
    }
    if (!EmitAllErrors)
      klee_message("NOTE: now ignoring this error at this location");
    
    std::ostringstream msg;
    msg << "Error: " << message << "\n";
    if (ii.file != "") {
      msg << "File: " << ii.file << "\n";
      msg << "Line: " << ii.line << "\n";
    }
    msg << "Stack: \n";
    unsigned idx = 0;
    const KInstruction *target = state.getPrevPC();
    for (ExecutionState::stack_ty::reverse_iterator
           it = state.getCurStack().rbegin(), ie = state.getCurStack().rend();
         it != ie; ++it) {
      StackFrame &sf = *it;
      Function *f = sf.kf->function;
      const InstructionInfo &ii = *target->info;
      msg << "\t#" << idx++ 
          << " " << std::setw(8) << std::setfill('0') << ii.assemblyLine
          << " in " << f->getName().str() << " (";
      // Yawn, we could go up and print varargs if we wanted to.
      unsigned index = 0;
      for (Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
           ai != ae; ++ai) {
        if (ai!=f->arg_begin()) msg << ", ";

        msg << ai->getName().str();
        // XXX should go through function
        klee::ref<Expr> value = sf.locals[sf.kf->getArgRegister(index++)].value; 
        if (isa<ConstantExpr>(value))
          msg << "=" << value;
      }
      msg << ")";
      if (ii.file != "")
        msg << " at " << ii.file << ":" << ii.line;
      msg << "\n";
      target = sf.caller;
    }

    std::string info_str = info.str();
    if (info_str != "")
      msg << "Info: \n" << info_str;
    interpreterHandler->processTestCase(state, msg.str().c_str(), suffix);
    processPerformDefectTestCase(state);
  }
  if (!UseSymbolicConfig)
    concludeRateStatistics(state);
  concludeExploredTime(state);
  terminateState(state);
  Gklee::Logging::exitFunc();
}

// XXX shoot me
static const char *okExternalsList[] = { "printf", 
                                         "fprintf", 
                                         "puts",
                                         "getpid" };
static std::set<std::string> okExternals(okExternalsList,
                                         okExternalsList + 
                                         (sizeof(okExternalsList)/sizeof(okExternalsList[0])));

void Executor::callExternalFunction(ExecutionState &state,
                                    KInstruction *target,
                                    Function *function,
                                    std::vector< klee::ref<Expr> > &arguments) {

  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  // check if specialFunctionHandler wants it
  if (specialFunctionHandler->handle(state, function, target, arguments)){
    Gklee::Logging::exitFunc();
    return;
  }
  if (NoExternals && !okExternals.count(function->getName())) {
    std::cerr << "KLEE:ERROR: Calling not-OK external function : " 
               << function->getName().str() << "\n";
    terminateStateOnError(state, "externals disallowed", "user.err");
    Gklee::Logging::exitFunc();
    return;
  }

  // normal external function handling path
  // allocate 128 bits for each argument (+return value) to support fp80's;
  // we could iterate through all the arguments first and determine the exact
  // size we need, but this is faster, and the memory usage isn't significant.
  uint64_t *args = (uint64_t*) alloca(2*sizeof(*args) * (arguments.size() + 1));
  memset(args, 0, 2 * sizeof(*args) * (arguments.size() + 1));
  unsigned wordIndex = 2;
  for (std::vector<klee::ref<Expr> >::iterator ai = arguments.begin(), 
       ae = arguments.end(); ai!=ae; ++ai) {
    if (AllowExternalSymCalls) { // don't bother checking uniqueness
      klee::ref<ConstantExpr> ce;
      ExecutorUtil::copyOutConstraintUnderSymbolic(state);
      bool success = solver->getValue(state, *ai, ce);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      ExecutorUtil::copyBackConstraintUnderSymbolic(state);
      ce->toMemory(&args[wordIndex]);
      wordIndex += (ce->getWidth()+63)/64;
    } else {
      klee::ref<Expr> arg = toUnique(state, *ai);

      if (ConstantExpr *ce = dyn_cast<ConstantExpr>(arg)) {
        // XXX kick toMemory functions from here
        ce->toMemory(&args[wordIndex]);
        wordIndex += (ce->getWidth()+63)/64;
      } else if (function->getName() == "printf") {
	klee_warning("attempt to print symbolic argument!\n");
	// arguments[1]->dump();
      }
      else {
        std::string reason = "The arguments for external function: ";
        reason += function->getName().str();
        arg = toConstantArguments(state, *ai, reason.data());
        if (ConstantExpr *ce = dyn_cast<ConstantExpr>(arg)) {
          ce->toMemory(&args[wordIndex]);
          wordIndex += (ce->getWidth()+63)/64;
        }
        //terminateStateOnExecError(state, 
        //                          "external call with symbolic argument: " + 
        //                          function->getName());
        //return;
      }
    }
  }

  // state.addressSpace.getAddressSpace(GPUConfig::HOST).copyOutConcretes();
  state.addressSpace.copyOutConcretes(state.tinfo.get_cur_tid());

  if (!SuppressExternalWarnings) {
    std::ostringstream os;
    os << "calling external: " << function->getName().str() << "(";
    for (unsigned i=0; i<arguments.size(); i++) {
      //os << arguments[i];
      if (i != arguments.size()-1)
	os << ", ";
    }
    os << ")";
    
    if (AllExternalWarnings)
      klee_warning("%s", os.str().c_str());
    else
      klee_warning_once(function, "%s", os.str().c_str());
  }
  
  bool success = externalDispatcher->executeCall(function, target->inst, args);
  if (!success) {
    terminateStateOnError(state, "failed external call: " + function->getName(),
                          "external.err");
    Gklee::Logging::exitFunc();
    return;
  }

  if (!state.addressSpace.copyInConcretes(state.tinfo.get_cur_tid())) {
    terminateStateOnError(state, "external modified read-only object",
                          "external.err");
    Gklee::Logging::exitFunc();
    return;
  }

  LLVM_TYPE_Q Type *resultType = target->inst->getType();
  if (resultType != Type::getVoidTy(getGlobalContext())) {
    klee::ref<Expr> e = ConstantExpr::fromMemory((void*) args, 
                                           getWidthForLLVMType(resultType));
    bindLocal(target, state, e);
  }
  Gklee::Logging::exitFunc();
}

/***/

klee::ref<Expr> Executor::replaceReadWithSymbolic(ExecutionState &state, 
                                            klee::ref<Expr> e) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  unsigned n = interpreterOpts.MakeConcreteSymbolic;
  if (!n || replayOut || replayPath){
    Gklee::Logging::exitFunc();
    return e;
  }
  // right now, we don't replace symbolics (is there any reason too?)
  if (!isa<ConstantExpr>(e)){
    Gklee::Logging::exitFunc(); 
    return e;
  }

  if (n != 1 && random() %  n){
    Gklee::Logging::exitFunc();
    return e;
  }
  // create a new fresh location, assert it is equal to concrete value in e
  // and return it.
  
  static unsigned id;
  const Array *array = new Array("rrws_arr" + llvm::utostr(++id), 
                                 Expr::getMinBytesForWidth(e->getWidth()));
  klee::ref<Expr> res = Expr::createTempRead(array, e->getWidth());
  klee::ref<Expr> eq = NotOptimizedExpr::create(EqExpr::create(e, res));
  std::cerr << "Making symbolic: " << eq << "\n";
  state.addConstraint(eq);
  Gklee::Logging::exitFunc();
  return res;
}

void Executor::executeMakeSymbolic(ExecutionState &state, 
                                   const MemoryObject *mo,
                                   const std::string &name) {

  Gklee::Logging::enterFunc< std::string >( name, __PRETTY_FUNCTION__ );  
  // Create a new object state for the memory object (instead of a copy).
  if (!replayOut) {
    // Find a unique name for this array.  First try the original name,
    // or if that fails try adding a unique identifier.
    unsigned id = 0;
    std::string uniqueName = name;
    while (!state.arrayNames.insert(uniqueName).second) {
      uniqueName = name + "_" + llvm::utostr(++id);
    }
    const Array *array = new Array(uniqueName, mo->size);
    if (GPUConfig::verbose > 0) {
      std::string result;
      mo->getAllocInfo(result);
      //std::cout << "The basic info: " << result << std::endl;
      //std::cout << "The executeMakeSymbolic, array info: " << array->name
      //          << " , " << array->size << std::endl;
    }
    bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);

    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
      seedMap.find(&state);
    if (it!=seedMap.end()) { // In seed mode we need to add this as a
                             // binding.
      for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
             siie = it->second.end(); siit != siie; ++siit) {
        SeedInfo &si = *siit;
        KTestObject *obj = si.getNextInput(mo, NamedSeedMatching);

        if (!obj) {
          if (ZeroSeedExtension) {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values = std::vector<unsigned char>(mo->size, '\0');
          } else if (!AllowSeedExtension) {
            terminateStateOnError(state,
                                  "ran out of inputs during seeding",
                                  "user.err");
            break;
          }
        } else {
          if (obj->numBytes != mo->size &&
              ((!(AllowSeedExtension || ZeroSeedExtension)
                && obj->numBytes < mo->size) ||
               (!AllowSeedTruncation && obj->numBytes > mo->size))) {
	    std::stringstream msg;
	    msg << "replace size mismatch: "
		<< mo->name << "[" << mo->size << "]"
		<< " vs " << obj->name << "[" << obj->numBytes << "]"
		<< " in test\n";

            terminateStateOnError(state,
                                  msg.str(),
                                  "user.err");
            break;
          } else {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values.insert(values.begin(), obj->bytes, 
                          obj->bytes + std::min(obj->numBytes, mo->size));
            if (ZeroSeedExtension) {
              for (unsigned i=obj->numBytes; i<mo->size; ++i)
                values.push_back('\0');
            }
          }
        }
      }
    }
  } else {     // replay
    ObjectState *os = bindObjectInState(state, mo, false);
    if (replayPosition >= replayOut->numObjects) {
      terminateStateOnError(state, "replay count mismatch", "user.err");
    } else {
      KTestObject *obj = &replayOut->objects[replayPosition++];
      if (obj->numBytes != mo->size) {
        terminateStateOnError(state, "replay size mismatch", "user.err");
      } else {
        for (unsigned i=0; i<mo->size; i++)
          os->write8(i, obj->bytes[i]);
      }
    }
  }
  Gklee::Logging::exitFunc();
}


/***/

void Executor::runFunctionAsMain(Function *f,
				 int argc,
				 char **argv,
				 char **envp) {
  Gklee::Logging::enterFunc< std::string >( f->getName(), __PRETTY_FUNCTION__ );  
  std::vector<klee::ref<Expr> > arguments;

  // force deterministic initialization of memory objects
  srand(1);
  srandom(1);
  
  MemoryObject *argvMO = 0;

  // In order to make uclibc happy and be closer to what the system is
  // doing we lay out the environments at the end of the argv array
  // (both are terminated by a null). There is also a final terminating
  // null that uclibc seems to expect, possibly the ELF header?

  int envc;
  for (envc=0; envp[envc]; ++envc) ;

  unsigned NumPtrBytes = Context::get().getPointerWidth() / 8;
  KFunction *kf = kmodule->functionMap[f];
  assert(kf);
  Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
  if (ai!=ae) {
    arguments.push_back(ConstantExpr::alloc(argc, Expr::Int32));

    if (++ai!=ae) {
      argvMO = memory->allocate((argc+1+envc+1+1) * NumPtrBytes, false, true, false,
                                is_GPU_mode, f->begin()->begin());
      
      arguments.push_back(argvMO->getBaseExpr());
      Gklee::Logging::outItem< klee::ref< klee::Expr >>( argvMO->getBaseExpr(), "argument expression" );

      if (++ai!=ae) {
        uint64_t envp_start = argvMO->address + (argc+1)*NumPtrBytes;
        arguments.push_back(Expr::createPointer(envp_start));

        if (++ai!=ae)
          klee_error("invalid main function (expect 0-3 arguments)");
      }
    }
  }

  ExecutionState *state = new ExecutionState(kmodule->functionMap[f]);
  
  if (pathWriter) 
    state->pathOS = pathWriter->open();
  if (symPathWriter) 
    state->symPathOS = symPathWriter->open();


  if (statsTracker)
    statsTracker->framePushed(*state, 0);

  assert(arguments.size() == f->arg_size() && "wrong number of arguments");
  for (unsigned i = 0, e = f->arg_size(); i != e; ++i)
    bindArgument(kf, i, *state, arguments[i]);

  if (argvMO) {
    ObjectState *argvOS = bindObjectInState(*state, argvMO, false);

    for (int i=0; i<argc+1+envc+1+1; i++) {
      MemoryObject *arg;
      
      if (i==argc || i>=argc+1+envc) {
        arg = 0;
      } else {
        char *s = i<argc ? argv[i] : envp[i-(argc+1)];
        int j, len = strlen(s);
        
        arg = memory->allocate(len+1, false, true, false, is_GPU_mode, state->getPC()->inst);
        ObjectState *os = bindObjectInState(*state, arg, false);
        for (j=0; j<len+1; j++)
          os->write8(j, s[j]);
      }

      if (arg) {
        argvOS->write(i * NumPtrBytes, arg->getBaseExpr());
      } else {
        argvOS->write(i * NumPtrBytes, Expr::createPointer(0));
      }
    }
  }
  
  initializeGlobals(*state);

  processTree = new PTree(state);
  state->ptreeNode = processTree->root;
  run(*state);

  delete processTree;
  processTree = 0;

  // hack to clear memory objects
  delete memory;
  memory = new MemoryManager();
  
  globalObjects.clear();
  globalAddresses.clear();

  if (statsTracker)
    statsTracker->done();

  if (theMMap) {
    munmap(theMMap, theMMapSize);
    theMMap = 0;
  }
  Gklee::Logging::exitFunc();
}

unsigned Executor::getPathStreamID(const ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  assert(pathWriter);
  auto id = state.pathOS.getID();
  Gklee::Logging::exitFunc();
  return id;
}

unsigned Executor::getSymbolicPathStreamID(const ExecutionState &state) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  assert(symPathWriter);
  auto id = state.symPathOS.getID();
  Gklee::Logging::exitFunc();
  return id;
}

void Executor::getConstraintLog(const ExecutionState &state,
                                std::string &res,
                                bool asCVC) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (asCVC) {
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool));
    char *log = solver->stpSolver->getConstraintLog(query);
    res = std::string(log);
    free(log);
  } else {
    std::ostringstream info;
    ExprPPrinter::printConstraints(info, state.constraints);
    res = info.str();    
  }
  Gklee::Logging::exitFunc();
}

bool Executor::getSymbolicConfig(ExecutionState &state, klee::ref<Expr> cond) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  solver->setTimeout(stpTimeout);

  ExecutionState tmp(state); 
  if (!NoPreferCex) {
    std::cout << "No Prefer Cex" << std::endl;
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;

  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);

  addConstraint(tmp, cond);
  tmp.dumpStateConstraint();  

  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(0);
  if (!success){
    Gklee::Logging::exitFunc();
    return false;
  }

  Assignment *binding = new Assignment(objects, values);
  ObjectState *os = state.addressSpace.findNonConstantObject(state.tinfo.thread_id_mo, 0);
  klee::ref<Expr> tidx = os->read(0, Expr::Int32);
  klee::ref<Expr> tidy = os->read(4, Expr::Int32);
  klee::ref<Expr> tidz = os->read(8, Expr::Int32);
  
  klee::ref<Expr> cTidx = binding->evaluate(tidx);
  klee::ref<Expr> cTidy = binding->evaluate(tidy);
  klee::ref<Expr> cTidz = binding->evaluate(tidz);

  klee::ref<Expr> cCond = binding->evaluate(cond);

  Gklee::Logging::exitFunc();
  return true;
}

bool Executor::getSymbolicConfigSolution(ExecutionState &state, klee::ref<Expr> condition,
                                         std::vector< klee::ref<Expr> > offsetVec,
                                         std::vector< klee::ref<Expr> > &cOffsetVec,
                                         klee::ref<Expr> val1, klee::ref<Expr> val2, bool &benign,
                                         std::vector<SymBlockID_t> &symBlockIDs, 
                                         std::vector<SymThreadID_t> &symThreadIDs, 
                                         SymBlockDim_t &symBlockDim) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  solver->setTimeout(stpTimeout);

  ExecutionState tmp(state);
  if (!NoPreferCex) {
    std::cout << "No Prefer Cex" << std::endl;
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;

  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);

  addConstraint(tmp, condition);
  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(0);
  if (!success){
    Gklee::Logging::exitFunc();
    return false;
  }
   
  Assignment *binding = new Assignment(objects, values);
  
  ObjectState *bos0 = tmp.addressSpace.sharedMemories[0].findNonConstantObject(tmp.tinfo.block_id_mo);
  ObjectState *bos1 = tmp.addressSpace.sharedMemories[1].findNonConstantObject(tmp.tinfo.block_id_mo);
  ObjectState *tos0 = tmp.addressSpace.localMemories[0].findNonConstantObject(tmp.tinfo.thread_id_mo); 
  ObjectState *tos1 = tmp.addressSpace.localMemories[1].findNonConstantObject(tmp.tinfo.thread_id_mo); 

  // bid ...
  klee::ref<Expr> bid0x = binding->evaluate(bos0->read(0, Expr::Int32));
  klee::ref<Expr> bid0y = binding->evaluate(bos0->read(4, Expr::Int32));
  klee::ref<Expr> bid0z = binding->evaluate(bos0->read(8, Expr::Int32));
  klee::ref<Expr> bid1x = binding->evaluate(bos1->read(0, Expr::Int32));
  klee::ref<Expr> bid1y = binding->evaluate(bos1->read(4, Expr::Int32));
  klee::ref<Expr> bid1z = binding->evaluate(bos1->read(8, Expr::Int32));
  // tid ...
  klee::ref<Expr> tid0x = binding->evaluate(tos0->read(0, Expr::Int32));
  klee::ref<Expr> tid0y = binding->evaluate(tos0->read(4, Expr::Int32));
  klee::ref<Expr> tid0z = binding->evaluate(tos0->read(8, Expr::Int32));
  klee::ref<Expr> tid1x = binding->evaluate(tos1->read(0, Expr::Int32));
  klee::ref<Expr> tid1y = binding->evaluate(tos1->read(4, Expr::Int32));
  klee::ref<Expr> tid1z = binding->evaluate(tos1->read(8, Expr::Int32));

  // thread 0
  if (ConstantExpr *bid0x_t = dyn_cast<ConstantExpr>(bid0x)) {
    symBlockIDs[0].x = bid0x_t->getZExtValue();
  }
  if (ConstantExpr *bid0y_t = dyn_cast<ConstantExpr>(bid0y)) {
    symBlockIDs[0].y = bid0y_t->getZExtValue();
  }
  if (ConstantExpr *bid0z_t = dyn_cast<ConstantExpr>(bid0z)) {
    symBlockIDs[0].z = bid0z_t->getZExtValue();
  }
  if (ConstantExpr *tid0x_t = dyn_cast<ConstantExpr>(tid0x)) {
    symThreadIDs[0].x = tid0x_t->getZExtValue();
  }
  if (ConstantExpr *tid0y_t = dyn_cast<ConstantExpr>(tid0y)) {
    symThreadIDs[0].y = tid0y_t->getZExtValue();
  }
  if (ConstantExpr *tid0z_t = dyn_cast<ConstantExpr>(tid0z)) {
    symThreadIDs[0].z = tid0z_t->getZExtValue();
  }
 
  // thread 1
  if (ConstantExpr *bid1x_t = dyn_cast<ConstantExpr>(bid1x)) {
    symBlockIDs[1].x = bid1x_t->getZExtValue();
  }
  if (ConstantExpr *bid1y_t = dyn_cast<ConstantExpr>(bid1y)) {
    symBlockIDs[1].y = bid1y_t->getZExtValue();
  }
  if (ConstantExpr *bid1z_t = dyn_cast<ConstantExpr>(bid1z)) {
    symBlockIDs[1].z = bid1z_t->getZExtValue();
  }
  if (ConstantExpr *tid1x_t = dyn_cast<ConstantExpr>(tid1x)) {
    symThreadIDs[1].x = tid1x_t->getZExtValue();
  }
  if (ConstantExpr *tid1y_t = dyn_cast<ConstantExpr>(tid1y)) {
    symThreadIDs[1].y = tid1y_t->getZExtValue();
  }
  if (ConstantExpr *tid1z_t = dyn_cast<ConstantExpr>(tid1z)) {
    symThreadIDs[1].z = tid1z_t->getZExtValue();
  }

  for (unsigned i = 0; i < offsetVec.size(); i++) {
    // concretize the 'offset' expression w.r.t bindings ...
    klee::ref<Expr> cOffset = binding->evaluate(offsetVec[i]); 
    cOffsetVec.push_back(cOffset);
  }
  // if benign = true, race checking ... 
  if (benign) {
    if (val1->getWidth() == val2->getWidth()) {
      klee::ref<Expr> eqExpr = EqExpr::create(val1, val2);
      bool result = false;
      solver->mustBeTrue(tmp, eqExpr, result); 
      benign = result;
    }
  }
  delete binding;
  // update the configuration vector of current parametric node ... 
  Gklee::Logging::exitFunc();
  return true;
}

bool Executor::dumpOffsetValue(const ExecutionState &state, 
                               klee::ref<Expr> offset) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  solver->setTimeout(stpTimeout);

  ExecutionState tmp(state);
  if (!NoPreferCex) {
    std::cout << "No Prefer Cex" << std::endl;
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;

  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);

  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(0);
  if (!success){
    Gklee::Logging::exitFunc();
    return false;
  }
   
  Assignment *binding = new Assignment(objects, values);
  
  ObjectState *bos0 = tmp.addressSpace.sharedMemories[0].findNonConstantObject(tmp.tinfo.block_id_mo);
  ObjectState *tos0 = tmp.addressSpace.localMemories[0].findNonConstantObject(tmp.tinfo.thread_id_mo); 

  // bid ...
  klee::ref<Expr> bid0x = binding->evaluate(bos0->read(0, Expr::Int32));
  klee::ref<Expr> bid0y = binding->evaluate(bos0->read(4, Expr::Int32));
  klee::ref<Expr> bid0z = binding->evaluate(bos0->read(8, Expr::Int32));
  // tid ...
  klee::ref<Expr> tid0x = binding->evaluate(tos0->read(0, Expr::Int32));
  klee::ref<Expr> tid0y = binding->evaluate(tos0->read(4, Expr::Int32));
  klee::ref<Expr> tid0z = binding->evaluate(tos0->read(8, Expr::Int32));

  // thread 0
  if (ConstantExpr *bid0x_t = dyn_cast<ConstantExpr>(bid0x)) {
    std::cout << "the block x: " << bid0x_t->getZExtValue() << std::endl;
  }
  if (ConstantExpr *bid0y_t = dyn_cast<ConstantExpr>(bid0y)) {
    std::cout << "the block y: " << bid0y_t->getZExtValue() << std::endl;
  }
  if (ConstantExpr *bid0z_t = dyn_cast<ConstantExpr>(bid0z)) {
    std::cout << "the block z: " << bid0z_t->getZExtValue() << std::endl;
  }
  if (ConstantExpr *tid0x_t = dyn_cast<ConstantExpr>(tid0x)) {
    std::cout << "the thread x: " << tid0x_t->getZExtValue() << std::endl;
  }
  if (ConstantExpr *tid0y_t = dyn_cast<ConstantExpr>(tid0y)) {
    std::cout << "the thread y: " << tid0y_t->getZExtValue() << std::endl;
  }
  if (ConstantExpr *tid0z_t = dyn_cast<ConstantExpr>(tid0z)) {
    std::cout << "the thread z: " << tid0z_t->getZExtValue() << std::endl;
  }

  klee::ref<Expr> offsetRef = binding->evaluate(offset); 
  std::cout << "offset in concrete : " << std::endl;
  offsetRef->dump();
  
  delete binding; 
  Gklee::Logging::exitFunc();
  return true;
}

bool Executor::getConditionSolution(const ExecutionState &state, 
                                    klee::ref<Expr> condition, 
                                    std::vector<
                                    std::pair<std::string,
                                    std::vector<unsigned char> > >
                                    &res) {
  Gklee::Logging::enterFunc( condition , __PRETTY_FUNCTION__ );  
  solver->setTimeout(stpTimeout);

  ExecutionState tmp(state);

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);

  addConstraint(tmp, condition);
  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(0);
  if (!success){
    Gklee::Logging::exitFunc();
    return false;
  }
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    res.push_back(std::make_pair(state.symbolics[i].first->name, values[i]));
  Gklee::Logging::exitFunc();
  return true;
}

bool Executor::getSymbolicSolution(const ExecutionState &state,
                                   std::vector<
                                   std::pair<std::string,
                                   std::vector<unsigned char> > >
                                   &res) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  solver->setTimeout(stpTimeout);

  ExecutionState tmp(state);
  if (!NoPreferCex) {
    for (unsigned i = 0; i != state.symbolics.size(); ++i) {
      const MemoryObject *mo = state.symbolics[i].first;
      std::vector< klee::ref<Expr> >::const_iterator pi = 
        mo->cexPreferences.begin(), pie = mo->cexPreferences.end();
      for (; pi != pie; ++pi) {
        bool mustBeTrue;
        bool success = solver->mustBeTrue(tmp, Expr::createIsZero(*pi), 
                                          mustBeTrue);
        if (!success) break;
        if (!mustBeTrue) tmp.addConstraint(*pi);
      }
      if (pi!=pie) break;
    }
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);
  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(0);
  if (!success) {
    // by Guodong
    // this path is infeasible; return here
    // klee_warning("unable to compute initial values (invalid constraints?)!");
    //ExprPPrinter::printQuery(std::cerr,
    //                         state.constraints, 
    //                         ConstantExpr::alloc(0, Expr::Bool));
    Gklee::Logging::exitFunc();
    return false;
  }
  // std::string dumpInfo;
  for (unsigned i = 0; i != state.symbolics.size(); ++i){
    res.push_back(std::make_pair(state.symbolics[i].first->name, values[i]));
    // dumpInfo += state.symbolics[i].first->name + std::to_string( values[i] );
  }
  // Gklee::Logging::outItem( dumpInfo , "sym info: " );
  Gklee::Logging::exitFunc();
  return true;
}

void Executor::getCoveredLines(const ExecutionState &state,
                               std::map<const std::string*, std::set<unsigned> > &res) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  res = state.coveredLines;
  std::string clines;
  for(auto x: state.coveredLines) clines +=  *x.first;
  Gklee::Logging::outItem( clines , "covered lines: " );
  Gklee::Logging::exitFunc();  
}

void Executor::doImpliedValueConcretization(ExecutionState &state,
                                            klee::ref<Expr> e,
                                            klee::ref<ConstantExpr> value) {
  abort(); // FIXME: Broken until we sort out how to do the write back.

  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  if (DebugCheckForImpliedValues)
    ImpliedValue::checkForImpliedValues(solver->solver, e, value);

  ImpliedValueList results;
  ImpliedValue::getImpliedValues(e, value, results);
  for (ImpliedValueList::iterator it = results.begin(), ie = results.end();
       it != ie; ++it) {
    ReadExpr *re = it->first.get();
    
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(re->index)) {
      // FIXME: This is the sole remaining usage of the Array object
      // variable. Kill me.
      const MemoryObject *mo = 0; //re->updates.root->object;
      const ObjectState *os = state.addressSpace.findObject(mo);

      if (!os) {
        // object has been free'd, no need to concretize (although as
        // in other cases we would like to concretize the outstanding
        // reads, but we have no facility for that yet)
      } else {
        assert(!os->readOnly && 
               "not possible? read only object with static read?");
        ObjectState *wos = state.addressSpace.getWriteable(mo, os);
        wos->write(CE, it->second);
      }
    }
  }
  Gklee::Logging::exitFunc();
}

Expr::Width Executor::getWidthForLLVMType(LLVM_TYPE_Q llvm::Type *type) const {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  //  Gklee::Logging::outLLVMObj< llvm::Type >( *type );
  auto tsib = kmodule->targetData->getTypeSizeInBits(type);
  Gklee::Logging::outItem( std::to_string( tsib ), "type size: " ); 
  Gklee::Logging::exitFunc();
  return tsib;
}

///

Interpreter *Interpreter::create(const InterpreterOptions &opts,
                                 InterpreterHandler *ih) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );  
  auto e = new Executor(opts, ih);
  Gklee::Logging::exitFunc();
  return e;
}
