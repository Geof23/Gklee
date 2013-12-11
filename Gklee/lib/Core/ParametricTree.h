#ifndef KLEE_PARAMETRICTREE_H
#define KLEE_PARAMETRICTREE_H

#include "klee/Expr.h"
#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"

using namespace klee;

class CorrespondTid {
public:
  unsigned rBid;
  unsigned rTid;
  unsigned warpNum;
  bool syncEncounter;    // explicit or implicit barrier
  bool barrierEncounter; // only explicit barrier 
  bool inBranch;
  ref<Expr> inheritExpr;
  bool slotUsed;

  CorrespondTid(unsigned _rBid, unsigned _rTid, 
                unsigned _warpNum, bool _syncEncounter, 
                bool _barrierEncounter, bool _inBranch, 
                ref<Expr> _inheritExpr, bool _slotUsed = false) : 
                rBid(_rBid), rTid(_rTid), warpNum(_warpNum), 
                syncEncounter(_syncEncounter), 
                barrierEncounter(_barrierEncounter), 
                inBranch(_inBranch), inheritExpr(_inheritExpr), 
                slotUsed(_slotUsed) {}
};

class ParaConfig {
public:
  unsigned sym_bid;
  unsigned sym_tid;
  ref<Expr> cond;
  unsigned start;
  unsigned end;
  bool syncEncounter;
  bool postDomEncounter;

  ParaConfig(unsigned _sym_bid, unsigned _sym_tid, ref<Expr> _cond, 
             unsigned _start, unsigned _end): 
             sym_bid(_sym_bid), sym_tid(_sym_tid), 
             cond(_cond), start(_start), end(_end) {
    syncEncounter = false;
    postDomEncounter = false;
  }

  ParaConfig(const ParaConfig &config) : 
  sym_bid(config.sym_bid), sym_tid(config.sym_tid),
  cond(config.cond), start(config.start), end(config.end), 
  syncEncounter(config.syncEncounter), 
  postDomEncounter(config.postDomEncounter) {};
};

enum SymBrType {
  TDC, // Block or Thread dependent condition
  SYM, // Symbolic conditional 
  ACCUM, // Accumulative condition
  Other // Conditions other than ones above
};

class ParaTreeNode {
public:
  llvm::Instruction *brInst;
  llvm::BasicBlock *postDom;
  SymBrType symBrType;
  bool isCondBr;
  bool allSync;
  unsigned whichSuccessor; // which flow is being explored right away 
  ref<Expr> inheritCond; // condition inherited from this node's parent
  ref<Expr> tdcCond; // condition only related to TDC  
   
  ParaTreeNode *parent;
  std::vector<ParaConfig> successorConfigVec;
  std::vector<ParaTreeNode*> successorTreeNodes;
  std::vector< std::set<unsigned> > repThreadSet;
  std::vector< std::set<unsigned> > divergeThreadSet; // used for checking porting race 

  ParaTreeNode(llvm::Instruction *_brInst, llvm::BasicBlock *_postDom,
               SymBrType _symBrType, bool _isCondBr, bool _allSync, 
               ref<Expr> _inheritCond, ref<Expr> _tdcCond):
  brInst(_brInst), postDom(_postDom), symBrType(_symBrType), 
  isCondBr(_isCondBr), allSync(_allSync), inheritCond(_inheritCond), 
  tdcCond(_tdcCond) {
    whichSuccessor = 0;
    parent = NULL;
  }

  ParaTreeNode(const ParaTreeNode &node) :
  brInst(node.brInst), postDom(node.postDom), 
  symBrType(node.symBrType), 
  isCondBr(node.isCondBr), allSync(node.allSync), 
  whichSuccessor(node.whichSuccessor), 
  inheritCond(node.inheritCond), 
  tdcCond(node.tdcCond),
  repThreadSet(node.repThreadSet), 
  divergeThreadSet(node.divergeThreadSet) {
    parent = NULL;
    successorConfigVec = node.successorConfigVec;
    unsigned size = node.successorTreeNodes.size();
    for (unsigned i = 0; i < size; i++)
      successorTreeNodes.push_back(NULL);  
  }

  ~ParaTreeNode() {
    successorConfigVec.clear();
    successorTreeNodes.clear();
    repThreadSet.clear();
    divergeThreadSet.clear();
  }

  void dumpParaTreeNode();
};

class ParaTree {
  public:
    ParaTree();
    ParaTree(const ParaTree &paraTree);
    ~ParaTree();
    ParaTreeNode *copyParaTree(ParaTreeNode *, ParaTreeNode *);
    ParaTreeNode *getRootNode();
    ParaTreeNode *getCurrentNode();
    unsigned getSymbolicTidFromCurrentNode(unsigned i);
    void updateCurrentNodeOnNewConfig(ParaConfig &config, SymBrType symBrType);
    void insertNodeIntoParaTree(ParaTreeNode *node);
    void initializeCurrentNodeRange(unsigned cur_tid, unsigned pos);
    void incrementCurrentNodeRange(unsigned cur_tid, unsigned pos);
    void updateConfigVecAfterBarriers(ParaTreeNode *tmpNode);
    void encounterImplicitBarrier(ParaTreeNode *tmpNode, ParaTreeNode *pNode);
    void encounterExplicitBarrier(std::vector<CorrespondTid> &cTidSets, 
                                  unsigned cur_tid);
    void destroyParaTree(ParaTreeNode *node);
    ref<Expr> getCurrentNodeTDCExpr();
    void negateNonTDCNodeCond();
    void resetNonTDCNodeCond();
    void dumpAllNodes(ParaTreeNode *node) const;
    void dumpParaTree() const;
    bool isRootNull() const;
    unsigned getNodeNum() const;
    bool currentSuccessorNull() const;
    unsigned getCurrentNodePath() const;
    void resetCurrentNodeToRoot();
  private:
    unsigned nodeNum; 
    ParaTreeNode *root;
    ParaTreeNode *current;
};

#endif
