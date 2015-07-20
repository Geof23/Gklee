#include "CUDA.h"
#include "ParametricTree.h"
#include "klee/logging.h"

using namespace Gklee;

ParaTreeNode::ParaTreeNode(llvm::Instruction *_brInst, llvm::BasicBlock *_postDom,
	     SymBrType _symBrType, bool _isCondBr, bool _allSync, 
	     klee::ref<Expr> _inheritCond, klee::ref<Expr> _tdcCond):
  brInst(_brInst), postDom(_postDom), 
  symBrType(_symBrType), isCondBr(_isCondBr), 
  allSync(_allSync), inheritCond(_inheritCond), 
  tdcCond(_tdcCond) {
  Gklee::Logging::enterFunc( *brInst, __PRETTY_FUNCTION__ );
  whichSuccessor = 0;
  parent = NULL;
  Gklee::Logging::exitFunc();
  }

ParaTreeNode::ParaTreeNode(const ParaTreeNode &node) :
  brInst(node.brInst), postDom(node.postDom), 
  symBrType(node.symBrType), 
  isCondBr(node.isCondBr), allSync(node.allSync), 
  whichSuccessor(node.whichSuccessor), 
  inheritCond(node.inheritCond), 
  tdcCond(node.tdcCond),
  repThreadSet(node.repThreadSet), 
  divergeThreadSet(node.divergeThreadSet) {
    Gklee::Logging::enterFunc( *node.brInst, __PRETTY_FUNCTION__ );
    parent = NULL;
    successorConfigVec = node.successorConfigVec;
    unsigned size = node.successorTreeNodes.size();
    for (unsigned i = 0; i < size; i++)
      successorTreeNodes.push_back(NULL);  
    Gklee::Logging::exitFunc();
  }

ParaTreeNode::~ParaTreeNode() {
    Gklee::Logging::enterFunc< std::string >( "destroying node", __PRETTY_FUNCTION__ );
    successorConfigVec.clear();
    successorTreeNodes.clear();
    repThreadSet.clear();
    divergeThreadSet.clear();
    Gklee::Logging::exitFunc();
  }

void ParaTreeNode::dumpParaTreeNode() {
  if (brInst) { 
    GKLEE_INFO << "**************" << std::endl; 
    GKLEE_INFO << "[isCond]: " << isCondBr << std::endl;
    GKLEE_INFO << "[inst]: " << std::endl;
    brInst->dump();
    GKLEE_INFO << "[inheritCond]: " << std::endl;
    inheritCond->dump();
    for (unsigned i = 0; i < successorConfigVec.size(); i++) {
      GKLEE_INFO << "[bid, tid, start, end, syncEncounter]: " 
                 << "[" << successorConfigVec[i].sym_bid << ", " 
                 << successorConfigVec[i].sym_tid << ", "
                 << successorConfigVec[i].start << ", " 
                 << successorConfigVec[i].end << ", " 
                 << successorConfigVec[i].syncEncounter << "]" << std::endl;
    }
    GKLEE_INFO << "++++++++++++++" << std::endl; 
  } else {
    GKLEE_INFO << "**************" << std::endl; 
    GKLEE_INFO << "[Post Dominator Node]" << std::endl;
    GKLEE_INFO << "++++++++++++++" << std::endl; 
  }
}

ParaTree::ParaTree() {
  Gklee::Logging::enterFunc< std::string >( "creating paraTree", __PRETTY_FUNCTION__ );
  root = current = NULL;
  nodeNum = 0;
  Gklee::Logging::exitFunc();
}

ParaTree::ParaTree(const ParaTree &_paraTree) {
  Gklee::Logging::enterFunc< std::string >( "copy construct paraTree", __PRETTY_FUNCTION__ );
  if (_paraTree.nodeNum == 0) {
    nodeNum = 0;
    root = current = NULL;
    Gklee::Logging::exitFunc();
    return;
  }

  nodeNum = _paraTree.nodeNum;
  root = copyParaTree(_paraTree.root, _paraTree.current);
  root->parent = NULL;
  Gklee::Logging::exitFunc();
}

ParaTreeNode* ParaTree::copyParaTree(ParaTreeNode *other, 
                                     ParaTreeNode *otherCurrent) {
  Gklee::Logging::enterFunc< std::string >( "copy paraTree", __PRETTY_FUNCTION__ );
  if (other == NULL){
    Logging::exitFunc();
    return NULL;
  }
  ParaTreeNode *newNode = new ParaTreeNode(*other);
  if (other == otherCurrent) current = newNode;
  std::vector<ParaTreeNode*> &treeNodes = newNode->successorTreeNodes;
  std::vector<ParaTreeNode*> &otherTreeNodes = other->successorTreeNodes;
  unsigned otherSize = otherTreeNodes.size();

  for (unsigned i = 0; i < otherSize; i++) {
    treeNodes[i] = copyParaTree(otherTreeNodes[i], otherCurrent);
    if (treeNodes[i] != NULL) treeNodes[i]->parent = newNode;
  }
  Gklee::Logging::exitFunc();
  return newNode;
}

ParaTree::~ParaTree() {
  Gklee::Logging::enterFunc< std::string >( "destructor", __PRETTY_FUNCTION__ );
  destroyParaTree(root);
  Gklee::Logging::exitFunc();
}

ParaTreeNode *ParaTree::getRootNode() {
  return root;
}

ParaTreeNode *ParaTree::getCurrentNode() {
  return current;
}

unsigned ParaTree::getSymbolicTidFromCurrentNode(unsigned exploreNum) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  //std::cout << "exploreNum:" << exploreNum << std::endl;
  assert(exploreNum < configVec.size() && 
         "exploreNum is greater than the number of all successors, check!\n");
  Gklee::Logging::exitFunc();
  return configVec[exploreNum].sym_tid;
}

void ParaTree::updateCurrentNodeOnNewConfig(ParaConfig &config, SymBrType symBrType) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  std::vector<ParaTreeNode*> &treeNodes = current->successorTreeNodes;
  configVec.push_back(config);
  treeNodes.push_back(NULL);
  std::set<unsigned> threadSet;
  unsigned sym_tid = config.sym_tid;
  threadSet.insert(sym_tid);
  current->repThreadSet.push_back(threadSet);
  current->divergeThreadSet.push_back(threadSet);

  if (symBrType == TDC) {
    ParaTreeNode *tmp = current->parent;
    while (tmp != NULL) {
      if (!tmp->allSync 
           && tmp->symBrType == TDC) {
        unsigned which = tmp->whichSuccessor;
        std::set<unsigned> &threadSet = tmp->repThreadSet[which];
        threadSet.insert(sym_tid);
        std::vector<ParaConfig> &configVec = tmp->successorConfigVec;
        if (!configVec[which].postDomEncounter) {
          std::set<unsigned> &divergeSet = tmp->divergeThreadSet[which];
          divergeSet.insert(sym_tid);
        }
      }
      tmp = tmp->parent;
    }
  }
  Gklee::Logging::exitFunc();
}

void ParaTree::insertNodeIntoParaTree(ParaTreeNode *node) {
  Gklee::Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if (root == NULL) {
    root = current = node;
    nodeNum = 1;
    Logging::exitFunc();
    return;
  }

  std::vector<ParaTreeNode*> &treeNodes = current->successorTreeNodes;
  if (current->brInst) {
    if (current->symBrType == TDC) {
      // TDC
      unsigned which = current->whichSuccessor;
      assert(treeNodes[which] == NULL && "treeNodes is not NULL, check here");
      treeNodes[which] = node;
    } else {
      // SYM || Accumulative
      treeNodes[0] = node;
    }
  } else {
    // Common post-dominator 
    treeNodes[0] = node;
  }

  node->parent = current;
  current = node;
  nodeNum++;
  Logging::exitFunc();
}

void ParaTree::initializeCurrentNodeRange(unsigned cur_tid, unsigned pos) {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  for (unsigned i = 0; i < configVec.size(); i++) {
    if (configVec[i].sym_tid == cur_tid) {
      configVec[i].start = pos;
      configVec[i].end = pos; 
      break;
    }
  }
  Logging::exitFunc();
}

void ParaTree::incrementCurrentNodeRange(unsigned cur_tid, unsigned pos) {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  ParaTreeNode *tmp = current;

  while (tmp != NULL) {
    std::vector<ParaConfig> &configVec = tmp->successorConfigVec;
    unsigned which = tmp->whichSuccessor;
    if (!configVec[which].postDomEncounter) {
      if (configVec[which].sym_tid == cur_tid)
        configVec[which].end = pos;
    }
    tmp = tmp->parent;
  }
  Logging::exitFunc();
}

void ParaTree::updateConfigVecAfterBarriers(ParaTreeNode *tmpNode) {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  bool allSync = true;
  std::vector<ParaConfig> &configVec = tmpNode->successorConfigVec;

  for (unsigned i = 0; i < configVec.size(); i++) {
    if (!configVec[i].syncEncounter) {
      allSync = false;
      break;
    }
  }
  tmpNode->allSync = allSync;
  Logging::exitFunc();
} 

void ParaTree::encounterImplicitBarrier(ParaTreeNode *tmpNode, ParaTreeNode *pNode) {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  std::vector<ParaConfig> &configVec = tmpNode->successorConfigVec;
  std::vector<ParaTreeNode*> &treeNodes = tmpNode->successorTreeNodes;
  unsigned which = tmpNode->whichSuccessor;

  if (!configVec[which].syncEncounter)
    configVec[which].syncEncounter = true;
  updateConfigVecAfterBarriers(tmpNode);

  if (!treeNodes[which]) {
    treeNodes[which] = pNode;
    pNode->parent = tmpNode;
    if (which+1 == configVec.size()) {
      ParaTreeNode *tmp = treeNodes[0];
      while (tmp->successorTreeNodes[0] != NULL) {
        tmp = tmp->successorTreeNodes[0];
      }
      current = tmp;
    }
  } else {
    // Identify the new "current" node
    if (!tmpNode->allSync) {
      // current node is a synchronization node
      if (!current->brInst)
        current = tmpNode;
    }
  }
  Logging::exitFunc();
}

void ParaTree::encounterExplicitBarrier(std::vector<CorrespondTid> &cTidSets, 
                                        unsigned cur_tid) {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  ParaTreeNode *tmp = current;  

  while (tmp != NULL) {
    if (tmp->brInst) {
      if (tmp->allSync) {
        std::vector< std::set<unsigned> > &threadSetVec = tmp->repThreadSet;
        for (unsigned i = 0; i < threadSetVec.size(); i++) {
          std::set<unsigned> &threadSet = threadSetVec[i];
          for (std::set<unsigned>::iterator ii = threadSet.begin(); 
               ii != threadSet.end(); ii++) {
            cTidSets[*ii].barrierEncounter = true;
            cTidSets[*ii].syncEncounter = true;
          }
        }
      } else {
        std::vector<ParaConfig> &configVec = tmp->successorConfigVec;
        unsigned which = tmp->whichSuccessor;
        std::vector< std::set<unsigned> > &threadSetVec = tmp->repThreadSet;
        std::set<unsigned> &threadSet = threadSetVec[which]; 
        bool allBarrier = true;
        for (std::set<unsigned>::iterator ii = threadSet.begin(); 
             ii != threadSet.end(); ii++) {
          if (!cTidSets[*ii].barrierEncounter) {
            allBarrier = false;
            break;
          } 
        } 
        if (allBarrier)
          configVec[which].syncEncounter = true;
        updateConfigVecAfterBarriers(tmp);
      }
    }
    tmp = tmp->parent;
  }

  // Update the current node 
  tmp = current;
  while (tmp != NULL) {
    if (tmp->brInst && !tmp->allSync) {
      break;
    }
    tmp = tmp->parent;
  }

  if (tmp != NULL) current = tmp;
  Logging::exitFunc();
}

void ParaTree::destroyParaTree(ParaTreeNode *node) {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if (node != NULL) {
    std::vector<ParaTreeNode*> &treeNodes = node->successorTreeNodes;

    for (unsigned i = 0; i < treeNodes.size(); i++) {
      destroyParaTree(treeNodes[i]);
      treeNodes[i] = NULL;
    } 

    node->parent = NULL;
    delete node;
  } 
  Logging::exitFunc();
}

klee::ref<Expr> ParaTree::getCurrentNodeTDCExpr() {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  unsigned which = current->whichSuccessor;
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  klee::ref<Expr> cond = 0;
  if (current->symBrType == TDC) 
    cond = AndExpr::create(current->tdcCond, configVec[which].cond);  
  else 
    cond = current->tdcCond;
  Logging::exitFunc();
  return cond;
}

void ParaTree::negateNonTDCNodeCond() {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if (current) {
    std::vector<ParaConfig> &configVec = current->successorConfigVec;
    configVec[0].cond = Expr::createIsZero(configVec[0].cond); 
  }
  Logging::exitFunc();
}

void ParaTree::resetNonTDCNodeCond() {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if (current) {
    std::vector<ParaConfig> &configVec = current->successorConfigVec;
    configVec[0].cond = ConstantExpr::create(1, Expr::Bool); 
  }
  Logging::exitFunc();
}

void ParaTree::dumpAllNodes(ParaTreeNode *node) const {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if (node->brInst == NULL)
    std::cout << "[Post Dominator Node]: " << std::endl;
  else 
    std::cout << "[Branch Node]: " << std::endl;
  node->dumpParaTreeNode();
  std::vector<ParaTreeNode*> &treeNodes = node->successorTreeNodes;

  for (unsigned i = 0; i < treeNodes.size(); i++) {
    if (treeNodes[i]) {
      std::cout << "The " << i << "th sub-node: " << std::endl;
      dumpAllNodes(treeNodes[i]);
    }
  }
  Logging::exitFunc();
}

void ParaTree::dumpParaTree() const {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if( root != NULL ){
    dumpAllNodes(root);
    std::cout << "ParaTree root is null" << std::endl;
  }
  Logging::exitFunc();
}

bool ParaTree::isRootNull() const {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  Logging::exitFunc();
  return root == NULL;
}

unsigned ParaTree::getNodeNum() const {
  return nodeNum;
}

bool ParaTree::currentSuccessorNull() const {
  Logging::enterFunc< std::string >( "", __PRETTY_FUNCTION__ );
  if (current == NULL){
    Logging::exitFunc();
    return true;
  } else {
    unsigned which = current->whichSuccessor;
    std::vector<ParaTreeNode*> &treeNodes = current->successorTreeNodes; 
    Logging::exitFunc();
    return (treeNodes[which] == NULL);  
  }
}

unsigned ParaTree::getCurrentNodePath() const {
  assert(current && "current node is NULL, Check");
  return current->whichSuccessor; 
}

void ParaTree::resetCurrentNodeToRoot() {
  current = root;
}
