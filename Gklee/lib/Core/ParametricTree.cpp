#include "CUDA.h"
#include <ParametricTree.h>

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
  root = current = NULL;
  nodeNum = 0;
}

ParaTree::ParaTree(const ParaTree &_paraTree) {
  if (_paraTree.nodeNum == 0) {
    nodeNum = 0;
    root = current = NULL;
    return;
  }

  nodeNum = _paraTree.nodeNum;
  root = copyParaTree(_paraTree.root, _paraTree.current);
  root->parent = NULL;
}

ParaTreeNode* ParaTree::copyParaTree(ParaTreeNode *other, 
                                     ParaTreeNode *otherCurrent) {
  if (other == NULL) return NULL;
  
  ParaTreeNode *newNode = new ParaTreeNode(*other);
  if (other == otherCurrent) current = newNode;
  std::vector<ParaTreeNode*> &treeNodes = newNode->successorTreeNodes;
  std::vector<ParaTreeNode*> &otherTreeNodes = other->successorTreeNodes;
  unsigned otherSize = otherTreeNodes.size();

  for (unsigned i = 0; i < otherSize; i++) {
    treeNodes[i] = copyParaTree(otherTreeNodes[i], otherCurrent);
    if (treeNodes[i] != NULL) treeNodes[i]->parent = newNode;
  }

  return newNode;
}

ParaTree::~ParaTree() {
  destroyParaTree(root);
}

ParaTreeNode *ParaTree::getRootNode() {
  return root;
}

ParaTreeNode *ParaTree::getCurrentNode() {
  return current;
}

unsigned ParaTree::getSymbolicTidFromCurrentNode(unsigned exploreNum) {
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  //std::cout << "exploreNum:" << exploreNum << std::endl;
  assert(exploreNum < configVec.size() && 
         "exploreNum is greater than the number of all successors, check!\n");
  return configVec[exploreNum].sym_tid;
}

void ParaTree::updateCurrentNodeOnNewConfig(ParaConfig &config, SymBrType symBrType) {
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
}

void ParaTree::insertNodeIntoParaTree(ParaTreeNode *node) {
  if (root == NULL) {
    root = current = node;
    nodeNum = 1;
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
}

void ParaTree::initializeCurrentNodeRange(unsigned cur_tid, unsigned pos) {
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  for (unsigned i = 0; i < configVec.size(); i++) {
    if (configVec[i].sym_tid == cur_tid) {
      configVec[i].start = pos;
      configVec[i].end = pos; 
      break;
    }
  }
}

void ParaTree::incrementCurrentNodeRange(unsigned cur_tid, unsigned pos) {
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
}

void ParaTree::updateConfigVecAfterBarriers(ParaTreeNode *tmpNode) {
  bool allSync = true;
  std::vector<ParaConfig> &configVec = tmpNode->successorConfigVec;

  for (unsigned i = 0; i < configVec.size(); i++) {
    if (!configVec[i].syncEncounter) {
      allSync = false;
      break;
    }
  }
  tmpNode->allSync = allSync;
} 

void ParaTree::encounterImplicitBarrier(ParaTreeNode *tmpNode, ParaTreeNode *pNode) {
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
}

void ParaTree::encounterExplicitBarrier(std::vector<CorrespondTid> &cTidSets, 
                                        unsigned cur_tid) {
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
}

void ParaTree::destroyParaTree(ParaTreeNode *node) {
  if (node != NULL) {
    std::vector<ParaTreeNode*> &treeNodes = node->successorTreeNodes;

    for (unsigned i = 0; i < treeNodes.size(); i++) {
      destroyParaTree(treeNodes[i]);
      treeNodes[i] = NULL;
    } 

    node->parent = NULL;
    delete node;
  } 
}

ref<Expr> ParaTree::getCurrentNodeTDCExpr() {
  unsigned which = current->whichSuccessor;
  std::vector<ParaConfig> &configVec = current->successorConfigVec;
  ref<Expr> cond = 0;
  if (current->symBrType == TDC) 
    cond = AndExpr::create(current->tdcCond, configVec[which].cond);  
  else 
    cond = current->tdcCond;
  return cond;
}

void ParaTree::negateNonTDCNodeCond() {
  if (current) {
    assert(current->symBrType != TDC && "Non-TDC condition property violated!");
    std::vector<ParaConfig> &configVec = current->successorConfigVec;
    configVec[0].cond = Expr::createIsZero(configVec[0].cond); 
  }
}

void ParaTree::resetNonTDCNodeCond() {
  if (current) {
    assert(current->symBrType != TDC && "Non-TDC condition property violated!");
    std::vector<ParaConfig> &configVec = current->successorConfigVec;
    configVec[0].cond = ConstantExpr::create(1, Expr::Bool); 
  }
}

void ParaTree::dumpAllNodes(ParaTreeNode *node) const {
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
}

void ParaTree::dumpParaTree() const {
  dumpAllNodes(root);
}

bool ParaTree::isRootNull() const {
  return root == NULL;
}

unsigned ParaTree::getNodeNum() const {
  return nodeNum;
}

bool ParaTree::currentSuccessorNull() const {
  if (current == NULL)
    return true;
  else {
    unsigned which = current->whichSuccessor;
    std::vector<ParaTreeNode*> &treeNodes = current->successorTreeNodes; 
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
