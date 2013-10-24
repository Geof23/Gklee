#include "ConstantBitPropagation.h"
#include "../../AST/AST.h"
#include "../../extlib-constbv/constantbv.h"
#include "../../printer/printers.h"
#include "../../AST/NodeFactory/NodeFactory.h"
#include "../../simplifier/simplifier.h"
#include "ConstantBitP_Utility.h"

#ifdef WITHCBITP
  #include "ConstantBitP_TransferFunctions.h"
  #include "ConstantBitP_MaxPrecision.h"
#endif

using std::endl;
using std::cout;

using namespace BEEV;

/*
 *	Propagates known fixed 0 or 1 bits, as well as TRUE/FALSE values through the formula.
 *
 *	Our approach differs from others because the transfer functions are (mostly) optimally precise.
 *
 *	FixedBits stores booleans in 1 bit-bitvectors.
 */

namespace simplifier
{
  namespace constantBitP
  {
    NodeToFixedBitsMap* PrintingHackfixedMap; // Used when debugging.

    Result
    dispatchToTransferFunctions(const Kind k, vector<FixedBits*>& children,
        FixedBits& output, const ASTNode n, MultiplicationStatsMap *msm = NULL);

    Result
    dispatchToMaximallyPrecise(const Kind k, vector<FixedBits*>& children,
        FixedBits& output, const ASTNode n);

    const bool debug_cBitProp_messages = true;
    const bool output_mult_like = true;

    ////////////////////

    void
    ConstantBitPropagation::printNodeWithFixings()
    {
      NodeToFixedBitsMap::NodeToFixedBitsMapType::const_iterator it =
          fixedMap->map->begin();

      cerr << "+Nodes with fixings" << endl;

      for (/**/; it != fixedMap->map->end(); it++) // iterates through all the pairs of node->fixedBits.
        {
          cerr << (it->first).GetNodeNum() << " " << *(it->second) << endl;
        }
      cerr << "-Nodes with fixings" << endl;

    }

    // Used when outputting when debugging.
    // Outputs the fixed bits for a particular node.
    string
    toString(const ASTNode& n)
    {
      NodeToFixedBitsMap::NodeToFixedBitsMapType::const_iterator it =
          PrintingHackfixedMap->map->find(n);
      if (it == PrintingHackfixedMap->map->end())
        return "";

      std::stringstream s;
      s << *it->second;
      return s.str();
    }

    // If the bits are totally fixed, then return a new matching ASTNode.
    ASTNode
    bitsToNode(const ASTNode& node, const FixedBits& bits)
    {
      ASTNode result;
      STPMgr & beev = *node.GetSTPMgr();

      assert (bits.isTotallyFixed());
      assert (!node.isConstant()); // Peformance. Shouldn't waste time calling it on constants.

      if (node.GetType() == BOOLEAN_TYPE)
        {
          if (bits.getValue(0))
            {
              result = beev.CreateNode(TRUE);
            }
          else
            {
              result = beev.CreateNode(FALSE);
            }
        }
      else if (node.GetType() == BITVECTOR_TYPE)
        {
          result = beev.CreateBVConst(bits.GetBVConst(), node.GetValueWidth());
        }
      else
        FatalError("sadf234s");

      assert(result.isConstant());
      return result;
    }

    // Put anything that's entirely fixed into a from->to map.
    ASTNodeMap
    getAllFixed(NodeToFixedBitsMap* fixedMap)
    {
      NodeToFixedBitsMap::NodeToFixedBitsMapType::iterator it;

      ASTNodeMap toFrom;

      // iterates through all the pairs of node->fixedBits.
      for (it = fixedMap->map->begin(); it != fixedMap->map->end(); it++)
        {
          const ASTNode& node = (it->first);
          const FixedBits& bits = *it->second;

          // Don't constrain nodes we already know all about.
          if (node.isConstant())
              continue;

          // other nodes will contain the same information (the extract doesn't change the fixings).
          if (BVEXTRACT == node.GetKind() || BVCONCAT == node.GetKind())
            continue;

          if (bits.isTotallyFixed())
            {
              toFrom.insert(std::make_pair(node, bitsToNode(node, bits)));
            }
        }

      return toFrom;
    }

    void
    ConstantBitPropagation::setNodeToTrue(const ASTNode& top)
    {
      FixedBits & topFB = *getCurrentFixedBits(top);
      topFB.setFixed(0, true);
      topFB.setValue(0, true);
      workList->push(top);
    }

    // Propagates. No writing in of values. Doesn't assume the top is true.
    ConstantBitPropagation::ConstantBitPropagation(BEEV::Simplifier* _sm, NodeFactory* _nf,const ASTNode & top)
    {
      assert (BOOLEAN_TYPE == top.GetType());
      assert (top.GetSTPMgr()->UserFlags.bitConstantProp_flag);

      status = NO_CHANGE;
      simplifier = _sm;
      nf = _nf;
      fixedMap = new NodeToFixedBitsMap(1000); // better to use the function that returns the number of nodes.. whatever that is.
      workList = new WorkList(top);
      dependents = new Dependencies(top); // List of the parents of a node.
      msm = NULL;
      //msm = new MultiplicationStatsMap();


      // not fixing the topnode.
      propagate();

      if (debug_cBitProp_messages)
        {
          cerr << "status:" << status <<endl;
          cerr << "ended propagation" << endl;
          printNodeWithFixings();
        }

      // is there are good reason to clear out some of them??
#if 0
      // remove constants, and things with nothing fixed.
      NodeToFixedBitsMap::NodeToFixedBitsMapType::iterator it =
          fixedMap->map->begin();
      NodeToFixedBitsMap::NodeToFixedBitsMapType::iterator it_end =
          fixedMap->map->end();
      while (it != it_end)
        {
          // No constants, nothing completely unfixed.
          if (  (it->second)->countFixed() == 0 )
            {
              delete it->second;
              // making this a reference causes reading from freed memory.
              const ASTNode n = it->first;
              it++;
              fixedMap->map->erase(n);
            }
          else
            it++;
        }
#endif

    }

    // Both way propagation. Initialising the top to "true".
    // The hardest thing to understand is the two cases:
    // 1) If we get the fixed bits of a node, without assuming the top node is true,
    //    then we can replace that node by its fixed bits.
    // 2) But if we assume the top node is true, then get the bits, we need to conjoin it.

    // NB: This expects that the constructor was called with teh same node. Sorry.
    ASTNode
    ConstantBitPropagation::topLevelBothWays(const ASTNode& top)
    {
      assert(top.GetSTPMgr()->UserFlags.bitConstantProp_flag);
      assert (BOOLEAN_TYPE == top.GetType());

      propagate();
      status = NO_CHANGE;

      //Determine what must always be true.
      ASTNodeMap fromTo = getAllFixed(fixedMap);

      if (debug_cBitProp_messages)
        {
          cerr << "Number removed by bottom UP:" << fromTo.size() << endl;
        }

      setNodeToTrue(top);

      if (debug_cBitProp_messages)
        {
          cerr << "starting propagation" << endl;
          printNodeWithFixings();
          cerr << "Initial Tree:" << endl;
          cerr << top;
        }

      propagate();

      if (debug_cBitProp_messages)
        {
          cerr << "status:" << status <<endl;
          cerr << "ended propagation" << endl;
          printNodeWithFixings();
        }

      // propagate may have stopped with a conflict.
      if (CONFLICT == status)
          return top.GetSTPMgr()->CreateNode(FALSE);

      ASTVec toConjoin;

      // go through the fixedBits. If a node is entirely fixed.
      // "and" it onto the top. Creates redundancy. Check that the
      // node doesn't already depend on "top" directly.

      for (NodeToFixedBitsMap::NodeToFixedBitsMapType::iterator it = fixedMap->map->begin(); it != fixedMap->map->end(); it++) // iterates through all the pairs of node->fixedBits.
        {
          const FixedBits& bits = *it->second;

          if (!bits.isTotallyFixed())
            continue;

          const ASTNode& node = (it->first);

          // Don't constrain nodes we already know all about.
          if (node.isConstant())
            continue;

          // other nodes will contain the same information (the extract doesn't change the fixings).
          if (BVEXTRACT == node.GetKind() || BVCONCAT == node.GetKind())
            continue;

          // toAssign: conjoin it with the top level.
          // toReplace: replace all references to it (except the one conjoined to the top) with this.
          ASTNode propositionToAssert;
          ASTNode constantToReplaceWith;
          // skip the assigning and replacing.
          bool doAssign = true;

            {
              // If it is already contained in the fromTo map, then it's one of the values
              // that have fully been determined (previously). Not conjoined.
              if (fromTo.find(node) != fromTo.end())
                continue;

              ASTNode constNode = bitsToNode(node,bits);

              if (node.GetType() == BOOLEAN_TYPE)
                {
                  if (SYMBOL == node.GetKind())
                    {
                      bool r = simplifier->UpdateSubstitutionMap(node, constNode);
                      assert(r);
                      doAssign = false;
                    }
                  else if (bits.getValue(0))
                    {
                      propositionToAssert = node;
                      constantToReplaceWith = constNode;
                    }
                  else
                    {
                      propositionToAssert = nf->CreateNode(NOT, node);
                      constantToReplaceWith = constNode;
                    }
                }
              else if (node.GetType() == BITVECTOR_TYPE)
                {
                  assert(((unsigned)bits.getWidth()) == node.GetValueWidth());
                  if (SYMBOL == node.GetKind())
                    {
                      bool r = simplifier->UpdateSubstitutionMap(node, constNode);
                      assert(r);
                      doAssign = false;
                    }
                  else
                    {
                      propositionToAssert = nf->CreateNode(EQ, node, constNode);
                      constantToReplaceWith = constNode;
                    }
                }
              else
                FatalError("sadf234s");
            }

          if (doAssign && top != propositionToAssert
              && !dependents->nodeDependsOn(top, propositionToAssert))
            {
              assert(!constantToReplaceWith.IsNull());
              assert(constantToReplaceWith.isConstant());
              assert(propositionToAssert.GetType() == BOOLEAN_TYPE);
              assert(node.GetValueWidth() == constantToReplaceWith.GetValueWidth());

              fromTo.insert(make_pair(node, constantToReplaceWith));
              toConjoin.push_back(propositionToAssert);
            }
        }


     // Write the constants into the main graph.
      ASTNodeMap cache;
      ASTNode result = SubstitutionMap::replace(top, fromTo, cache,nf);

      if (0 != toConjoin.size())
        {
          result = nf->CreateNode(AND, result, toConjoin); // conjoin the new conditions.
        }

      assert(BVTypeCheck(result));
      assert(status != CONFLICT); // conflict should have been seen earlier.
      return result;
    }

    void
    notHandled(const Kind& k)
    {
      if (READ != k && WRITE != k)
      if (debug_cBitProp_messages)
        {
          cerr << "!" << k << endl;
        }
    }


    // add to the work list any nodes that take the result of the "n" node.
    void
    ConstantBitPropagation::scheduleUp(const ASTNode& n)
    {
      const set<ASTNode>* toAdd = dependents->getDependents(n);
      set<ASTNode>::iterator it = toAdd->begin();
      while (it != toAdd->end())
        {
          workList->push(*it);
          it++;
        }
    }

    void
    ConstantBitPropagation::scheduleDown(const ASTNode& n)
    {
      for (int i = 0; i < n.Degree(); i++)
        workList->push(n[i]);
    }

    void
    ConstantBitPropagation::scheduleNode(const ASTNode& n)
    {
      workList->push(n);
    }

    bool
    ConstantBitPropagation::checkAtFixedPoint(const ASTNode& n, ASTNodeSet & visited)
    {
      if (status == CONFLICT)
        return true; // can't do anything.

      if (visited.find(n) != visited.end())
        return true;

      visited.insert(n);

      // get the current for the children.
      vector<FixedBits> childrenFixedBits;
      childrenFixedBits.reserve(n.GetChildren().size());

      // get a copy of the current fixing from the cache.
      for (unsigned i = 0; i < n.GetChildren().size(); i++)
        {
          childrenFixedBits.push_back(*getCurrentFixedBits(n[i]));
        }
      FixedBits current = *getCurrentFixedBits(n);
      FixedBits newBits = *getUpdatedFixedBits(n);

      assert(FixedBits::equals(newBits, current));

      for (int i = 0; i < n.Degree(); i++)
        {
          if (!FixedBits::equals(*getUpdatedFixedBits(n[i]),
              childrenFixedBits[i]))
            {
              cerr << "Not fixed point";
              assert(false);
            }

          checkAtFixedPoint(n[i], visited);
        }
      return true;
    }

    void
    ConstantBitPropagation::propagate()
    {
      if (CONFLICT == status)
        return;

      assert(NULL != fixedMap);

      while (!workList->isEmpty())
        {
          // get the next node from the worklist.
          const ASTNode& n = workList->pop();

          assert (!n.isConstant()); // shouldn't get into the worklist..
          assert (CONFLICT != status); // should have stopped already.

          if (debug_cBitProp_messages)
            {
              cerr << "working on" << n.GetNodeNum() << endl;
            }

          // get a copy of the current fixing from the cache.
          FixedBits current = *getCurrentFixedBits(n);

          // get the current for the children.
          vector<FixedBits> childrenFixedBits;
          childrenFixedBits.reserve(n.GetChildren().size());

          // get a copy of the current fixing from the cache.
          for (unsigned i = 0; i < n.GetChildren().size(); i++)
            {
              childrenFixedBits.push_back(*getCurrentFixedBits(n[i]));
            }

          // derive the new ones.
          FixedBits newBits = *getUpdatedFixedBits(n);

          if (CONFLICT == status)
            return;

          // Not all transfer function update the status. But if they report NO_CHANGE. There really is no change.
          if (status != NO_CHANGE)
            {
              if (!FixedBits::equals(newBits, current)) // has been a change.
                {
                  assert(FixedBits::updateOK(current,newBits));

                  scheduleUp(n); // schedule everything that depends on n.
                  if (debug_cBitProp_messages)
                    {
                      cerr << "Changed " << n.GetNodeNum() << "from:" << current << "to:" << newBits << endl;
                    }
                }

              for (unsigned i = 0; i < n.GetChildren().size(); i++)
                {
                  if (!FixedBits::equals(*getCurrentFixedBits(n[i]), childrenFixedBits[i]))
                    {
                      assert(FixedBits::updateOK(childrenFixedBits[i], *getCurrentFixedBits(n[i])) );

                      if (debug_cBitProp_messages)
                        {
                          cerr << "Changed: " << n[i].GetNodeNum() << " from:" << childrenFixedBits[i] << "to:"
                              << *getCurrentFixedBits(n[i]) << endl;
                        }

                      assert(!n[i].isConstant());

                      // All the immediate parents of this child need to be rescheduled.
                      // Shouldn't reschuedule 'n' but it does.
                      scheduleUp(n[i]);

                      // Scheduling the child updates all the values that feed into it.
                      workList->push(n[i]);
                    }
                }
            }
        }
    }

    // get the current value from the map. If no value is in the map. Make a new value.
    FixedBits*
    ConstantBitPropagation::getCurrentFixedBits(const ASTNode& n)
    {
      assert (NULL != fixedMap);

      NodeToFixedBitsMap::NodeToFixedBitsMapType::iterator it = fixedMap->map->find(n);
      if (it != fixedMap->map->end())
        {
          return it->second;
        }

      int bw;
      if (0 == n.GetValueWidth())
        {
          bw = 1;
        }
      else
        {
          bw = n.GetValueWidth();
        }

      FixedBits* output = new FixedBits(bw, (BOOLEAN_TYPE == n.GetType()));

      if (BVCONST == n.GetKind() || BITVECTOR == n.GetKind())
        {
          // the CBV doesn't leak. it is a copy of the cbv inside the node.
          CBV cbv = n.GetBVConst();

          for (unsigned int j = 0; j < n.GetValueWidth(); j++)
            {
              output->setFixed(j, true);
              output->setValue(j, CONSTANTBV::BitVector_bit_test(cbv, j));
            }
        }
      else if (TRUE == n.GetKind())
        {
          output->setFixed(0, true);
          output->setValue(0, true);
        }
      else if (FALSE == n.GetKind())
        {
          output->setFixed(0, true);
          output->setValue(0, false);
        }

       fixedMap->map->insert(pair<ASTNode, FixedBits*> (n, output));
      return output;
    }

    // For the given node, update which bits are fixed.

    FixedBits*
    ConstantBitPropagation::getUpdatedFixedBits(const ASTNode& n)
    {
      FixedBits* output = getCurrentFixedBits(n);
      const Kind k = n.GetKind();

      if (n.isConstant())
        {
          assert(output->isTotallyFixed());
          return output;
        }

      if (SYMBOL == k)
        return output; // No transfer functions for these.

      vector<FixedBits*> children;
      const int numberOfChildren = n.GetChildren().size();
      children.reserve(numberOfChildren);

      for (int i = 0; i < numberOfChildren; i++)
        {
          children.push_back(getCurrentFixedBits(n.GetChildren()[i]));
        }

      assert(status != CONFLICT);
      status = dispatchToTransferFunctions(k, children, *output, n, msm);
      //result = dispatchToMaximallyPrecise(k, children, *output, n,msm);

      assert(((unsigned)output->getWidth()) == n.GetValueWidth() || output->getWidth() ==1);

      return output;
    }

    Result
    dispatchToTransferFunctions(const Kind k, vector<FixedBits*>& children,
        FixedBits& output, const ASTNode n, MultiplicationStatsMap * msm)
    {
      Result result = NO_CHANGE;

      assert(!n.isConstant());

#ifdef WITHCBITP

      Result(*transfer)(vector<FixedBits*>&, FixedBits&);

      switch (k)
        {
          case READ:
          case WRITE:
          // do nothing. Seems difficult to track properly.
          return NO_CHANGE;
          break;

#define MAPTFN(caseV, FN) case caseV: transfer = FN; break;

          // Shifting
          MAPTFN(BVLEFTSHIFT, bvLeftShiftBothWays)
          MAPTFN(BVRIGHTSHIFT, bvRightShiftBothWays)
          MAPTFN(BVSRSHIFT, bvArithmeticRightShiftBothWays)

          // Unsigned Comparison.
          MAPTFN(BVLT,bvLessThanBothWays)
          MAPTFN(BVLE,bvLessThanEqualsBothWays)
          MAPTFN(BVGT, bvGreaterThanBothWays)
          MAPTFN(BVGE, bvGreaterThanEqualsBothWays)

          // Signed Comparison.
          MAPTFN(BVSLT, bvSignedLessThanBothWays)
          MAPTFN(BVSGT,bvSignedGreaterThanBothWays)
          MAPTFN(BVSLE, bvSignedLessThanEqualsBothWays)
          MAPTFN(BVSGE, bvSignedGreaterThanEqualsBothWays)

          // Logic.
          MAPTFN(XOR,bvXorBothWays)
          MAPTFN(BVXOR, bvXorBothWays)
          MAPTFN(OR, bvOrBothWays)
          MAPTFN(BVOR, bvOrBothWays)
          MAPTFN(AND,bvAndBothWays)
          MAPTFN(BVAND,bvAndBothWays)
          MAPTFN(IFF, bvEqualsBothWays)
          MAPTFN(EQ, bvEqualsBothWays)
          MAPTFN(IMPLIES,bvImpliesBothWays)
          MAPTFN(NOT,bvNotBothWays)
          MAPTFN(BVNEG, bvNotBothWays)

          // OTHER
          MAPTFN(BVZX, bvZeroExtendBothWays)
          MAPTFN(BVSX, bvSignExtendBothWays)
          MAPTFN(BVUMINUS,bvUnaryMinusBothWays)
          MAPTFN(BVEXTRACT,bvExtractBothWays)
          MAPTFN(BVPLUS, bvAddBothWays)
          MAPTFN(BVSUB, bvSubtractBothWays)
          MAPTFN(ITE,bvITEBothWays)
          MAPTFN(BVCONCAT, bvConcatBothWays)

          case BVMULT: // handled specially later.
          case BVDIV:
          case BVMOD:
          case SBVDIV:
          case SBVREM:
          case SBVMOD:
          transfer = NULL;
          break;
          default:
            {
              notHandled(k);
              return NO_CHANGE;
            }
        }
#undef MAPTFN
      bool mult_like = false;

      // safe approximation to no overflow multiplication.
      if (k == BVMULT)
        {
          //MultiplicationStats ms;
          //result = bvMultiplyBothWays(children, output, n.GetSTPMgr(),&ms);
          result = bvMultiplyBothWays(children, output, n.GetSTPMgr());
          //		if (CONFLICT != result)
          //			msm->map[n] = ms;
          mult_like=true;
        }
      else if (k == BVDIV)
        {
          result = bvUnsignedDivisionBothWays(children, output, n.GetSTPMgr());
          mult_like=true;
        }
      else if (k == BVMOD)
        {
          result = bvUnsignedModulusBothWays(children, output, n.GetSTPMgr());
          mult_like=true;
        }
      else if (k == SBVDIV)
        {
          result = bvSignedDivisionBothWays(children, output, n.GetSTPMgr());
          mult_like=true;
        }
      else if (k == SBVREM)
        {
          result = bvSignedRemainderBothWays(children, output, n.GetSTPMgr());
          mult_like=true;
        }
      else if (k == SBVMOD)
        {
          result = bvSignedModulusBothWays(children, output, n.GetSTPMgr());
          mult_like=true;
        }
      else
      result = transfer(children, output);

      if (mult_like && output_mult_like)
        {
          cerr << output << "=";
          cerr << *children[0] << k;
          cerr << *children[1] << std::endl;
        }

#endif
      return result;

    }


  Result dispatchToMaximallyPrecise(const Kind k, vector<FixedBits*>& children,
      FixedBits& output, const ASTNode n)
    {
  #if WITHCBITP

      Signature signature;
      signature.kind = k;

      vector<FixedBits> childrenCopy;

      for (int i = 0; i < (int) children.size(); i++)
      childrenCopy.push_back(*(children[i]));
      FixedBits outputCopy(output);

      if (k == BVMULT)
        {
          // We've got some of multiply already implemented. So help it out by getting some done first.
          Result r = bvMultiplyBothWays(children, output, n.GetSTPMgr());
          if (CONFLICT == r)
          return CONFLICT;
        }

      bool bad = maxPrecision(children, output, k, n.GetSTPMgr());

      if (bad)
      return CONFLICT;

      if (!FixedBits::equals(outputCopy, output))
      return CHANGED;

      for (int i = 0; i < (int) children.size(); i++)
        {
          if (!FixedBits::equals(*(children[i]), childrenCopy[i]))
          return CHANGED;
        }

  #endif
      return NOT_IMPLEMENTED;
    }
  }
}

