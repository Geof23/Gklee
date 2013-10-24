/********************************************************************
 * AUTHORS: David L. Dill, Vijay Ganesh, Trevor Hansen
 *
 * BEGIN DATE: November, 2005
 *
 * LICENSE: Please view LICENSE file in the home dir of this Program
 ********************************************************************/
// -*- c++ -*-

#include <cmath>
#include <cassert>
#include "BitBlaster.h"
#include "AIG/BBNodeManagerAIG.h"
#include "BBNodeManagerASTNode.h"
#include "../simplifier/constantBitP/FixedBits.h"
#include "../simplifier/constantBitP/ConstantBitPropagation.h"
#include "../simplifier/constantBitP/NodeToFixedBitsMap.h"

namespace BEEV {

/********************************************************************
 * BitBlast
 *
 * Convert bitvector terms and formulas to boolean formulas.  A term
 * is something that can represent a multi-bit bitvector, such as
 * BVPLUS or BVXOR (or a BV variable or constant).  A formula (form)
 * represents a boolean value, such as EQ or BVLE.  Bit blasting a
 * term representing an n-bit bitvector with BBTerm yields a vector
 * of n boolean formulas (returning ASTVec).  Bit blasting a formula
 * returns a single boolean formula (type ASTNode).  A bitblasted
 * term is a vector of ASTNodes for formulas.  The 0th element of
 * the vector corresponds to bit 0 -- the low-order bit.
 ********************************************************************/

using simplifier::constantBitP::FixedBits;

#define BBNodeVec vector<BBNode>
#define BBNodeVecMap map<ASTNode, vector<BBNode> >
#define BBNodeSet set<BBNode>

vector<BBNodeAIG> _empty_BBNodeAIGVec;

// Bit blast a bitvector term.  The term must have a kind for a
// bitvector term.  Result is a ref to a vector of formula nodes
// representing the boolean formula.

// This prints out each constant expression that the bitblaster
// discovers. I use this to check that the expressions that are
// reaching the bitblaster don't have obvious simplifications
// that should have already been applied.
const bool debug_do_check = false;
const bool debug_bitblaster = false;

template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::commonCheck(const ASTNode& n) {
        cerr << "Non constant is constant:";
        cerr << n << endl;

        if (cb == NULL)
                return;
        if (cb->fixedMap->map->find(n) != cb->fixedMap->map->end()) {
                FixedBits* b = cb->fixedMap->map->find(n)->second;
                cerr <<"fixed bits are:"<< *b << endl;
        }
}

// If x isn't a constant, and the bit-blasted version is. Print out the
// AST nodes and the fixed bits.
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::check(const BBNode& x, const ASTNode& n) {
        if (n.isConstant())
                return;

        if (x != BBTrue && x != BBFalse)
                return;

        commonCheck(n);
}

template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::check(const vector<BBNode>& x, const ASTNode& n) {
        if (n.isConstant())
                return;

        for (int i = 0; i < (int) x.size(); i++) {
                if (x[i] != BBTrue && x[i] != BBFalse)
                        return;
        }

        commonCheck(n);
}

template <class BBNode, class BBNodeManagerT>
bool BitBlaster<BBNode,BBNodeManagerT>::update(const ASTNode&n, const int i, simplifier::constantBitP::FixedBits* b, BBNode& bb,  BBNodeSet& support)
{
        if (b->isFixed(i) && (!(bb == BBTrue || bb == BBFalse)))
        {
                //We have a fixed bit, but the bitblasted values aren't constant true or false.

                //if (b->getValue(i))
                        //support.insert(bb);
                //else
                  //      support.insert(nf->CreateNode(NOT,bb));

                bb = b->getValue(i) ? BBTrue : BBFalse;
        }
        else if (!b->isFixed(i) && (bb == BBTrue || bb == BBFalse))
        {
                b->setFixed(i,true);
                b->setValue(i,bb == BBTrue ? true : false);
                return true; // Need to propagate.
        }

        return false;
}

template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::updateForm(const ASTNode&n, BBNode& bb, BBNodeSet& support)
{
  if (cb == NULL || n.isConstant())
          return;

  BBNodeVec v(1,bb);
  updateTerm(n, v, support);
  bb = v[0];
}

template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::updateTerm(const ASTNode&n, BBNodeVec& bb, BBNodeSet& support) {

        if (cb == NULL)
                return;

        if (n.isConstant())
          {
              simplifier::constantBitP::NodeToFixedBitsMap::NodeToFixedBitsMapType::const_iterator it;
              it = cb->fixedMap->map->find(n);
              if(it == cb->fixedMap->map->end())
                {
                cerr << n;
                assert(it != cb->fixedMap->map->end());
                }
              assert(it->second->isTotallyFixed());
              return;
          }


        bool bbFixed  = false;
        for (int i =0; i < (int)bb.size(); i++)
        {
                if (bb[i] == BBTrue || bb[i] == BBFalse)
                {
                        bbFixed = true;
                        break;
                }
        }

        FixedBits * b = NULL;

        simplifier::constantBitP::NodeToFixedBitsMap::NodeToFixedBitsMapType::const_iterator it;
        if ((it = cb->fixedMap->map->find(n)) == cb->fixedMap->map->end())
        {
                if (bbFixed)
                {
                        b = new FixedBits(n.GetType() == BOOLEAN_TYPE? 1:n.GetValueWidth(),n.GetType() == BOOLEAN_TYPE);
                        cb->fixedMap->map->insert(pair<ASTNode, FixedBits*> (n, b));
                        if (debug_bitblaster)
                                cerr << "inserting" << n.GetNodeNum() << endl;
                }
                else
                        return; // nothing to update.
        }
        else
                b = it->second;

        assert(b != NULL);

        bool changed = false;
        for (int i = 0; i < (int)bb.size(); i++)
                if(update(n,i, b, bb[i], support))
                  changed = true; // don't break, we want to run update(..) on each bit.
        if (changed) {
                //cerr <<  "NN" << n.GetNodeNum() << endl;
                //cerr << *fixedBits;
                cb->scheduleNode(n);
                cb->scheduleUp(n);
                //cerr << "##!" << endl;
                cb->propagate();
                //cerr << "##!!" << endl;
        }
}


template <class BBNode, class BBNodeManagerT>
const BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBTerm(const ASTNode& term, BBNodeSet& support) {
        typename BBNodeVecMap::iterator it = BBTermMemo.find(term);
	if (it != BBTermMemo.end()) {
		// already there.  Just return it.
		return it->second;
	}

	BBNodeVec result;

	const Kind k = term.GetKind();
	if (!is_Term_kind(k))
		FatalError("BBTerm: Illegal kind to BBTerm", term);

	const ASTVec::const_iterator kids_end = term.end();
	const unsigned int num_bits = term.GetValueWidth();
	switch (k) {
	case BVNEG: {
		// bitwise complement
		const BBNodeVec& bbkids = BBTerm(term[0], support);
		result = BBNeg(bbkids);
		break;
	}

	case BVRIGHTSHIFT:
	case BVSRSHIFT:
	case BVLEFTSHIFT: {
		// Barrel shifter
		const BBNodeVec& bbarg1 = BBTerm(term[0], support);
		const BBNodeVec& bbarg2 = BBTerm(term[1], support);

		// Signed right shift, need to copy the sign bit.
		BBNode toFill;
		if (BVSRSHIFT == k)
			toFill = bbarg1.back();
		else
			toFill = nf->getFalse();

		BBNodeVec temp_result(bbarg1);
		// if any bit is set in bbarg2 higher than log2Width, then we know that the result is zero.
		// Add one to make allowance for rounding down. For example, given 300 bits, the log2 is about
		// 8.2 so round up to 9.

		const unsigned width = bbarg1.size();
		const unsigned log2Width = (unsigned) log2(width) + 1;

		if (k == BVSRSHIFT || k == BVRIGHTSHIFT)
			for (unsigned int i = 0; i < log2Width; i++) {
				if (bbarg2[i] == nf->getFalse())
					continue; // Not shifting by anything.

				unsigned int shift_amount = 1 << i;

				for (unsigned int j = 0; j < width; j++) {
					if (j + shift_amount >= width)
						temp_result[j] = nf->CreateNode(ITE, bbarg2[i], toFill,
								temp_result[j]);
					else
						temp_result[j] = nf->CreateNode(ITE, bbarg2[i],
								temp_result[j + shift_amount], temp_result[j]);
				}
			}
		else
			for (unsigned int i = 0; i < log2Width; i++) {
				if (bbarg2[i] == nf->getFalse())
					continue; // Not shifting by anything.

				int shift_amount = 1 << i;

				for (signed int j = width - 1; j >= 0; j--) {
					if (j < shift_amount)
						temp_result[j] = nf->CreateNode(ITE, bbarg2[i], toFill,
								temp_result[j]);
					else
						temp_result[j] = nf->CreateNode(ITE, bbarg2[i],
								temp_result[j - shift_amount], temp_result[j]);
				}
			}

		// If any of the remainder are true. Then the whole thing gets the fill value.
		BBNode remainder = nf->getFalse();
		for (unsigned int i = log2Width; i < width; i++) {
			remainder = nf->CreateNode(OR, remainder, bbarg2[i]);
		}

		for (unsigned int i = 0; i < width; i++) {
			temp_result[i] = nf->CreateNode(ITE, remainder, toFill,
					temp_result[i]);
		}

		result = temp_result;
	}
		break;

	case ITE: {
		// Term version of ITE.
		const BBNode& cond = BBForm(term[0], support);
		const BBNodeVec& thn = BBTerm(term[1], support);
		const BBNodeVec& els = BBTerm(term[2], support);
		result = BBITE(cond, thn, els);
		break;
	}

	case BVSX:
	case BVZX:
	  {
		// Replicate high-order bit as many times as necessary.
		// Arg 0 is expression to be sign extended.
		const ASTNode& arg = term[0];
		const unsigned long result_width = term.GetValueWidth();
		const unsigned long arg_width = arg.GetValueWidth();
		const BBNodeVec& bbarg = BBTerm(arg, support);

		if (result_width == arg_width) {
			//nothing to sign extend
			result = bbarg;
			break;
		} else {
			//we need to sign extend
			const BBNode& msb = (k == BVSX) ?bbarg.back() : BBFalse;

			BBNodeVec tmp_res(result_width);

			typename BBNodeVec::const_iterator bb_it = bbarg.begin();
			typename  BBNodeVec::iterator res_it = tmp_res.begin();
			typename  BBNodeVec::iterator res_ext = res_it + arg_width; // first bit of extended part
			typename BBNodeVec::iterator res_end = tmp_res.end();

			// copy LSBs directly from bbvec
			for (; res_it < res_ext; (res_it++, bb_it++)) {
				*res_it = *bb_it;
			}
			// repeat MSB to fill up rest of result.
			for (; res_it < res_end; (res_it++)) {
				*res_it = msb;
			}

			result = tmp_res;
			break;
		}
	}

	case BVEXTRACT: {
		// bitblast the child, then extract the relevant bits.
		// Note: This could be optimized by not bitblasting the bits
		// that aren't fetched.  But that would be tricky, especially
		// with memo-ization.

		const BBNodeVec& bbkids = BBTerm(term[0], support);
		const unsigned int high = term[1].GetUnsignedConst();
		const unsigned int low = term[2].GetUnsignedConst();

		typename BBNodeVec::const_iterator bbkfit = bbkids.begin();
		// I should have used pointers to BBNodeVec, to avoid this crock

		result = BBNodeVec(bbkfit + low, bbkfit + high + 1);
		break;
	}
	case BVCONCAT: {
		const BBNodeVec& vec1 = BBTerm(term[0], support);
		const BBNodeVec& vec2 = BBTerm(term[1], support);

		BBNodeVec tmp_res(vec2);
		tmp_res.insert(tmp_res.end(), vec1.begin(), vec1.end());
		result = tmp_res;
		break;
	}
	case BVPLUS: {
		assert(term.Degree() >=1);
		// Add children pairwise and accumulate in BBsum

		ASTVec::const_iterator it = term.begin();
		BBNodeVec tmp_res = BBTerm(*it, support);
		for (++it; it < kids_end; it++) {
			const BBNodeVec& tmp = BBTerm(*it, support);
			assert(tmp.size() == num_bits);
			BBPlus2(tmp_res, tmp, nf->getFalse());
		}

		result = tmp_res;
		break;
	}
	case BVUMINUS: {
		const BBNodeVec& bbkid = BBTerm(term[0], support);
		result = BBUminus(bbkid);
		break;
	}
	case BVSUB: {
		// complement of subtrahend
		// copy, since BBSub writes into it.

		BBNodeVec tmp_res = BBTerm(term[0], support);

		const BBNodeVec& bbkid1 = BBTerm(term[1], support);
		BBSub(tmp_res, bbkid1, support);
		result = tmp_res;
		break;
	}
	case BVMULT: {
		assert(BVTypeCheck(term));

		const ASTNode& t0 = term[0];
		const ASTNode& t1 = term[1];

		const BBNodeVec& mpcd1 = BBTerm(t0, support);
		const BBNodeVec& mpcd2 = BBTerm(t1, support);
		assert(mpcd1.size() == mpcd2.size());
		//Revereses the order of the nodes w/out the need for temporaries
		//This is needed because t0 an t1 must be const
		if ((BVCONST != t0.GetKind()) && (BVCONST == t1.GetKind())) {

			result = BBMult(mpcd2, mpcd1, support,t1,t0);
		} else {
			result = BBMult(mpcd1, mpcd2, support,t0,t1);
		}
		break;
	}
	case BVDIV:
	case BVMOD: {
		const BBNodeVec& dvdd = BBTerm(term[0], support);
		const BBNodeVec& dvsr = BBTerm(term[1], support);
		assert (dvdd.size() == num_bits);
		assert (dvsr.size() == num_bits);
		BBNodeVec q(num_bits);
		BBNodeVec r(num_bits);
		BBDivMod(dvdd, dvsr, q, r, num_bits, support);
		if (k == BVDIV)
			result = q;
		else
			result = r;
		break;
	}
		//  n-ary bitwise operators.
	case BVXOR:
	case BVXNOR:
	case BVAND:
	case BVOR:
	case BVNOR:
	case BVNAND: {
		// Add children pairwise and accumulate in BBsum
		ASTVec::const_iterator it = term.begin();
		Kind bk = UNDEFINED; // Kind of individual bit op.
		switch (k) {
		case BVXOR:
			bk = XOR;
			break;
		case BVXNOR:
			bk = IFF;
			break;
		case BVAND:
			bk = AND;
			break;
		case BVOR:
			bk = OR;
			break;
		case BVNOR:
			bk = NOR;
			break;
		case BVNAND:
			bk = NAND;
			break;
		default:
			FatalError("BBTerm: Illegal kind to BBTerm", term);
			break;
		}

		// Sum is destructively modified in the loop, so make a copy of value
		// returned by BBTerm.
		BBNodeVec temp = BBTerm(*it, support);
		BBNodeVec sum(temp); // First operand.

		// Iterate over remaining bitvector term operands
		for (++it; it < kids_end; it++) {
			//FIXME FIXME FIXME: Why does using a temp. var change the behavior?
			temp = BBTerm(*it, support);
			const BBNodeVec& y = temp;

			assert(y.size() == num_bits);
			for (unsigned i = 0; i < num_bits; i++) {
				sum[i] = nf->CreateNode(bk, sum[i], y[i]);
			}
		}
		result = sum;
		break;
	}
	case SYMBOL: {
		assert(num_bits >0);

		BBNodeVec bbvec;
		bbvec.reserve(num_bits);

		for (unsigned int i = 0; i < num_bits; i++) {
			BBNode bit_node = nf->CreateSymbol(term, i);
			bbvec.push_back(bit_node);
		}
		result = bbvec;
		break;
	}
	case BVCONST: {
		BBNodeVec tmp_res(num_bits);
		CBV bv = term.GetBVConst();
		for (unsigned int i = 0; i < num_bits; i++) {
			tmp_res[i] = CONSTANTBV::BitVector_bit_test(bv, i) ? nf->getTrue()
					: nf->getFalse();
		}
		result = tmp_res;
		break;
	}
	default:
		FatalError("BBTerm: Illegal kind to BBTerm", term);
	}

	assert(result.size() == term.GetValueWidth());

	if (debug_do_check)
		check(result, term);

	updateTerm(term,result,support);
	return (BBTermMemo[term] = result);
}

template <class BBNode, class BBNodeManagerT>
const BBNode BitBlaster<BBNode,BBNodeManagerT>::BBForm(const ASTNode& form)
{
    BBNodeSet support;
    BBNode r= BBForm(form,support);
    //vector<BBNode> v;
    //v.insert(v.end(), support.begin(), support.end());
    //v.push_back(r);
    assert(support.size() ==0);


    if (cb != NULL && !cb->isUnsatisfiable())
      {
      ASTNodeSet visited;
      assert(cb->checkAtFixedPoint(form,visited));
      }
    //    return nf->CreateNode(AND,v);
    return r;
}

// bit blast a formula (boolean term).  Result is one bit wide,
template <class BBNode, class BBNodeManagerT>
const BBNode BitBlaster<BBNode,BBNodeManagerT>::BBForm(const ASTNode& form, BBNodeSet& support) {
	typename map<ASTNode,BBNode>::iterator it = BBFormMemo.find(form);
	if (it != BBFormMemo.end()) {
		// already there.  Just return it.
		return it->second;
	}

	BBNode result;

	const Kind k = form.GetKind();
	if (!is_Form_kind(k)) {
		FatalError("BBForm: Illegal kind: ", form);
	}

	//  Not returning until end, and memoizing everything, makes it easier
	// to trace coherently.

	// Various special cases
	switch (k) {

	case TRUE: {
		result = nf->getTrue();
		break;
	}

	case FALSE: {
		result = nf->getFalse();
		break;
	}

	case SYMBOL:
		assert (form.GetType() == BOOLEAN_TYPE);

		result = nf->CreateSymbol(form, 0); // 1 bit symbol.
		break;

	case NOT:
		result = nf->CreateNode(NOT, BBForm(form[0], support));
		break;

	case ITE:
		result = nf->CreateNode(ITE, BBForm(form[0], support), BBForm(form[1],
				support), BBForm(form[2], support));
		break;

	case AND:
	case OR:
	case NAND:
	case NOR:
	case IFF:
	case XOR:
	case IMPLIES: {
		BBNodeVec bbkids; // bit-blasted children (formulas)

		ASTVec::const_iterator kids_end = form.end();
		for (ASTVec::const_iterator it = form.begin(); it != kids_end; it++) {
			bbkids.push_back(BBForm(*it, support));
		}
		result = nf->CreateNode(k, bbkids);
		break;
	}

	case EQ: {
		const BBNodeVec left = BBTerm(form[0], support);
		const BBNodeVec right = BBTerm(form[1], support);
		assert (left.size() == right.size());

		result = BBEQ(left, right);
		break;
	}

	case BVLE:
	case BVGE:
	case BVGT:
	case BVLT:
	case BVSLE:
	case BVSGE:
	case BVSGT:
	case BVSLT: {
		result = BBcompare(form, support);
		break;
	}
	default:
		FatalError("BBForm: Illegal kind: ", form);
		break;
	}

	assert(!result.IsNull());

	if (debug_do_check)
		check(result, form);

	updateForm(form,result,support);

	return (BBFormMemo[form] = result);
}

// Bit blast a sum of two equal length BVs.
// Update sum vector destructively with new sum.
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::BBPlus2(BBNodeVec& sum, const BBNodeVec& y, BBNode cin) {

	const int n = sum.size();
	assert(y.size() == (unsigned)n);
	// Revision 320 avoided creating the nextcin, at I suspect unjustified cost.
	for (int i = 0; i < n; i++) {
		BBNode nextcin = Majority(sum[i], y[i], cin);
		sum[i] = nf->CreateNode(XOR, sum[i], y[i], cin);
		cin = nextcin;
	}
}

// Stores result - x in result, destructively
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::BBSub(BBNodeVec& result, const BBNodeVec& y,
		BBNodeSet& support) {
	BBNodeVec compsubtrahend = BBNeg(y);
	BBPlus2(result, compsubtrahend, nf->getTrue());
}

// Add one bit
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBAddOneBit(const BBNodeVec& x, BBNode cin) {
	BBNodeVec result;
	result.reserve(x.size());
	const typename BBNodeVec::const_iterator itend = x.end();
	for (typename BBNodeVec::const_iterator it = x.begin(); it < itend; it++) {
		BBNode nextcin = nf->CreateNode(AND, *it, cin);
		result.push_back(nf->CreateNode(XOR, *it, cin));
		cin = nextcin;
	}
	return result;
}

// Increment bit-blasted vector and return result.
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBInc(const BBNodeVec& x) {
	return BBAddOneBit(x, nf->getTrue());
}

// Return formula for majority function of three bits.
// Pass arguments by reference to reduce refcounting.
template <class BBNode, class BBNodeManagerT>
BBNode BitBlaster<BBNode,BBNodeManagerT>::Majority(const BBNode& a, const BBNode& b,
		const BBNode& c) {
	// Checking explicitly for constant a, b and c could
	// be more efficient, because they are repeated in the logic.
	if (nf->getTrue() == a) {
		return nf->CreateNode(OR, b, c);
	} else if (nf->getFalse() == a) {
		return nf->CreateNode(AND, b, c);
	} else if (nf->getTrue() == b) {
		return nf->CreateNode(OR, a, c);
	} else if (nf->getFalse() == b) {
		return nf->CreateNode(AND, a, c);
	} else if (nf->getTrue() == c) {
		return nf->CreateNode(OR, a, b);
	} else if (nf->getFalse() == c) {
		return nf->CreateNode(AND, a, b);
	}
	// there are lots more simplifications, but I'm not sure they're
	// worth doing explicitly (e.g., a = b, a = ~b, etc.)
	else {
		return nf->CreateNode(OR, nf->CreateNode(AND, a, b), nf->CreateNode(
				AND, b, c), nf->CreateNode(AND, a, c));
	}
}

// Bitwise complement
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBNeg(const BBNodeVec& x) {
	BBNodeVec result;
	result.reserve(x.size());
	// Negate each bit.
	const typename BBNodeVec::const_iterator& xend = x.end();
	for (typename BBNodeVec::const_iterator it = x.begin(); it < xend; it++) {
		result.push_back(nf->CreateNode(NOT, *it));
	}
	return result;
}

// Compute unary minus
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBUminus(const BBNodeVec& x) {
	BBNodeVec xneg = BBNeg(x);
	return BBInc(xneg);
}

// AND each bit of vector y with single bit b and return the result.
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBAndBit(const BBNodeVec& y, BBNode b) {
	if (nf->getTrue() == b) {
		return y;
	}

	BBNodeVec result;
	result.reserve(y.size());

	const typename BBNodeVec::const_iterator yend = y.end();
	for (typename BBNodeVec::const_iterator yit = y.begin(); yit < yend; yit++) {
		result.push_back(nf->CreateNode(AND, *yit, b));
	}
	return result;
}

typedef enum {SYMBOL_MT, ZERO_MT, ONE_MT, MINUS_ONE_MT} mult_type;

void printP(mult_type* m, int width)
{
	for (int i =width-1; i >=0;i--)
	{
		if (m[i] == SYMBOL_MT)
		cerr << "s";
		else if (m[i] == ZERO_MT)
			cerr << "0";
		else if (m[i] == ONE_MT)
					cerr << "1";
		else if (m[i] == MINUS_ONE_MT)
					cerr << "-1";
	}
}

template <class BBNode, class BBNodeManagerT>
void convert(const BBNodeVec& v, BBNodeManagerT*nf, mult_type* result)
{
	const BBNode& BBTrue = nf->getTrue();
	const BBNode& BBFalse = nf->getFalse();

	for (int i =0; i < v.size(); i++)
	{
		if (v[i] == BBTrue)
			result[i] = ONE_MT;
		else if (v[i] == BBFalse)
			result[i] = ZERO_MT;
		else
			result[i] = SYMBOL_MT;
	}

	// find runs of ones.
	int lastOne=-1;
	for (int i =0; i < v.size(); i++)
	{
		assert(result[i] != MINUS_ONE_MT);

		if (result[i] == ONE_MT && lastOne == -1)
			lastOne = i;

		if (result[i] != ONE_MT && lastOne != -1 && (i-lastOne >=3))
		{
			result[lastOne] = MINUS_ONE_MT;
			for (int j = lastOne+1; j<i;j++)
				result[j] = ZERO_MT;
			// Should this be lastOne = i?
			lastOne = i;
			result[i] = ONE_MT;
		} else 	if (result[i] != ONE_MT)
			lastOne = -1;
	}

	// finished with a one.
	if (lastOne != -1 && (v.size() -lastOne >1))
	{
		result[lastOne] = MINUS_ONE_MT;
		for (int j = lastOne+1; j< v.size();j++)
			result[j] = ZERO_MT;
	}
}

// Multiply "multiplier" by y[start ... bitWidth].
template <class BBNode, class BBNodeManagerT>
void pushP(stack<BBNode> *products, const int start, const BBNodeVec& y, const BBNode& multiplier, BBNodeManagerT*nf)
{
	const int bitWidth = y.size();

	int c = 0;
	for (int i = start; i < bitWidth; i++)
	{
		BBNode n = nf->CreateNode(AND, y[c], multiplier);
		products[i].push(n);
		c++;
	}
}

const bool debug_multiply = false;

template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::pairWiseAdd(stack<BBNode>* products,
		const int bitWidth)
{
	const BBNode& BBFalse  = nf->getFalse();
	BBNodeVec prod = BBfill(bitWidth, BBFalse);

	bool finished = false;
	while (!finished) {
		finished = true;
		BBNodeVec a;
		for (int i = 0; i < bitWidth; i++)
		{
			if (products[i].empty())
				a.push_back(BBFalse);
			else
			{
				BBNode t = products[i].top();
				a.push_back(t);
				products[i].pop();
				finished = false;
			}
		}
		BBPlus2(prod, a, nf->getFalse());
	}

	return prod;
}



// Use full adders to create an addition network that adds together each of the
// partial products.
template <class BBNode, class BBNodeManagerT>
BBNodeVec buildAdditionNetworkResult(stack<BBNode>* products,
		const int bitWidth, BBNodeManagerT* nf) {
	int adderCount = 0;

	// I experimented with sorting the partial products to put the known values together.
	// But it didn't help.

	BBNodeVec result;

	for (int i = 0; i < bitWidth; i++) {
		while (products[i].size() >= 2) {
			BBNode c;

			if (products[i].size() == 2)
				c = nf->getFalse();
			else {
				c = products[i].top();
				products[i].pop();
			}

			const BBNode a = products[i].top();
			products[i].pop();
			const BBNode b = products[i].top();
			products[i].pop();

			BBNode carry = nf->CreateNode(OR, nf->CreateNode(AND, a, b),
					nf->CreateNode(AND, b, c), nf->CreateNode(AND, a, c));
			// I tested all 6 ternary XORs, and all 12 2 x 2-arity. The formulae with "a" separate were quickest.
			BBNode sum = nf->CreateNode(XOR, nf->CreateNode(XOR, c, b), a);
			adderCount++;

			if (debug_multiply) {
				cerr << "Column " << i << endl;
				cerr << "a" << a;
				cerr << "b" << b;
				cerr << "c" << c;
				cerr << "Carry" << carry;
				cerr << "Sum" << sum;
			}

			products[i].push(sum);
			if (i + 1 != bitWidth)
				products[i + 1].push(carry);
		}

		assert(1==products[i].size());
		result.push_back(products[i].top());
	}

	if (debug_multiply)
		cerr << "adder count" << adderCount << endl;
	assert(result.size() == ((unsigned)bitWidth));
	return result;
}

template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::mult_Booth(const BBNodeVec& x_i, const BBNodeVec& y_i, BBNodeSet& support, const ASTNode& xN, const ASTNode& yN, stack<BBNode> * products)
{
	 const int bitWidth = x_i.size();
	 assert(x_i.size() == y_i.size());

	 const BBNodeVec& x = x_i;
	 const BBNodeVec& y = y_i;

	 const BBNode& BBTrue = nf->getTrue();
	 const BBNode& BBFalse = nf->getFalse();


	 for (int i =0 ; i < bitWidth;i++)
	 {
		 products[i].push(BBFalse);
	 }

	 BBNodeVec notY;
	 for (int i =0 ; i < y.size();i++)
	 {
		 notY.push_back(nf->CreateNode(NOT,y[i]));
	 }


	 mult_type xt[x.size()];
	 convert(x,nf,xt);

	 if (debug_multiply)
	 {
		 cerr << "--------" << endl;
		 cerr << "x:";
		 printP(xt,x.size());
		 cerr << xN << endl;
	 }

	 mult_type yt[x.size()];
	 convert(y,nf,yt);

	 if (debug_multiply)
	 {
		 cerr << "y:";
		 printP(yt,y.size());
		 cerr << yN << endl;
	 }

	 for (int i =0; i < bitWidth;i++)
	 {
		 if (x[i] != BBTrue && x[i] != BBFalse)
		 {
			 pushP(products,i,y,x[i],nf);
		 }

		 if (xt[i] == MINUS_ONE_MT)
		 {
			 pushP(products,i,notY,BBTrue,nf);
			 products[i].push(BBTrue);
		 }

		 if (xt[i] == ONE_MT)
		 {
			 pushP(products,i,y,BBTrue,nf);
		 }
	 }
  }


// Uses addition networks explicitly.
// I've copied this in from my the "trevor" branch r482.
// I've not measured if this is better than the current variant.
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::mult_allPairs(const BBNodeVec& x, const BBNodeVec& y, BBNodeSet& support, stack<BBNode> * products)
  {
	 // Make a table of partial products.
	 const int bitWidth = x.size();
	 assert(x.size() == y.size());


	  for (int i =0 ; i < bitWidth; i++)
	  {
		  for (int j=0; j<= i; j++)
			  {
			  BBNode n = nf->CreateNode(AND, y[j], x[i-j]);
			  products[i].push(n);
			  }
	  }
	  // I experimented with sorting the partial products to put the known values together.
	  // But it didn't help.
  }

template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::mult_normal(const BBNodeVec& x,
		const BBNodeVec& y, BBNodeSet& support) {
	BBNodeVec ycopy(y);
	const typename BBNodeVec::const_iterator xend = x.end();
	typename BBNodeVec::const_iterator xit = x.begin();
	// start prod with first partial product.
	BBNodeVec prod = BBNodeVec(BBAndBit(y, *xit));
	// start loop at next bit.
	for (xit++; xit < xend; xit++) {
		// shift first
		BBLShift(ycopy, 1);

		if (nf->getFalse() == *xit) {
			// If this bit is zero, the partial product will
			// be zero.  No reason to add that in.
			continue;
		}

		BBNodeVec pprod = BBAndBit(ycopy, *xit);
		// accumulate in the product.
		BBPlus2(prod, pprod, nf->getFalse());
	}
	return prod;
}

// ONLY SELECT ONE OF THESE!
const bool multiplication_variant1 = false; // multiplication with repeat addition.
const bool multiplication_variant2 = false; // multiplication with partial products.
const bool multiplication_variant3 = true; // multiplication with booth recoding.

// Multiply two bitblasted numbers
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBMult(const BBNodeVec& x, const BBNodeVec& y,
		BBNodeSet& support, const ASTNode& xN, const ASTNode& yN) {

	if (multiplication_variant1) {
		return mult_normal(x, y, support);
	}

	if (multiplication_variant2) {
		const int bitWidth = x.size();
		stack<BBNode> products[bitWidth];
		mult_allPairs(x, y, support,products);
		return buildAdditionNetworkResult(products,bitWidth,nf);
	}

	if (multiplication_variant3) {
		const int bitWidth = x.size();
		stack<BBNode> products[bitWidth];
		mult_Booth(x, y, support,xN,yN,products);
		//return pairWiseAdd(products,bitWidth);
		return buildAdditionNetworkResult(products,bitWidth,nf);
	}

	FatalError("sda44f");
}

// on factoring12bitsx12.cvc
// variant1 = t, variant2 = t: 66 seconds
// variant1 = t, variant2 = f: 37 seconds
// variant1 = f, variant2 = f: 46 seconds
// variant1 = f, variant2 = t: 65 seconds

// You can select these with any combination you want of true & false.
const bool division_variant_1 = true;
const bool division_variant_2 = false;

// This implements a variant of binary long division.
// q and r are "out" parameters.  rwidth puts a bound on the
// recursion depth.
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::BBDivMod(const BBNodeVec &y, const BBNodeVec &x,
		BBNodeVec &q, BBNodeVec &r, unsigned int rwidth, BBNodeSet& support) {
	const unsigned int width = y.size();
	const BBNodeVec zero = BBfill(width, nf->getFalse());
	BBNodeVec one = zero;
	one[0] = nf->getTrue();

	// check if y is already zero.
	bool isZero=true;
	for (int i =0; i < rwidth;i++)
          if (y[i] != nf->getFalse())
          {
            isZero = false;
            break;
            }


	if (isZero || rwidth == 0) {
		// When we have shifted the entire width, y is guaranteed to be 0.
		q = zero;
		r = zero;
	} else {
		BBNodeVec q1, r1;
		BBNodeVec yrshift1(y);
		BBRShift(yrshift1, 1);

		// recursively divide y/2 by x.
		BBDivMod(yrshift1, x, q1, r1, rwidth - 1, support);

		BBNodeVec q1lshift1(q1);
		BBLShift(q1lshift1, 1);

		BBNodeVec r1lshift1(r1);
		BBLShift(r1lshift1, 1);

		BBNodeVec r1lshift1plusyodd(r1lshift1);
		r1lshift1plusyodd[0] = y[0];

                // By extending rminusx by one bit, we can use that top-bit
                // to check whether r>=x. We need to compute rminusx anyway,
                // so this saves having a separate compare circut.
                BBNodeVec rminusx(r1lshift1plusyodd);
                rminusx.push_back(nf->getFalse());
                BBNodeVec xCopy(x);
                xCopy.push_back(nf->getFalse());
                BBSub(rminusx, xCopy, support);
                BBNode sign = rminusx[width];
                rminusx.pop_back();

                // Adjusted q, r values when when r is too large.
                BBNodeVec ygtrxqval = BBITE(sign, q1lshift1, BBInc(q1lshift1));
                BBNodeVec ygtrxrval = BBITE(sign, r1lshift1plusyodd, rminusx );

		BBNodeVec notylessxqval;
		BBNodeVec notylessxrval;


		/* variant_1 removes the equality check of (x=y). When we get to here I think
		 that r and q already have the proper values.
		 Let x =y, so we are solving y/y.
		 As a first step solve (y/2)/y.
		 If y != 0, then y/2 < y. (strictly less than).
		 By the last rule in this block, that will return q=0, r=(y/2)
		 On return, that will be rightshifted, and the parity bit added back,
		 giving q = 0, r=y.
		 By the immediately preceeding rule, 0 <= 0 is true, so q becomes 1,
		 and r becomes 0.
		 So q and r are already set correctly when we get here.

		 If y=0 & x=0, then (y/2)/0 will return q=0, r=0.
		 By the preceeding rule  0 <= 0 is true, so q =1, r=0.
		 So q and r are already set correctly when we get here.
		 */

		if (division_variant_1)
		{
			notylessxqval = ygtrxqval;
			notylessxrval = ygtrxrval;
		}
		else
		{
			// q & r values when y >= x
			BBNode yeqx = BBEQ(y, x);
			// *** Problem: the bbfill for qval is wrong.  Should be 1, not -1.
			notylessxqval = BBITE(yeqx, one, ygtrxqval);
			notylessxrval = BBITE(yeqx, zero,ygtrxrval);
		}

		/****************/
		BBNode ylessx;
		if (division_variant_2)
		{
			ylessx = BBBVLE(y, x, false, true);
		}
		else
		{
			// y < x <=> not x >= y.
			ylessx = nf->CreateNode(NOT, BBBVLE(x, y, false));
		}

		// final values of q and r
		q = BBITE(ylessx, zero, notylessxqval);
		r = BBITE(ylessx, y, notylessxrval);

                // The above does nothing. This gives the correct answer too:
                //q = notylessxqval;
                //r = notylessxrval;

                // But, it's about 30% slower on factoring12x12.
                // Either these help during solving, or, the CNF generation
                // generates nicer code with them present.


		/****************/
	}
}

// build ITE's (ITE cond then[i] else[i]) for each i.
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBITE(const BBNode& cond, const BBNodeVec& thn,
		const BBNodeVec& els) {
	// Fast exits.
	if (cond == nf->getTrue()) {
		return thn;
	} else if (cond == nf->getFalse()) {
		return els;
	}

	BBNodeVec result;
	result.reserve(els.size());
	const typename BBNodeVec::const_iterator th_it_end = thn.end();
	typename BBNodeVec::const_iterator el_it = els.begin();
	for (typename BBNodeVec::const_iterator th_it = thn.begin(); th_it < th_it_end; th_it++, el_it++) {
		result.push_back(nf->CreateNode(ITE, cond, *th_it, *el_it));
	}
	return result;
}


// On some cases I suspect this is better than the new variant.
template <class BBNode, class BBNodeManagerT>
BBNode BBBVLE_variant(const BBNodeVec& left, const BBNodeVec& right, bool is_signed, BBNodeManagerASTNode* nf)
{
  // "thisbit" represents BVLE of the suffixes of the BVs
  // from that position .  if R < L, return TRUE, else if L < R
  // return FALSE, else return BVLE of lower-order bits.  MSB is
  // treated separately, because signed comparison is done by
  // complementing the MSB of each BV, then doing an unsigned
  // comparison.
  typename BBNodeVec::const_iterator lit = left.begin();
  typename BBNodeVec::const_iterator litend = left.end();
  typename BBNodeVec::const_iterator rit = right.begin();
  BBNode prevbit = nf->getTrue();
  for (; lit < litend - 1; lit++, rit++)
    {
      BBNode thisbit = nf->CreateNode(ITE, nf->CreateNode(IFF, *rit, *lit), prevbit, *rit);
      prevbit = thisbit;
    }

  // Handle MSB -- negate MSBs if signed comparison
  BBNode lmsb = *lit;
  BBNode rmsb = *rit;
  if (is_signed)
    {
      lmsb = nf->CreateNode(NOT, *lit);
      rmsb = nf->CreateNode(NOT, *rit);
    }

  BBNode msb = nf->CreateNode(ITE, nf->CreateNode(IFF, rmsb, lmsb), prevbit, rmsb);
  return msb;
}



// Workhorse for comparison routines.  This does a signed BVLE if is_signed
// is true, else it's unsigned.  All other comparison operators can be reduced
// to this by swapping args or complementing the result bit.
template <class BBNode, class BBNodeManagerT>
BBNode BitBlaster<BBNode,BBNodeManagerT>::BBBVLE(const BBNodeVec& left, const BBNodeVec& right,
		bool is_signed, bool is_bvlt) {
	typename BBNodeVec::const_reverse_iterator lit = left.rbegin();
	const typename BBNodeVec::const_reverse_iterator litend = left.rend();
	typename BBNodeVec::const_reverse_iterator rit = right.rbegin();

	BBNode this_compare_bit = is_signed ? nf->CreateNode(AND, *lit,
			nf->CreateNode(NOT,*rit)) : nf->CreateNode(AND,
					nf->CreateNode(NOT,*lit), *rit);

	BBNodeVec bit_comparisons;
	bit_comparisons.reserve(left.size() +1);

	bit_comparisons.push_back(this_compare_bit);

	//(lit IFF rit) is the same as (NOT(lit) XOR rit)
	BBNode prev_eq_bit = nf->CreateNode(XOR, nf->CreateNode(NOT,
			*lit), *rit);
	for (lit++, rit++; lit < litend; lit++, rit++) {
		this_compare_bit = nf->CreateNode(AND, nf->CreateNode(NOT,*lit),
				*rit);

		BBNode thisbit_output = nf->CreateNode(AND, this_compare_bit,
				prev_eq_bit);
		bit_comparisons.push_back(thisbit_output);

		BBNode this_eq_bit = nf->CreateNode(AND, nf->CreateNode(XOR,
				nf->CreateNode(NOT, *lit), *rit), prev_eq_bit);
		prev_eq_bit = this_eq_bit;
	}

	if (!is_bvlt) {
		bit_comparisons.push_back(prev_eq_bit);
	}

	// Either zero or one of the nodes in bit_comparisons can be true.

	BBNode output;
#ifdef CRYPTOMINISAT__2
	    if (bit_comparisons.size() > 1)
	      output = nf->CreateNode(XOR, bit_comparisons);
	    else
             output = bit_comparisons[0];
#else
			output = nf->CreateNode(OR, bit_comparisons);
#endif
	return output;
}

// Left shift  within fixed field inserting zeros at LSB.
// Writes result into first argument.
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::BBLShift(BBNodeVec& x, unsigned int shift) {
	// left shift x (destructively) within width.
	// loop backwards so that copy to self works correctly. (DON'T use STL insert!)
	for (int i =((int)x.size())-1; i >=0; i--)
	{
		if (i-(int)shift >= 0)
			x[i] = x[i-(int)shift];
		else
			x[i] = nf->getFalse(); // new LSB is zero.
	}
}

// Right shift within fixed field inserting zeros at MSB.
// Writes result into first argument.
template <class BBNode, class BBNodeManagerT>
void BitBlaster<BBNode,BBNodeManagerT>::BBRShift(BBNodeVec& x, unsigned int shift) {
	// right shift x (destructively) within width.
	const typename BBNodeVec::iterator xend = x.end();
	typename BBNodeVec::iterator xit = x.begin();
	for (; xit < xend; xit++) {
		if (xit + shift < xend)
			*xit = *(xit + shift);
		else
			*xit = nf->getFalse(); // new MSB is zero.
	}
}

// Return bit-blasted form for BVLE, BVGE, BVGT, SBLE, etc.
template <class BBNode, class BBNodeManagerT>
BBNode BitBlaster<BBNode,BBNodeManagerT>::BBcompare(const ASTNode& form, BBNodeSet& support) {
	const BBNodeVec& left = BBTerm(form[0], support);
	const BBNodeVec& right = BBTerm(form[1], support);

	const Kind k = form.GetKind();
	switch (k) {
	case BVLE: {
		return BBBVLE(left, right, false);
		break;
	}
	case BVGE: {
		return BBBVLE(right, left, false);
		break;
	}
	case BVGT: {
		return BBBVLE(right, left, false, true);
		break;
	}
	case BVLT: {
		return BBBVLE(left, right, false, true);
		break;
	}
	case BVSLE: {
		return BBBVLE(left, right, true);
		break;
	}
	case BVSGE: {
		return BBBVLE(right, left, true);
		break;
	}
	case BVSGT: {
		return nf->CreateNode(NOT, BBBVLE(left, right, true));
		break;
	}
	case BVSLT: {
		return nf->CreateNode(NOT, BBBVLE(right, left, true));
		break;
	}
	default:
		cerr << "BBCompare: Illegal kind" << form << endl;
		FatalError("", form);
	}
}

// return a vector with n copies of fillval
template <class BBNode, class BBNodeManagerT>
BBNodeVec BitBlaster<BBNode,BBNodeManagerT>::BBfill(unsigned int width, BBNode fillval) {
	BBNodeVec zvec(width, fillval);
	return zvec;
}

template <class BBNode, class BBNodeManagerT>
BBNode BitBlaster<BBNode,BBNodeManagerT>::BBEQ(const BBNodeVec& left, const BBNodeVec& right) {
	BBNodeVec andvec;
	andvec.reserve(left.size());
	typename BBNodeVec::const_iterator lit = left.begin();
	const typename BBNodeVec::const_iterator litend = left.end();
	typename BBNodeVec::const_iterator rit = right.begin();

	if (left.size() > 1) {
		for (; lit != litend; lit++, rit++) {
			BBNode biteq = nf->CreateNode(IFF, *lit, *rit);
			// fast path exit
			if (biteq == nf->getFalse()) {
				return nf->getFalse();
			} else {
				andvec.push_back(biteq);
			}
		}
		BBNode n = nf->CreateNode(AND, andvec);
		return n;
	} else
		return nf->CreateNode(IFF, *lit, *rit);
}

// This creates all the specialisations of the class that are ever needed.
template class BitBlaster<ASTNode, BBNodeManagerASTNode>;
template class BitBlaster<BBNodeAIG, BBNodeManagerAIG>;

std::ostream& operator<<(std::ostream& output, const BBNodeAIG& h)
{
  FatalError("This isn't implemented  yet sorry;");
  return output;
}

#undef BBNodeVec
#undef BBNodeVecMap
#undef BBNodeSet


} // BEEV namespace
