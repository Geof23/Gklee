#include "HashingNodeFactory.h"
#include "AST.h"
#include "../STPManager/STP.h"

using BEEV::Kind;
using BEEV::ASTInterior;
using BEEV::ASTVec;
using BEEV::ASTNode;


HashingNodeFactory::~HashingNodeFactory()
{
}

// Get structurally hashed version of the node.
BEEV::ASTNode HashingNodeFactory::CreateNode(const Kind kind,	const BEEV::ASTVec & back_children)
{
	// create a new node.  Children will be modified.
	ASTInterior *n_ptr = new ASTInterior(kind);

	ASTVec children(back_children);
	// The Bitvector solver seems to expect constants on the RHS, variables on the LHS.
	// We leave the order of equals children as we find them.
	if (BEEV::isCommutative(kind) && kind != BEEV::AND)
	{
		SortByArith(children);
	}

	// insert all of children at end of new_children.
	ASTNode n(bm.CreateInteriorNode(kind, n_ptr, children));
	return n;
}

// Create and return an ASTNode for a term
ASTNode HashingNodeFactory::CreateTerm(Kind kind, unsigned int width,const ASTVec &children)
{

	ASTNode n = CreateNode(kind, children);
	n.SetValueWidth(width);

	//by default we assume that the term is a Bitvector. If
	//necessary the indexwidth can be changed later
	n.SetIndexWidth(0);
	return n;
}


