// Abstract base class for all the node factories.
#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include <vector>
#include "ASTKind.h"

namespace BEEV
{
class ASTNode;
typedef std::vector<ASTNode> ASTVec;
extern ASTVec _empty_ASTVec;
}

using BEEV::ASTNode;
using BEEV::Kind;
using BEEV::ASTVec;
using BEEV::_empty_ASTVec;

class NodeFactory
{
public:
	virtual ~NodeFactory();

	virtual BEEV::ASTNode CreateTerm(Kind kind, unsigned int width,
				const BEEV::ASTVec &children) =0;

	virtual BEEV::ASTNode CreateArrayTerm(Kind kind, unsigned int index, unsigned int width,
				const BEEV::ASTVec &children);

	virtual BEEV::ASTNode CreateNode(Kind kind,
			const BEEV::ASTVec& children) =0;


	ASTNode CreateTerm(Kind kind, unsigned int width, const ASTNode& child0,
			const ASTVec &children = _empty_ASTVec);
	ASTNode CreateTerm(Kind kind, unsigned int width, const ASTNode& child0,
			const ASTNode& child1, const ASTVec &children = _empty_ASTVec);
	ASTNode CreateTerm(Kind kind, unsigned int width, const ASTNode& child0,
			const ASTNode& child1, const ASTNode& child2,
			const ASTVec &children = _empty_ASTVec);

	ASTNode CreateNode(Kind kind, const ASTNode& child0,
			const ASTVec & back_children = _empty_ASTVec);
	ASTNode CreateNode(Kind kind, const ASTNode& child0, const ASTNode& child1,
			const ASTVec & back_children = _empty_ASTVec);
	ASTNode	CreateNode(Kind kind, const ASTNode& child0, const ASTNode& child1,
			const ASTNode& child2, const ASTVec & back_children =
			_empty_ASTVec);


	ASTNode CreateArrayTerm(Kind kind, unsigned int index, unsigned int width, const ASTNode& child0,
			const ASTNode& child1, const ASTNode& child2,
			const ASTVec &children = _empty_ASTVec);

};

#endif
