#include <sstream>
#include <map>
#include <algorithm>
#include <iostream>
#include <klee/FlowGraph.h>
#include <assert.h>

namespace Gklee {

  const string FLOW_COLOR = "green";
  const string BRANCH_COLOR = "yellow";
  const string BAR_COLOR = "blue";
  const string MERGE_COLOR = "red";

string
FlowGraph::getGraphLabel(const string& inst,
			const string& type){
  auto& s = graphLabels[ inst ];
  if( s.length() == 0 ){
    s =  type + "_" + to_string( counter++ );
  }
  return s;
}

string 
FlowGraph::encodeFlowNode(int fNum, const string& cond){
  return string("Flow_") + std::to_string( fNum ) + 
    "_" + std::to_string( counter++ );
}

void
FlowGraph::handleStep(flowInfo& fi){
  if( !inKernel && fi.type != enterGPU ) return;
  //TODO step dump begins
  std::cerr << "type:" << fi.type << "  flow:" << fi.flow << "  instr:" << fi.instr
	    << "  cond:" << fi.cond << " merged:";
  for( auto i: fi.merged) 
    std::cerr << i << " ";
  std::cerr << std::endl;
  //TODO step dump ends
  switch( fi.type ){
  case enterGPU:
    inKernel = true;
    flows.resize( fi.threads );
    flows[ 0 ].pendBr = fi.kName;
    openGraph( fi.kName );
    step("contextSwitch", "0", "true"); //artificially force a context switch to avoid instructions execed before natural context sw hit    
    break;
  case encounterRet:
    handleRet();
    break;
  case genInstruction:
    if( fi.instr.find( "br i1" ) == string::npos &&
	fi.instr.find( "call void @__syncthreads()" ) == string::npos ){
      curInsts.push_back( fi.instr );
    }
    break;
  case encounterBranch:
    groupTerm( fi.instr, fi.cond );
    break;
  case encounterBarrier:
    groupTerm( fi.instr );
    break;
  case contextSwitch:
    {
      auto& f = flows[ fi.flow ];
      if( f.pendBr.length() > 0 &&
	  // f.head == "" ){
	  f.connected == true){
	f.head = encodeFlowNode( fi.flow, fi.cond );
	graphF << f.head + " [label=\"flow " + to_string( fi.flow ) + ":" + fi.cond + "\" color=" + FLOW_COLOR + "];" << std::endl;
	connectNodes( f.pendBr, f.head );
	f.pendBr = "";
	f.connected = false;
      }
      if( currentFlow != fi.flow ){
	currentFlow = fi.flow;
      }
    }
    break;
  case exitGPU:
    handleExit();
    break;
  case flowMerge:
    handleMerge(fi);
  default:
    break;
  }
}

struct FlowGraph::flowInfo 
FlowGraph::decode(const string& inst, 
		  const std::vector< unsigned >& data, 
		  const string& extra){
  flowInfo fi;
  if( inst == "flowMerge" ){
    fi.type = flowMerge;
    fi.cond = extra;
    fi.flow = data[0];
    fi.merged.resize(data.size());
    std::copy_n(data.begin(), data.size(), fi.merged.begin());
  }else{
    assert( false && string(string("bad call to") + __PRETTY_FUNCTION__).c_str());
  }
  return fi;
}

struct FlowGraph::flowInfo 
FlowGraph::decode(const string& inst, const string& data, const string& extra){
  flowInfo fi;
  if ( inst == "enterGPU" ){
    fi.type = enterGPU;
    std::istringstream( data ) >> fi.threads;
    fi.kName = "kernel";
    return fi;
  }
  if( inst == "encounterRet" ){
    fi.type = encounterRet;
    return fi;
  }
  if( inst == "genInstruction" ){
    fi.type = genInstruction;
    fi.instr = data;
    return fi;
  }
  if( inst == "encounterBranch" ){
    fi.type = encounterBranch;
    fi.instr = data;
    fi.cond = extra;
    return fi;
  }
  if( inst == "encounterBarrier" ){
    fi.type = encounterBarrier;
    fi.instr = data;
    return fi;
  }
  if( inst == "contextSwitch" ){
    fi.type = contextSwitch;
    std::istringstream( data ) >> fi.flow;
    fi.cond = extra;
    return fi;
  }
  if( inst == "exitGPU" ){
    fi.type = exitGPU;
    return fi;
  }
  fi.type = other;
  return fi;
}

void
FlowGraph::handleMerge(flowInfo& fi){
  for( auto flow: fi.merged ){
    graphF << "\"" << flows[ flow ].head << "\" -> \"" << flows[ fi.flow ].head 
	   << "\" [label = \"" << fi.cond << "\" color = " << MERGE_COLOR <<
	   " ];" << std::endl;
  }
}

void
FlowGraph::openGraph(const string& kName){
  graphF << "digraph " << kName << "{" << std::endl;
  graphF << kName << ";" << std::endl;;
}

void 
FlowGraph::handleRet(){
  auto& f = flows[ currentFlow ];
  if( f.head != "" ){
    connectNodes( flows[ currentFlow ].head,
		"return" );
  }
}

void 
FlowGraph::connectNodes( const string& pre, const string& post){
  graphF << "\"" << pre << "\"" << " -> " << 
    "\"" << post << "\"";
  if( curInsts.size() > 0 ){
    graphF << " [label = <<TABLE>";
    for( auto s: curInsts ){
      graphF << "<tr><td>" << s << "</td></tr>";
    }
    graphF << "</TABLE>>];" << std::endl;
  }else{
    graphF << ";" << std::endl;
  }
  curInsts.clear();
}

void
FlowGraph::groupTerm( const string& instr, const string& cond ){
  string node;
  if( graphLabels.count( instr ) == 0 ){ //branch not declared in graph
    node = getGraphLabel( instr, (cond == "" ? "Barrier" : "Branch")); //adds the branch to the graphLabels map
    string color = cond == "" ? BAR_COLOR : BRANCH_COLOR;
    graphF << node <<  " [label = <<table>" <<
      "<tr><td>\"" + instr + "\"</td></tr>";
    if( cond == "" ){
      graphF << "</table>>];" << std::endl;
    }else{
      graphF << "<tr><td>\"" + cond + "\"</td></tr></table>> color = " +
	color + "];" << std::endl;
    }
  }else{
    node = graphLabels[ instr ];
  }
  if( flows[ currentFlow ].connected == false &&
      flows[ currentFlow ].head != "" ){
    connectNodes( flows[currentFlow].head, node );
    flows[ currentFlow ].connected = true;
    //    flows[ currentFlow ].head = "";
    curInsts.clear();  
  }
  flows[ currentFlow ].pendBr = node;
}

void
FlowGraph::handleExit(){
  graphF << "}" << std::endl;
  inKernel = false;
}
}
