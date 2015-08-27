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
  const string SPAWN_COLOR = "purple";

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
FlowGraph::encodeFlowNode(int fNum){
  return string("Flow_") + std::to_string( fNum ) + 
    "_" + std::to_string( counter++ );
}

void
FlowGraph::constructFlows(int size){
  flows.resize( size );
  for(auto i = 0; i < size; ++i){
    flows[i] = flow(i);
  }
}

void
FlowGraph::handleStep(flowInfo& fi){
  if( !inKernel && fi.type != enterGPU ) return;
  if( fi.type != flowMerge && lineBuffer.size() > 0 ){
    graphF << lineBuffer;
    lineBuffer = "";
  }
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
    constructFlows( fi.threads );
    flows[ 0 ].pendBr = fi.kName;
    openGraph( fi.kName );
    step("contextSwitch", "0", "true"); //artificially force a context switch to avoid instructions execed before natural context sw hit    
    break;
  case encounterRet:
    handleRet();
    break;
  case genInstruction:
    if( fi.instr.find( "br i1" ) == string::npos && //skip branch and barrier
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
    handleContextSwitch(fi);
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

  //this is a node creation method
void
FlowGraph::handleContextSwitch(flowInfo& fi){
  if( currentFlow != fi.flow ){
    if( flows[fi.flow].hitRet ) return;
    currentFlow = fi.flow;
    auto& f = flows[ fi.flow ];
    f.active = true;
    f.head = encodeFlowNode( fi.flow );
    string tcond;
    if( f.lastBarrier != ""){
      tcond = f.condition;
      f.lastBarrier = "";
    } else {
      
      tcond = fi.cond;
    }
    lineBuffer =  f.head + " [label=\"flow " + to_string( fi.flow ) + ":" + 
      tcond + "\" color=" + FLOW_COLOR + "];\n";
    f.prevCondition = f.condition;
    f.condition = fi.cond;
    if( f.pendBr.length() > 0 &&
	f.connected == true){
      connectNodes( f.pendBr, f.head );
      f.pendBr = "";
    }else{ //then this is a spawned flow and we need to make an edge from spawner to this (is it thisIndex - 1?)
      for(auto s = flows.crbegin() + (flows.size() - currentFlow);
	  s != flows.crend();
	  ++s){
	if( s->active ){
	  lineBuffer = f.head + " [label=\"flow " + 
	    to_string( fi.flow ) + ":" + 
	    s->prevCondition + "\" color=" + FLOW_COLOR + "];\n";
	  lineBuffer += s->head + " -> " + f.head + 
	    " [label = \"spawn\" color = " + SPAWN_COLOR +
	    "];\n";
	  break;
	}
      }
    }
    f.connected = false; //TODO -- we may not need this? connected
  }
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
  // output updated context switch for this node
  auto& f = flows[ fi.flow ];
  //we're assuming that the flow being merged into can't be spawned this bi
  graphF << f.head + " [label=\"flow " + to_string( fi.flow ) + ":" + 
      fi.cond + "\" color=" + FLOW_COLOR + "];\n";
  // create node for merged items, use previous condition
  for( auto flow = fi.merged.rbegin();
       *flow != fi.flow && flow != fi.merged.rend(); 
       ++flow ){
    flows[*flow].pendBr = "";
    flows[*flow].active = false;
    flows[*flow].head = encodeFlowNode( *flow );
    graphF << flows[*flow].head + " [label=\"flow " + to_string( flows[*flow].index ) + ":" + 
      flows[*flow].condition + "\" color=" + MERGE_COLOR + "];\n";

    graphF << "\"" << graphLabels[ flows[*flow].lastBarrier ] << 
      "\" -> \"" << flows[ *flow ].head << "\";" << std::endl;
    graphF << "\"" << flows[ *flow ].head << "\" -> " <<
      "\"" << flows[ fi.merged[0] ].head << "\" [label = \"merge " << 
      fi.cond << "\" color = " <<
      MERGE_COLOR << "];" << std::endl;
    flows[ fi.merged[0] ].prevCondition = fi.cond;
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
  f.hitRet = true;
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
  //create branch/barrier (terminator) node in graph, if not created
  bool barrier = cond == "";
  if( barrier ) flows[ currentFlow ].lastBarrier = instr;
  if( graphLabels.count( instr ) == 0 ){ //branch not declared in graph
    node = getGraphLabel( instr, (barrier ? "Barrier" : "Branch")); //adds the branch to the graphLabels map
    graphF << node <<  " [label = <<table>" <<
      "<tr><td>\"" + instr + "\"</td></tr>";
    if( cond == "" ){
      graphF << "</table>> color = " << BAR_COLOR << "];" << std::endl;
    }else{
      graphF << "<tr><td>\"" + cond + "\"</td></tr></table>> color = " +
	BRANCH_COLOR + "];" << std::endl;
    }
  }else{
    node = graphLabels[ instr ];
  }
  //connect current flow to terminator
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
