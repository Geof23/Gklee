#include <string>
#include <vector>
#include <fstream>
#include <map>

using std::string;
using std::to_string;

namespace Gklee {

class FlowGraph {
public:
 FlowGraph(const string& file): 
  curInsts(), currentFlow(-1){ 
    graphF.open( file, std::ofstream::out |
		 std::ofstream::trunc);
    
  }
  ~FlowGraph(){graphF.close();}
  enum fType { 
    enterGPU,
    encounterRet,
    genInstruction,
    encounterBranch,
    encounterBarrier,
    contextSwitch,
    flowMerge,
    exitGPU,
    other
  };
  struct flowInfo{
  flowInfo(): type(other), kName(""), 
      instr(""), threads(0), flow(0), cond(""),
      merged(){}
    fType type;
    string kName;
    string instr;
    int threads;
    int flow;
    std::vector< unsigned > merged;
    string cond;
  };
 private:
  void handleStep(flowInfo& fi);
 public:
  template< typename T >
  void step(const string& inst, const T& data, const string& extra){
    flowInfo fi = decode( inst, data, extra );
    handleStep(fi);
  }
  bool
  isInKernel() const {
    return inKernel;
  }


 private:
  void constructFlows(int count);
  void handleContextSwitch(flowInfo& fi);
  string getGraphLabel(const string& inst, const string& cond);
  string encodeBranch(const string& inst, const string& cond);
  string encodeFlowNode(int f);
  string encodeBarrier(const string& inst);
  struct flowInfo decode(const string& inst, const string& data, const string& extra);
  struct flowInfo decode(const string& inst, 
			 const std::vector< unsigned >& data, 
			 const string& extra);
  void openGraph(const string& kName);
  void handleRet();
  void handleMerge(flowInfo& fi);
  void connectNodes(const string& pre, const string& post);
  void groupTerm( const string& instr,
		  const string& cond = string(""));
  void handleExit();

  struct flow {
    flow(int i): pendBr(""), head(""),
      connected(true), active(false),
      condition(""), index(i),
      lastBarrier(""), hitRet(false),
      prevCondition(""), lastBranch(""){}
    flow(){}
    string pendBr;
    string head;
    string condition;
    string prevCondition;
    bool connected;
    bool active;
    int index;
    string lastBarrier;
    string lastBranch;
    bool hitRet;
  };
  string lineBuffer;
  std::map< string, string > graphLabels; //maps instructions to graph node names (br & bar)
  std::vector< flow > flows;
  std::ofstream graphF;
  bool inKernel = false;
  int currentFlow;
  std::vector<string> curInsts;
  unsigned long counter = 0;
  
};
}
