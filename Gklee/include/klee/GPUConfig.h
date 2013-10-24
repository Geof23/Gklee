#ifndef KLEE_GPUCONFIG_H
#define KLEE_GPUCONFIG_H

#define GKLEE_INFO std::cout << "[GKLEE]: "
#define GKLEE_INFO2 std::cout << "\n[GKLEE]: "

namespace klee {

struct GPUConfig {
  static unsigned GridSize[3];
  static unsigned BlockSize[3];  // = {2, 1};   // x dimemsion and y dimension
  static unsigned block_size;    // #threads in a block 
  static unsigned num_blocks; //  = GridSize[0];   // number of blocks in the grid
  static unsigned num_threads; //  = BlockSize[0] * BlockSize[1];   // number of threads in a block

  static unsigned SymGridSize[3];
  static unsigned SymBlockSize[3] ; //  = {2, 1};   // x dimemsion and y dimension
  static unsigned SymMaxGridSize[3];
  static unsigned SymMaxBlockSize[3];
  static unsigned sym_block_size;    // #threads in a block 
  static unsigned sym_num_blocks; //  = GridSize[0];   // number of blocks in the grid
  static unsigned sym_num_threads; //  = BlockSize[0] * BlockSize[1];   // number of threads in a block

  static unsigned warpsize;
  static unsigned check_level;
  static unsigned verbose;

  enum CTYPE {
    UNKNOWN = 0,
    LOCAL,
    SHARED,
    DEVICE,
    HOST,
    CONSTANT
  };

};

} // end namespace klee

#endif
