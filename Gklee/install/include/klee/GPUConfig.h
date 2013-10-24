#ifndef KLEE_GPUCONFIG_H
#define KLEE_GPUCONFIG_H

#define CKLEE_INFO std::cout << "CKLEE: "
#define CKLEE_INFO2 std::cout << "\nCKLEE: "

namespace klee {

struct GPUConfig {
  static unsigned GridDim;
  static unsigned GridSize[2];
  static unsigned BlockDim;
  static unsigned BlockSize[3] ; //  = {2, 1};   // x dimemsion and y dimension
  static unsigned block_size;    // #threads in a block 
  static unsigned num_blocks; //  = GridSize[0];   // number of blocks in the grid
  static unsigned num_threads; //  = BlockSize[0] * BlockSize[1];   // number of threads in a block

  static unsigned check_level;
  static unsigned verbose;

  enum CTYPE {
    UNKNOWN = 0,
    LOCAL,
    SHARED,
    DEVICE,
    HOST
  };

};

} // end namespace klee

#endif
