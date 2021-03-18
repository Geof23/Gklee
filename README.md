## GKLEE

GKLEE is a symbolic analyser and test generator tailored for CUDA C++ programs,
it supports many CUDA intrinsic functions in kernel code and CUDA runtimes in host code.
For more details, please visit GKLEE's homepage:
www.cs.utah.edu/fv/GKLEE

Please refer to the tutorial on this wiki, located here:
https://github.com/Geof23/Gklee/wiki/flows_tutorial

This is a fork of the original Github version of GKLEE, https://github.com/PengPengHub/Gklee

### Prerequisites for GKLEE

* a *unix* type system
* CMake, version 3.3+
* boost developer libraries
* bison
* flex
* python (at least 2.7, < 3)
* perl

Many package managers have suitable versions of the prerequisits for your distro.
(see below for a quick config example)

### Building GKLEE

1. clone GKLEE
2. make an out-of-source directory to build in
3. cd _chosen build directory_
4. run cmake [path to GKLEE root]
5. build the system

Here's a quick example of config and GNU Makefile based build (on Ubuntu):

```bash
sudo apt-get install cmake libboost-all-dev bison flex python-2.7 perl
git clone git@github.com:Geof23/Gklee.git
mkdir Gklee/build
cd Gklee/build
cmake ..
make -j
```

### Running GKLEE
#### Required environment variables
Another quick example, in a *bash* shell:
```bash
	export KLEE_HOME_DIR=/Path/To/Gklee
	export PATH=$KLEE_HOME_DIR/bin:$PATH
```

After building GKLEE and setting environment variables properly, you could run GKLEE.

1. Compile your CUDA programs into LLVM bytecode with the command: 

```bash
   gklee-nvcc xxx.cu [-o executable] [other options used in nvcc ...]
```
   
   NOTE: if you do not specify a executable name, gklee-nvcc will use the prefix 
   of the CUDA program as the default name

2. Run your program with GKLEE command (normal mode -- no symbolic execution)

```bash
   gklee executable
```

... or with symbolic execution:
```bash
   gklee --symbolic-config executable
```

3. The test cases will be generated in the same dir where you execute gklee, 
   and appear as 'klee-out-[number]' and 'klee-last'.

#### TaintAnalysis: Gklee 'SESA'

taint analyser consists of three *clang* compiler passes, which perform a combination of 
use-def chain analysis and alias analysis to enable taint analysis on CUDA kernels.

To run the analyser:

```bash
	sesa < [input_llvm_bytecode] > [output_llvm_bytecode]
```

1. The taint analyser outputs informative messages on which variables shall 
be symbolized in a file named "summary.txt"

2. The taint analyser annotates the input llvm bytecode with LLVM metadata 
and generates the new llvm bytecode containing LLVM annotations that are used 
to prune redundant flows in the symbolic execution stage.

3. To run this annotated bytecode in Gklee:

```bash
	gklee --symbolic-config --race-prune [path to annotated bytecode]
```
