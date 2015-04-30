#!/bin/bash

function checkRes {
   if [ "$1" != 0 ]
   then
       echo "$SECTION $2 failure"
#       exit "$1"
   fi
}

echo "Setting up Parboil tests"

cd SESABench
tar -zxf ../pb2.5driver.tgz

cd SESABench/Table-3-Parboil

TESTHOME=$(pwd)

SECTION=spmv
echo "Beginning $SECTION"

cd $TESTHOME/spmv

gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.o -D_SYM
klee-lcc -o convert_dataset.o convert_dataset.c
klee-lcc -o mmio.o mmio.c
klee-l++ -o gpu_info.o gpu_info.cc
klee-l++ -o file.o file.cc
llvm-link -o spmv main.o convert_dataset.o mmio.o gpu_info.o file.o

gklee --symbolic-config spmv 2>&1 | tee $TESTHOME/gklee_out_$SECTION

########################################

SECTION=histo_final
echo "Beginning $SECTION"

cd $TESTHOME/histo/histo_final

gklee-nvcc histo_final.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_final.o
gklee-nvcc histo_intermediates.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_intermediates.o
gklee-nvcc histo_main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_main.o
gklee-nvcc histo_prescan.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_prescan.o
gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.o -D_FINAL
llvm-link -o histo histo_final.o histo_intermediates.o histo_main.o histo_prescan.o main.o

#run:

gklee --symbolic-config histo 2>&1 | tee $TESTHOME/gklee_out_$SECTION

##########################
SECTION=histo_prescan
echo "Beginning $SECTION"

cd $TESTHOME/histo/histo_prescan

gklee-nvcc histo_final.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_final.o
gklee-nvcc histo_intermediates.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_intermediates.o
gklee-nvcc histo_main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_main.o
gklee-nvcc histo_prescan.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_prescan.o
gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.conc.o -D_PRESCAN
gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.sym.o -D_PRESCAN -D_SYM

llvm-link -o histo.conc histo_final.o histo_intermediates.o histo_main.o histo_prescan.o main.conc.o
llvm-link -o histo.sym histo_final.o histo_intermediates.o histo_main.o histo_prescan.o main.sym.o

sesa -scev-aa < histo.conc > histo.conc.new 2>&1
gklee --symbolic-config histo.conc.new 2>&1 $TESTHOME/gklee_out_$SECTION_conc

sesa -scev-aa < histo.sym > histo.sym.new 2>&1
gklee --symbolic-config histo.sym.new 2>&1 $TESTHOME/gklee_out_$SECTION_sym


#./histo/histo_main/README
#Build:

# SECTION=histo_main

# gklee-nvcc histo_final.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_final.o
# gklee-nvcc histo_intermediates.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_intermediates.o
# gklee-nvcc histo_main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_main.o
# gklee-nvcc histo_prescan.cu -I$TESTHOME/../parboil/common/include -O3 -c -o histo_prescan.o
# gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.o -D_INTERMEDIATE
# llvm-link -o histo histo_final.o histo_intermediates.o histo_main.o histo_prescan.o main.o

#Run:  Peng left blank -- hopefully this one doesn't have problems



##################
SECTION=mri_binning
echo "Beginning $SECTION"
cd $TESTHOME/mri-gridding/mri_binning
#./mri-gridding/mri_binning/README

gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -o main.o -D_SYM
gklee-nvcc CUDA_interface.cu -I$TESTHOME/../parboil/common/include -O3 -o CUDA_interface.o -D_BINNING 
gklee-nvcc GPU_kernels.cu -I$TESTHOME/../parboil/common/include -O3 -o GPU_kernels.o

gklee-nvcc CUDA_interface.cu -I$TESTHOME/../parboil/common/include -O3 -c -o CUDA_interface.o -D_BINNING -D_SYM
llvm-link -o mri-gridding main.o CUDA_interface.o

gklee --symbolic-config mri-gridding 2>&1 | tee $TESTHOME/gklee_out_$SECTION

########################
# SECTION=mri_reorder
# cd $TESTHOME/mri-gridding/mri_reorder
# # ./mri-gridding/mri_reorder/README
# # How to build mri-gridding:

# gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.o -D_SYM
# gklee-nvcc CUDA_interface.cu -I$TESTHOME/../parboil/common/include -O3 -c -o CUDA_interface.o -D_REORDER
# llvm-link -o mri-gridding main.o CUDA_interface.o

# #Run : not provided by Peng -- will skip for now


######################
SECTION=mri_gridding
echo "Beginning $SECTION"
cd $TESTHOME/mri-gridding/mri_gridding
#./mri-gridding/mri_gridding/README
#Build mri-gridding:

gklee-nvcc main.cu -I$TESTHOME/../parboil/common/include -O3 -c -o main.o -D_SYM
gklee-nvcc CUDA_interface.cu -I$TESTHOME/../parboil/common/include -O3 -c -o CUDA_interface.o -D_GRIDDING -D_SYM
llvm-link -o mri-gridding CUDA_interface.o main.o

#Run:
gklee --symbolic-config mri-gridding 2>&1 | tee $TESTHOME/gklee_out_$SECTION


#####################
SECTION=$TESTHOME/stencil
echo "Beginning $SECTION"
cd $TESTHOME/stencil
#./stencil/README
#Build:

gklee-nvcc -I /home/peng/  -o stencil main.cu 

#Run:

gklee --symbolic-config --max-time=7200 stencil.new 2>&1 | tee $TESTHOME/gklee_out_$SECTION

######################
SECTION=bfs
echo "Beginning $SECTION"
cd=$TESTHOME/bfs
# ./bfs/README
# Build bfs:

gklee-nvcc main.cu -I $TESTHOME/../parboil/common/include -O3 -Xptxas -dlcm=cg -o main.o -D_SYM

#GKLEE_p running:

sesa -scev-aa < main.o > main.new 2>&1
gklee --symbolic-config --max-sym-array-size=2048 main.new 2>&1 | tee $TESTHOME/gklee_out_$SECTION

######################
SECTION=cutcp
echo "Beginning $SECTION"
cd=$TESTHOME/cutcp
##  ./cutcp/README
# # How to build cutcp:

# klee-lcc -I$TESTHOME/../parboil/common/include -I/usr/local/cuda/include -c main.c -o build/main.o
# klee-lcc -I$TESTHOME/../parboil/common/include -I/usr/local/cuda/include -c readatom.c -o build/readatom.o
# klee-lcc -I$TESTHOME/../parboil/common/include -I/usr/local/cuda/include -c output.c -o build/output.o
# klee-lcc -I$TESTHOME/../parboil/common/include -I/usr/local/cuda/include -c excl.c -o build/excl.o
# klee-lcc -I$TESTHOME/../parboil/common/include -I/usr/local/cuda/include -c cutcpu.c -o build/cutcpu.o
# gklee-nvcc cutoff6overlap.cu -I$TESTHOME/../parboil/common/include -O3 -c -o build/cutoff6overlap.o -D_SYM
# llvm-link -o cutcp cutcpu.o cutoff6overlap.o excl.o main.o output.o readatom.o

# #How to run with GKLEEp: skipped by Peng


