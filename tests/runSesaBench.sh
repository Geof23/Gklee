#!/bin/bash
#This is used to run the SesaBench suite

function checkRes {
   if [ "$1" != 0 ]
   then
       echo "$SECTION $2 failure"
#       exit "$1"
   fi
}

echo "Beginning $0 at $(pwd) on $(hostname)"

GKP=gklee_p.results
SE=gklee_sesa.results
CON=concrete
SYM=symbolic

#get SesaBench
SECTION="setup SESABench"
if [[ ! -e SESABench ]]
then
    git clone https://github.com/PengPengHub/SESABench.git
    checkRes $? 
    cd SESABench
    checkRes $?
else
    cd SESABench
    checkRes $?
    git pull
    checkRes $?
fi

echo "Setup completed"
    
TESTHOME=$(pwd)

#LoneStar
##################
SECTION=BFS_ATOMIC
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/bfs/BFS_ATOMIC
checkRes $? cd

gklee-nvcc -o bfs_atomic main.cu -I ../../../include -DVARIANT=BFS_ATOMIC 
checkRes $? build-concrete

sesa < bfs_atomic > bfs_atomic.new 2>&1
checkRes $? sesa

gklee --symbolic-config --max-sym-array-size=1024 bfs_atomic.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$CON
checkRes $?
gklee --symbolic-config --race-prune --max-sym-array-size=1024 bfs_atomic.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$CON
checkRes $?

gklee-nvcc -o bfs_atomic main.cu -I ../../../include -DVARIANT=BFS_ATOMIC -D_SYM
checkRes $? build-sym

sesa < bfs_atomic > bfs_atomic.new 2>&1
checkRes $? sesa

gklee --symbolic-config --max-sym-array-size=1024 --avoid-oob-check bfs_atomic.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
checkRes $?

gklee --symbolic-config --race-prune --max-sym-array-size=1024 --avoid-oob-check bfs_atomic.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM

##############
SECTION=BFS_LS
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/bfs/BFS_LS
checkRes $? cd

gklee-nvcc -o bfs_ls main.cu -I ../../../include -DVARIANT=BFS_LS
checkRes $? build-concrete

sesa < bfs_ls > bfs_ls.new 2>&1
checkRes $? sesa

gklee --symbolic-config bfs_ls.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$CON
gklee --symbolic-config --race-prune bfs_ls.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$CON

gklee-nvcc -o bfs_ls main.cu -I ../../../include -DVARIANT=BFS_LS -D_SYM
checkRes $? build-sym

sesa < bfs_ls > bfs_ls.new 2>&1
checkRes $? sesa

gklee --symbolic-config --avoid-oob-check bfs_ls.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
gklee --symbolic-config --race-prune --avoid-oob-check bfs_ls.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM

#####################
SECTION=BFS_WORKLISTA
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/bfs/BFS_WORKLISTA
checkRes $? cd

gklee-nvcc -o bfs_worklista main.cu -I ../../../include -DVARIANT=BFS_WORKLISTA
checkRes $? build-concrete

sesa < bfs_worklista > bfs_worklista.new 2>&1
checkRes $? sesa

gklee --symbolic-config --max-sym-array-size=1024 bfs_worklista.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$CON
gklee --symbolic-config --race-prune --max-sym-array-size=1024 bfs_worklista.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$CON

gklee-nvcc -o bfs_worklista main.cu -I ../../../include -DVARIANT=BFS_WORKLISTA -D_SYM
checkRes $? build-sym

sesa < bfs_worklista > bfs_worklista.new 2>&1
checkRes $? sesa

gklee --symbolic-config --max-sym-array-size=1024 --avoid-oob-check bfs_worklista.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
gklee --symbolic-config --race-prune --max-sym-array-size=1024 --avoid-oob-check bfs_worklista.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM

##########################
SECTION=BFS_WORKLISTW
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/bfs/BFS_WORKLISTW
checkRes $? cd
gklee-nvcc -o bfs_worklistw main.cu -I ../../../include -DVARIANT=BFS_WORKLISTW
checkRes $? buld-concrete

sesa < bfs_worklistw > bfs_worklistw.new 2>&1
checkRes $?

gklee --symbolic-config --max-sym-array-size=1024 bfs_worklistw.new 2>&1 | tee  $TESTHOME/$SECTION_$GKP_$CON
gklee --symbolic-config --race-prune --max-sym-array-size=1024 bfs_worklistw.new 2>&1 | tee  $TESTHOME/$SECTION_$SE_$CON

gklee-nvcc -o bfs_worklistw main.cu -I ../../../include -DVARIANT=BFS_WORKLISTW -D_SYM
checkRes $? build-symbolic

sesa < bfs_worklistw > bfs_worklistw.new 2>&1
checkRes $?

gklee --symbolic-config --max-sym-array-size=1024 --avoid-oob-check bfs_worklistw.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
gklee --symbolic-config --race-prune --max-sym-array-size=1024 --avoid-oob-check bfs_worklistw.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM

###################
SECTION=bh
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/bh
checkRes $? cd 

gklee-nvcc -o bh main.cu -I ../../../include
checkRes $? build-concrete

sesa < bh > bh.new 2>&1
checkRes $? sesa

gklee --symbolic-config bh.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$CON
gklee --symbolic-config --race-prune bh.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$CON

gklee-nvcc -o bh main.cu -I ../../../include -D_SYM
checkRes $? build-symbolic

sesa < bh > bh.new 2>&1 
checkRes $? sesa

gklee --symbolic-config --avoid-oob-check bh.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
gklee --symbolic-config --race-prune --avoid-oob-check bh.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM

################
SECTION=SSSP_WLN
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/sssp/SSSP_WLN
checkRes $? cd 

gklee-nvcc -o sssp_wln main.cu -I ../../../include -DVARIANT=SSSP_WLN
checkRes $? build-concrete

sesa < sssp_wln > sssp_wln.new 2>&1
checkRes $? sesa

gklee --symbolic-config sssp_wln.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$CON
gklee --symbolic-config --race-prune sssp_wln.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$CON

gklee-nvcc -o sssp_wln main.cu -I ../../../include -DVARIANT=SSSP_WLN -D_SYM
checkRes $? build-symbolic

sesa < sssp_wln > sssp_wln.new 2>&1
checkRes $? sesa

gklee --symbolic-config --avoid-oob-check sssp_wln.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
gklee --symbolic-config --race-prune --avoid-oob-check sssp_wln.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM

####################
SECTION=SSSP_LS
echo "Beginning $SECTION"
cd $TESTHOME/LoneStar/apps/sssp/SSSP_LS
checkRes $? cd 

gklee-nvcc -o sssp_ls main.cu -I ../../../include -DVARIANT=SSSP_LS
checkRes $? build-concrete

sesa < sssp_ls > sssp_ls.new 2>&1
checkRes $? sesa

gklee --symbolic-config sssp_ls.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$CON
gklee --symbolic-config --race-prune sssp_ls.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$CON

gklee-nvcc -o sssp_ls main.cu -I ../../../include -DVARIANT=SSSP_LS -D_SYM
checkRes $? build-symbolic

sesa < sssp_ls > sssp_ls.new 2>&1
checkRes $? sesa

gklee --symbolic-config --avoid-oob-check sssp_ls.new 2>&1 | tee $TESTHOME/$SECTION_$GKP_$SYM
gklee --symbolic-config --race-prune --avoid-oob-check sssp_ls.new 2>&1 | tee $TESTHOME/$SECTION_$SE_$SYM


